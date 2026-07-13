//! Functional MX-Gemmini co-model for cyclotron.
//!
//! Cyclotron models the Muon SIMT cores but historically treated the MX-Gemmini
//! accelerator as a pure timing stub (`timeflow::tensor::TensorQueue`) — stores to
//! the Gemmini MMIO block were dead writes, so a combined Muon+MX kernel produced
//! wrong numeric results. This module fills that gap: it decodes the RoCC command
//! stream the Muon kernel pushes over MMIO and executes the MX matmul directly on
//! the shared scratchpad (which, under the Radiance `use_shared_ext_mem` config, IS
//! cluster SMEM). The numerics are a bit-faithful port of spike libgemmini
//! (`mx_loop_ws_spad` + `mx_fp_math.h`); see `fp_math`.
//!
//! Integration: the device lives inside the SMEM `FlatMemory`, but is driven from the
//! LSU (`Warp::mem_response`) rather than `FlatMemory::write_impl`, because the
//! accelerator's `loop_ws` load phase DMAs A/B operands out of *global* memory and so
//! needs both memories in scope.
//!
//! A command's functional effect is applied immediately, but its cost is charged to
//! `busy_until`; the kernel's `gemmini_fence*` loops poll BUSY/OCCUPANCY, which is what
//! makes accelerator time appear in the SIMT cycle count. Untimed (functional-only) runs
//! pass `now = None` and the accelerator always reads as idle.
//!
//! Modeled datapaths, all validated bit-exact against REAL spike libgemmini
//! (`spike --extension=gemmini`, harnesses in chipyard-mx `bareMetalC/radiance_xcheck_*.c`):
//!   - matmul: fp8 e4m3 (16x16 tiles), fp4 e2m1 and fp6 e3m2 (32x32 tiles, nibble-packed;
//!     fp6 nibbles index a 16-entry LUT staged in SMEM)
//!   - loop_ws load phase (A/B DMA from DRAM), honoring the `skips` field
//!   - requant post-pass for out_fmt 0/1/2 (packed codes + e8m0 scales to DRAM)
//!   - MX softmax (I-BERT), gated by config_norm's activation MSB
//!   - MX_READ_SMEM drain, FLUSH accumulator clear
//! `act_fmt=3` (FORMAT_FULL) is an *output* format only, never an input dtype: kernels
//! always configure A/B as fp8/fp6/fp4, and spike has no actf==3 branch either. We treat it
//! as an illegal config and compute nothing, loudly (matching spike).
//! See `scripts/muon/mx_golden` in autocomp for a host-side reference implementing the same
//! (hardware) semantics.
//!
//! Address contract (GPU-local shared space, from `mxgemmini_mmio.h`):
//!   GEMMINI_CTRL = 0x0008_4000
//!     +0x00 INST   (RoCC inst word: opcode 0x7B | funct<<25)
//!     +0x08 READY  (RO: 1 = accepts new command)
//!     +0x10 RS1    (64-bit)
//!     +0x18 RS2    (64-bit)
//!     +0x20 BUSY   (RO: 0 = idle)
//!     +0x28 OCCUPANCY (RO: in-flight command count)
//! The kernel writes RS1 then RS2 then INST; we decode on the INST store, reading
//! RS1/RS2 back out of the buffer.

pub mod fp_math;
pub mod state;

/// Bit-pattern vectors captured from spike libgemmini `mx_fp_math.h` (the numeric
/// oracle). Regenerate with `scratchpad/gen_spike_vectors.cpp`. Used by the
/// bit-exactness cross-check tests in `fp_math`.
#[cfg(test)]
mod spike_vectors;

/// Small FP4/FP6 cases whose expected output came from autocomp's `mx_golden`, itself
/// bit-exact against REAL spike libgemmini. Exercises the sub-byte datapaths.
#[cfg(test)]
mod mx_fixtures;

use crate::timeq::Cycle;
use state::{AccAct, MxGemminiState};

pub const GEMMINI_CTRL_DEFAULT: u64 = 0x0008_4000;
pub const GEMMINI_MMIO_SIZE: u64 = 0x1000;

const OFF_INST: u64 = 0x00;
const OFF_READY: u64 = 0x08;
const OFF_RS1: u64 = 0x10;
const OFF_RS2: u64 = 0x18;
const OFF_BUSY: u64 = 0x20;
const OFF_OCCUPANCY: u64 = 0x28;

/// Systolic dimension (Gemmini `DIM`). Radiance MX build uses 16.
const DIM: usize = 16;
/// MX block-scaling group size.
const GROUP: usize = 32;

/// Per-block e8m0 scale-factor SRAM, in the GPU-local shared address space. Under
/// `use_shared_ext_mem` the kernel stages scales here with plain SIMT stores
/// (`mxgemm_lib.hpp::load_scale_factors`) instead of DMA'ing them from DRAM, so the
/// accelerator reads them straight out of SMEM. (mxgemmini_mmio.h)
const SF_MEM_B: usize = 0x0008_8000;
const SF_MEM_A: usize = 0x0008_A000;
/// Second scale buffer for K-tile double-buffering: odd tiles stage their e8m0
/// scales at `SF_MEM_* + SF_MEM_BUFFER_OFFSET` and set the mxquant select bits.
/// = GEMMINI_SF_MEM_SIZE(0x2000) / 4 (mxgemmini_mmio.h).
const SF_MEM_BUFFER_OFFSET: usize = 0x800;

/// FP6 dequant LUTs, also staged into SMEM by the kernel (mxgemmini_mmio.h).
/// LUT0 = weights (B), LUT1 = activations (A), LUT2 = output. Each LUT is 3 LE u32
/// words (96 bits = 16 x 6-bit E3M2 codes), so LUT `li` starts at `base + 12*li`.
const LUT0_WGT: usize = GEMMINI_CTRL_DEFAULT as usize + 0x80;
const LUT1_ACT: usize = GEMMINI_CTRL_DEFAULT as usize + 0x380;
const LUT2_OUT: usize = GEMMINI_CTRL_DEFAULT as usize + 0x680;
const LUT_STRIDE: usize = 12;

/// Host<->device window: `copy_elf` relocates GPU sections from [4 GiB, 8 GiB) down to
/// 0, and kernels hand the accelerator host-side pointers (`rad_device_to_host_address`).
const GPU_DRAM_OFFSET: u64 = 0x1_0000_0000;

// Coarse latency model, ported from spike's `GEMMINI_FAITHFUL_CYCLES` path (coefficients
// fitted against VCS cycle data; see chipyard-mx mx-autocomp/SPIKE_MODEL_CHANGES.md).
// Units are milli-cycles per unit of work.
//   compute: MACs/peak -> TI*TJ*TK*16 (fp8, 16-wide tiles, peak 256)
//                       / TI*TJ*TK*32 (sub-byte fp6/fp4, 32-wide, peak 512)
//   dma:     bytes moved
// TODO(PhaseD): re-fit against RadianceSingleClusterConfig; these came from
// RadianceGemminiOnlyConfig and are ranking-grade, not absolute.
const K_CMP_MILLI: u64 = 1804;
const K_DMA_MILLI: u64 = 143;

/// Read a milli-cycle coefficient, allowing an env override for calibration sweeps
/// against RTL (`CYCLOTRON_MXGEMMINI_KCMP`, `CYCLOTRON_MXGEMMINI_KDMA`).
fn coeff(var: &str, default: u64) -> u64 {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn k_cmp() -> u64 {
    coeff("CYCLOTRON_MXGEMMINI_KCMP", K_CMP_MILLI)
}

fn k_dma() -> u64 {
    coeff("CYCLOTRON_MXGEMMINI_KDMA", K_DMA_MILLI)
}

// RoCC funct codes (spike libgemmini gemmini.h).
const F_CONFIG: u32 = 0;
const F_FLUSH: u32 = 7;
const F_LOOP_WS: u32 = 8;
const F_LOOP_WS_CONFIG_BOUNDS: u32 = 9;
const F_LOOP_WS_CONFIG_ADDRS_AB: u32 = 10;
const F_LOOP_WS_CONFIG_STRIDES_AB: u32 = 12;
const F_LOOP_WS_CONFIG_SPAD_AB: u32 = 24;
const F_LOOP_WS_CONFIG_SPAD_C: u32 = 25;
const F_MXQUANT_CONFIG_MVOUT: u32 = 26;
const F_MX_LOAD_SCALES: u32 = 27;
const F_MX_READ_SMEM: u32 = 28;
const F_MX_LOAD_LUT: u32 = 29;

#[derive(Debug, Clone)]
pub struct MxConfig {
    pub mmio_base: u64,
    pub mmio_size: u64,
}

impl Default for MxConfig {
    fn default() -> Self {
        Self {
            mmio_base: GEMMINI_CTRL_DEFAULT,
            mmio_size: GEMMINI_MMIO_SIZE,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MxGemminiDevice {
    cfg: MxConfig,
    st: MxGemminiState,
    /// Cycle at which the accelerator finishes its outstanding work. The functional
    /// effect of a command is applied immediately, but the kernel's `gemmini_fence*`
    /// loops poll BUSY/OCCUPANCY, so this is what makes accelerator time show up in the
    /// SIMT cycle count. Zero (and any untimed run) means "always idle".
    busy_until: Cycle,
    /// (byte offset, byte length) of the accumulator region written by the last
    /// `loop_ws`. FLUSH clears it: unlike spike -- whose `mx_smem` is a private buffer it
    /// can zero wholesale -- our C region shares SMEM with the operands, so we may only
    /// clear the part we know is C, or a software-tiled matmul would pile each output
    /// block onto the previous block's residue.
    last_c_region: Option<(usize, usize)>,
}

impl MxGemminiDevice {
    pub fn new(cfg: MxConfig) -> Self {
        Self {
            cfg,
            st: MxGemminiState::new(),
            busy_until: 0,
            last_c_region: None,
        }
    }

    #[inline]
    pub fn matches_mmio(&self, addr: u64) -> bool {
        addr >= self.cfg.mmio_base && addr < self.cfg.mmio_base + self.cfg.mmio_size
    }

    /// Initialize the read-only status registers so the kernel's fence polls pass.
    /// Called once when the device is attached to a SMEM `FlatMemory`.
    pub fn init_status(&self, bytes: &mut [u8]) {
        self.refresh_status(bytes, None);
    }

    /// Publish BUSY/OCCUPANCY/READY for the current cycle. In an untimed (functional)
    /// run `now` is None and the accelerator always reads as idle.
    pub fn refresh_status(&self, bytes: &mut [u8], now: Option<Cycle>) {
        let busy = now.map_or(false, |t| t < self.busy_until);
        // READY stays 1: the command queue accepts a new command even while the array is
        // busy (the kernel uses gemmini_fence_ready to overlap issue with compute).
        write_u32(bytes, self.cfg.mmio_base + OFF_READY, 1);
        write_u32(bytes, self.cfg.mmio_base + OFF_BUSY, busy as u32);
        write_u32(bytes, self.cfg.mmio_base + OFF_OCCUPANCY, busy as u32);
    }

    /// Charge `cycles` of accelerator work, serializing behind any outstanding work.
    fn charge(&mut self, now: Option<Cycle>, cycles: u64) {
        if let Some(t) = now {
            self.busy_until = self.busy_until.max(t).saturating_add(cycles);
            if std::env::var("CYCLOTRON_MXGEMMINI_DEBUG").is_ok() {
                eprintln!("[mxgemmini] charge {cycles} @{t} -> busy_until={}", self.busy_until);
            }
        } else if std::env::var("CYCLOTRON_MXGEMMINI_DEBUG").is_ok() {
            eprintln!("[mxgemmini] charge {cycles} but now=None (untimed)");
        }
    }

    /// Called from the LSU after a store within the MMIO window has landed in `smem`.
    /// Decodes/executes when the store hits the INST reg. `gmem` is read-only: the
    /// `loop_ws` load phase DMAs A/B operand tiles out of DRAM into the scratchpad.
    pub fn on_write(&mut self, addr: u64, smem: &mut [u8], gmem: &mut [u8], now: Option<Cycle>) {
        // Only the INST-word store triggers a command; RS1/RS2 just accumulate in
        // the buffer and are read back here.
        if addr != self.cfg.mmio_base + OFF_INST {
            return;
        }
        let inst = read_u32(smem, self.cfg.mmio_base + OFF_INST);
        let funct = inst >> 25;
        let rs1 = read_u64(smem, self.cfg.mmio_base + OFF_RS1);
        let rs2 = read_u64(smem, self.cfg.mmio_base + OFF_RS2);
        if std::env::var("CYCLOTRON_MXGEMMINI_DEBUG").is_ok() {
            eprintln!("[mxgemmini] funct={funct} rs1={rs1:#x} rs2={rs2:#x}");
        }
        self.execute(funct, rs1, rs2, smem, gmem, now);
        self.refresh_status(smem, now);
    }

    fn execute(
        &mut self,
        funct: u32,
        rs1: u64,
        rs2: u64,
        smem: &mut [u8],
        gmem: &mut [u8],
        now: Option<Cycle>,
    ) {
        match funct {
            F_CONFIG => self.config(rs1, rs2),
            F_LOOP_WS_CONFIG_BOUNDS => self.loop_ws_config_bounds(rs1, rs2),
            F_LOOP_WS_CONFIG_ADDRS_AB => {
                self.st.loop_ws_a_dram = rs1;
                self.st.loop_ws_b_dram = rs2;
            }
            F_LOOP_WS_CONFIG_STRIDES_AB => {
                self.st.loop_ws_a_stride = rs1;
                self.st.loop_ws_b_stride = rs2;
            }
            F_LOOP_WS_CONFIG_SPAD_AB => {
                self.st.mx_loop_a_spad = rs1 as u32;
                self.st.mx_loop_b_spad = rs2 as u32;
            }
            F_LOOP_WS_CONFIG_SPAD_C => {
                self.st.mx_loop_c_spad = rs1 as u32;
            }
            F_MXQUANT_CONFIG_MVOUT => self.mxquant_config_mvout(rs1, rs2),
            F_MX_LOAD_SCALES => {
                // Scales are staged into SMEM (SF_MEM_A/B) by SIMT stores in the
                // shared-ext-mem deployment, so there is nothing to DMA here; the
                // matmul reads them from SMEM directly.
                let _ = (rs1, rs2);
            }
            F_MX_LOAD_LUT => {
                // Like scales, LUTs are staged into SMEM (LUT0/LUT1) by SIMT stores in
                // the shared-ext-mem deployment, so there is nothing to DMA here.
                let _ = (rs1, rs2);
            }
            F_LOOP_WS => {
                // rs2[9] marks the MX-spad variant; the Radiance kernel always sets it.
                if (rs2 >> 9) & 0x1 == 0 {
                    return;
                }
                // skips = (lda | ldb<<1 | ldd<<2 | ex<<3 | stc<<4) << 3  (gemmini.h
                // loop_matmul_skips). The Radiance kernel issues loop_ws TWICE: once
                // load-only (skip_ex|skip_stc) to DMA A/B into the scratchpad, then once
                // compute-only (skip_lda|skip_ldb). Honoring these is mandatory here —
                // unlike spike, whose `mx_smem` is a separate buffer, our C region lives
                // in the same SMEM as the operands, so computing during the load-only
                // call would write C over the A tile at spad 0.
                let skips = (rs2 >> 3) & 0x1F;
                let (skip_lda, skip_ldb, skip_ex) =
                    (skips & 1 != 0, skips & 2 != 0, skips & 8 != 0);
                let (ti, tj, tk) = (
                    self.st.loop_ws_i as u64,
                    self.st.loop_ws_j as u64,
                    self.st.loop_ws_k as u64,
                );
                if !skip_lda {
                    self.dma_load_a(smem, &*gmem);
                    let bytes = ti * tk * (DIM * DIM) as u64;
                    self.charge(now, bytes * k_dma() / 1000);
                }
                if !skip_ldb {
                    self.dma_load_b(smem, &*gmem);
                    let bytes = tk * tj * (DIM * DIM) as u64;
                    self.charge(now, bytes * k_dma() / 1000);
                }
                if !skip_ex {
                    self.mx_loop_ws_spad(rs1, rs2, smem, gmem);
                    // MACs/peak: 16 for fp8 (16-wide tiles), 32 for sub-byte fp6/fp4.
                    let per_tile = if self.st.mx_act_fmt == 0 { 16 } else { 32 };
                    self.charge(now, ti * tj * tk * per_tile * k_cmp() / 1000);
                }
            }
            F_MX_READ_SMEM => {
                // Drain the accumulator to DRAM: rs1 = dram addr, rs2 = offset | count<<32.
                // The offset the kernel passes is `SPAD_DEST * DIM`, which is exactly our C
                // byte base (spike reads it as an element index into its private mx_smem;
                // both denote the same place). Each element is 2 bytes.
                let dram = gpu_dram(rs1);
                let off = (rs2 & 0xFFFF_FFFF) as usize;
                let count = ((rs2 >> 32) & 0xFFFF_FFFF) as usize;
                for e in 0..count {
                    let v = read_bf16_at(smem, off + e * 2);
                    let le = v.to_le_bytes();
                    if let Some(dst) = gmem.get_mut(dram + e * 2..dram + e * 2 + 2) {
                        dst.copy_from_slice(&le);
                    }
                }
            }
            F_FLUSH => {
                // Reset accelerator state between matmuls. The bf16 accumulator must be
                // cleared, else a software-tiled matmul (one loop_ws per output block)
                // accumulates each block onto the previous block's residue.
                if let Some((base, len)) = self.last_c_region.take() {
                    let end = (base + len).min(smem.len());
                    if base < end {
                        smem[base..end].fill(0);
                    }
                }
            }
            _ => {}
        }
    }

    // --- loop_ws load phase: DMA operand tiles DRAM -> scratchpad (== SMEM) ---
    //
    // Mirrors gemmini's LoopMatmul mvin of A/B. FP8 = 1 byte/elem, so a spad row is
    // DIM bytes and holds one 16-wide slice of a matrix row.

    fn dma_load_a(&self, smem: &mut [u8], gmem: &[u8]) {
        let (ti, tk) = (self.st.loop_ws_i as usize, self.st.loop_ws_k as usize);
        let a_sp = self.st.mx_loop_a_spad as usize;
        let stride = self.st.loop_ws_a_stride as usize;
        let base = gpu_dram(self.st.loop_ws_a_dram);
        for i in 0..ti {
            for k in 0..tk {
                let spad_row = a_sp + (i * tk + k) * DIM;
                for r in 0..DIM {
                    let src = base + (i * DIM + r) * stride + k * DIM;
                    let dst = (spad_row + r) * DIM;
                    copy_row(smem, dst, gmem, src, DIM);
                }
            }
        }
    }

    fn dma_load_b(&self, smem: &mut [u8], gmem: &[u8]) {
        let (tj, tk) = (self.st.loop_ws_j as usize, self.st.loop_ws_k as usize);
        // The kernel configures the *end* of the B region; tiles start TK*TJ*DIM below.
        let b_sp = (self.st.mx_loop_b_spad as usize).wrapping_sub(tk * tj * DIM);
        let stride = self.st.loop_ws_b_stride as usize;
        let base = gpu_dram(self.st.loop_ws_b_dram);
        for k in 0..tk {
            for j in 0..tj {
                let spad_row = b_sp + (k * tj + j) * DIM;
                for r in 0..DIM {
                    let src = base + (k * DIM + r) * stride + j * DIM;
                    let dst = (spad_row + r) * DIM;
                    copy_row(smem, dst, gmem, src, DIM);
                }
            }
        }
    }

    // --- config handlers (subset of spike gemmini_t::config / *_config_*) ---

    fn config(&mut self, rs1: u64, rs2: u64) {
        let _ = rs2;
        match rs1 & 0b11 {
            0 => {
                // config_ex
                let set_only_strides = (rs1 >> 7) & 0x1 != 0;
                // acc activation: rs1[4:3]==0 keeps Softmax if it was already selected.
                if !set_only_strides {
                    self.st.mx_use_lut = (rs1 >> 5) & 0x1 != 0;
                    self.st.mx_act_fmt = ((rs1 >> 10) & 0x3) as u8;
                    self.st.mx_wgt_fmt = ((rs1 >> 12) & 0x3) as u8;
                    self.st.mx_out_fmt = ((rs1 >> 14) & 0x3) as u8;
                }
            }
            3 => {
                // config_norm: I-BERT constants + the activation MSB. Setting that MSB is
                // the ONLY way acc_act becomes SOFTMAX(4) -- config_mvout can only encode
                // 0..3 -- so the MX softmax pass is gated here.
                if (rs1 >> 17) & 1 == 0 {
                    self.st.igelu_qb = (rs2 & 0xFFFF_FFFF) as u32;
                    self.st.igelu_qc = ((rs2 >> 32) & 0xFFFF_FFFF) as u32;
                    let act_msb = (rs1 >> 16) & 0x1;
                    if act_msb != 0 {
                        self.st.acc_act = AccAct::Softmax;
                    }
                    if (rs1 >> 18) & 1 != 0 {
                        self.st.qln2_inv = ((rs1 >> 32) & 0xFFFF_FFFF) as u32;
                    } else {
                        self.st.qln2 = ((rs1 >> 32) & 0xFFFF_FFFF) as u32;
                    }
                }
            }
            2 => {
                // config_mvout: acc activation
                let sel = (rs1 >> 2) & 0b11;
                self.st.acc_act = match sel {
                    0 => {
                        if self.st.acc_act == AccAct::Softmax {
                            AccAct::Softmax
                        } else {
                            AccAct::None
                        }
                    }
                    1 => AccAct::Relu,
                    2 => AccAct::LayerNorm,
                    3 => AccAct::IGelu,
                    _ => AccAct::None,
                };
            }
            _ => {}
        }
    }

    fn loop_ws_config_bounds(&mut self, rs1: u64, rs2: u64) {
        self.st.loop_ws_i = (rs2 & 0xFFFF) as u16;
        self.st.loop_ws_j = ((rs2 >> 16) & 0xFFFF) as u16;
        self.st.loop_ws_k = ((rs2 >> 32) & 0xFFFF) as u16;
        self.st.loop_ws_pad_i = (rs1 & 0xFFFF) as u16;
        self.st.loop_ws_pad_j = ((rs1 >> 16) & 0xFFFF) as u16;
        self.st.loop_ws_pad_k = ((rs1 >> 32) & 0xFFFF) as u16;
    }

    fn mxquant_config_mvout(&mut self, rs1: u64, rs2: u64) {
        self.st.mx_scale_dram = rs1 & 0x1_FFFF_FFFF;
        self.st.mx_tiles_i = ((rs1 >> 33) & 0x1FF) as u32;
        self.st.mx_tiles_j = ((rs1 >> 42) & 0x1FF) as u32;
        self.st.mx_tiles_k = ((rs1 >> 51) & 0x1FF) as u32;
        self.st.mx_lut_update_granularity = (rs2 & 0xFFFF) as u32;
        // A/B scale-SRAM double-buffer selects. The kernel toggles these per K-tile
        // (odd_k) so the accelerator reads the correct SF buffer; see mx_loop_ws_spad.
        self.st.mx_scale_act_sel = (rs1 >> 60) & 0x1 != 0;
        self.st.mx_scale_wgt_sel = (rs1 >> 61) & 0x1 != 0;
    }

    // --- MX matmul: port of spike mx_loop_ws_spad, all three activation formats ---
    //
    // FP8 (actf=0): 16x16 tiles, one fp8 e4m3 code per spad byte.
    // FP4 (actf=2): 32x32 tiles, two e2m1 codes nibble-packed per byte -- along M for A,
    //               along N for B.
    // FP6 (actf=1): 32x32 tiles, same nibble packing, but each nibble is an *index* into
    //               a 16-entry LUT of 6-bit e3m2 codes. LUTs are staged in SMEM by the
    //               kernel (LUT0 = weights/B, LUT1 = activations/A); which LUT a row/col
    //               uses is `(global_index >> lut_update_granularity)`.

    fn mx_loop_ws_spad(&mut self, _rs1: u64, rs2: u64, smem: &mut [u8], gmem: &mut [u8]) {
        let c_spad = ((rs2 >> 32) & 0xFFFF_FFFF) as u32;
        let ti = self.st.loop_ws_i as usize;
        let tj = self.st.loop_ws_j as usize;
        let tk = self.st.loop_ws_k as usize;
        let actf = self.st.mx_act_fmt;

        // Per-column accumulator precision schedule (spike acc_e/acc_m).
        const ACC_E: [i32; 16] = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8];
        const ACC_M: [i32; 16] = [4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 6, 7];
        const PROD_E: i32 = 4;
        const PROD_M: i32 = 3;

        // C is bf16 in SMEM at BYTE base `C_spad * DIM` (the kernel reads it as
        // `reinterpret_cast<__shared uint8_t*>(SPAD_DEST * DIM)`, mxgemm_lib.hpp),
        // element (m,n) at byte `c_base + (m*N + n)*2`. Note the base is a byte
        // offset, not a bf16-element index — do NOT scale it by 2.
        let c_base = c_spad as usize * DIM;

        if actf == 3 {
            // Not a modeling gap: FORMAT_FULL(3) is an *output* format only. The A/B dtype
            // is always fp8/fp6/fp4 (mxgemm_lib.hpp::configure_mxgemmini), and spike's
            // mx_loop_ws_spad likewise has no actf==3 branch -- it falls through computing
            // nothing. Mirror that, loudly.
            warn_once(
                "illegal config: act_fmt=3 (FORMAT_FULL) is an output format, not an input \
                 dtype; no matmul performed (matches spike)",
            );
            return;
        }

        // Sub-byte formats process 32-wide tiles; fp8 processes 16.
        let sub_byte = actf != 0;
        let tile = if sub_byte { 32 } else { DIM };

        let a_sp = self.st.mx_loop_a_spad as usize;
        let b_sp = (self.st.mx_loop_b_spad as usize).wrapping_sub(tk * tj * DIM);
        let m_dim = ti * tile;
        let n_dim = tj * tile;

        // FP6 LUTs, staged in SMEM by the kernel. One LUT per `1 << G` rows/cols.
        let g = self.st.mx_lut_update_granularity as usize;
        let (lut_a, lut_b) = if actf == 1 {
            (
                read_luts(smem, LUT1_ACT, (m_dim >> g) + 1),
                read_luts(smem, LUT0_WGT, (n_dim >> g) + 1),
            )
        } else {
            (Vec::new(), Vec::new())
        };

        // Decode one A element (row m of the tile) / B element (col n of the tile).
        let dec_a = |smem: &[u8], a_t: usize, m: usize, kk: usize, i: usize| -> f32 {
            match actf {
                0 => fp_math::fp8_e4m3_decode(spad_u8(smem, a_t + m, kk)),
                _ => {
                    let byte = spad_u8(smem, a_t + (m >> 1), kk);
                    let nib = if m & 1 == 1 { (byte >> 4) & 0xF } else { byte & 0xF };
                    if actf == 2 {
                        fp_math::fp4_e2m1_decode(nib)
                    } else {
                        let li = (i * tile + m) >> g;
                        let code = lut_a.get(li * 16 + nib as usize).copied().unwrap_or(0);
                        fp_math::fp6_e3m2_decode(code)
                    }
                }
            }
        };
        let dec_b = |smem: &[u8], b_t: usize, n: usize, kk: usize, j: usize| -> f32 {
            match actf {
                0 => fp_math::fp8_e4m3_decode(spad_u8(smem, b_t + kk, n)),
                _ => {
                    let byte = spad_u8(smem, b_t + kk, n >> 1);
                    let nib = if n & 1 == 1 { (byte >> 4) & 0xF } else { byte & 0xF };
                    if actf == 2 {
                        fp_math::fp4_e2m1_decode(nib)
                    } else {
                        let li = (j * tile + n) >> g;
                        let code = lut_b.get(li * 16 + nib as usize).copied().unwrap_or(0);
                        fp_math::fp6_e3m2_decode(code)
                    }
                }
            }
        };

        for k_outer in 0..tk {
            let group = (k_outer * DIM) / GROUP;
            for j in 0..tj {
                for i in 0..ti {
                    let a_t = a_sp + (i * tk + k_outer) * DIM;
                    let b_t = b_sp + (k_outer * tj + j) * DIM;
                    let mut ct = vec![[0.0f32; 32]; 32];
                    for kk in 0..DIM {
                        let mut a_col = [0.0f32; 32];
                        let mut b_row = [0.0f32; 32];
                        for r in 0..tile {
                            a_col[r] = dec_a(smem, a_t, r, kk, i);
                        }
                        for c in 0..tile {
                            b_row[c] = dec_b(smem, b_t, c, kk, j);
                        }
                        let ae = ACC_E[kk];
                        let am = ACC_M[kk];
                        for r in 0..tile {
                            for c in 0..tile {
                                let p = fp_math::mx_product_quantize_trunc(
                                    a_col[r] * b_row[c],
                                    PROD_E,
                                    PROD_M,
                                );
                                let cq = fp_math::fp_quantize_rne(ct[r][c], ae, am);
                                let pq = fp_math::fp_quantize_rne(p, ae, am);
                                ct[r][c] = fp_math::fp_add_exact(cq, pq, ae, am);
                            }
                        }
                    }
                    for r in 0..tile {
                        for c in 0..tile {
                            let a_off = group * m_dim + (i * tile + r);
                            let b_off = group * n_dim + (j * tile + c);
                            // e8m0 scales staged in SMEM by the kernel (see SF_MEM_*).
                            // For K-tile double-buffering the kernel puts odd tiles'
                            // scales in the second buffer and sets the mxquant select
                            // bits; honor them or every odd tile reads stale scales.
                            let a_buf = if self.st.mx_scale_act_sel { SF_MEM_BUFFER_OFFSET } else { 0 };
                            let b_buf = if self.st.mx_scale_wgt_sel { SF_MEM_BUFFER_OFFSET } else { 0 };
                            let sa = smem.get(SF_MEM_A + a_buf + a_off).copied().unwrap_or(0x7f);
                            let sb = smem.get(SF_MEM_B + b_buf + b_off).copied().unwrap_or(0x7f);
                            let e = sa as i32 + sb as i32 - 127;
                            let s_code = e.clamp(0, 254) as u8;
                            let s = fp_math::fpe8m0_decode(s_code);
                            let scaled = fp_math::bf16_round(ct[r][c] * s);
                            let local = (i * tile + r) * n_dim + (j * tile + c);
                            let byte = c_base + local * 2;
                            let prev = fp_math::bf16_to_f32(read_bf16_at(smem, byte));
                            let acc = fp_math::bf16_accum_add(prev, scaled);
                            write_bf16_at(smem, byte, fp_math::f32_to_bf16_rne(acc));
                        }
                    }
                }
            }
        }

        // Remember the accumulator extent so FLUSH can clear it (see `flush`).
        self.last_c_region = Some((c_base, m_dim * n_dim * 2));

        // HW MX SOFTMAX pass: the matmul output is an attention score row; apply the
        // I-BERT integer softmax in place, per output row, BEFORE requant -- so requant
        // then encodes the probabilities directly. (spike gemmini.cc:1362)
        if self.st.acc_act == AccAct::Softmax {
            self.mx_softmax(smem, c_base, m_dim, n_dim);
        }

        // Requant post-pass: replaces the BF16 tile in place with packed codes and emits
        // per-row per-N-group e8m0 scale codes to user DRAM. out_fmt 3 == full/bf16.
        if self.st.mx_out_fmt != 3 {
            self.mx_requant(smem, gmem, c_base, m_dim, n_dim);
        }
    }

    /// I-BERT integer softmax over each output row (spike gemmini.cc:1375).
    fn mx_softmax(&self, smem: &mut [u8], c_base: usize, m_dim: usize, n_dim: usize) {
        // Constants come from config_norm; fall back to the validated set when unset.
        let qln2 = if self.st.qln2 != 0 { self.st.qln2 as i64 } else { 710 };
        let qln2_inv = if self.st.qln2_inv != 0 { self.st.qln2_inv as i64 } else { 92 };
        let qb = if self.st.igelu_qb != 0 { self.st.igelu_qb as i64 } else { 1385 };
        let qc = if self.st.igelu_qc != 0 { self.st.igelu_qc as i64 } else { 1_006_165 };
        let iexp = |qa: i64| -> i64 {
            // qa <= 0
            let z = (-qa * qln2_inv) >> 16;
            let qp = qa + z * qln2;
            let qe = (qp + qb) * (qp + qb) + qc;
            if z >= 63 {
                return 0; // shift-UB guard (validated fix)
            }
            qe >> z
        };
        for m in 0..m_dim {
            let mut q = vec![0i64; n_dim];
            let mut mq = i64::MIN / 4;
            for (j, qj) in q.iter_mut().enumerate() {
                let v = fp_math::bf16_to_f32(read_bf16_at(smem, c_base + (m * n_dim + j) * 2));
                // Q(1/1024), matches mx_ibf16_to_q
                *qj = (v * 1024.0).round() as i64;
                if *qj > mq {
                    mq = *qj;
                }
            }
            let p: Vec<i64> = q.iter().map(|&qq| iexp(qq - mq)).collect();
            let sum: i64 = p.iter().sum::<i64>().max(1);
            for (j, &pv) in p.iter().enumerate() {
                let prob = pv as f32 / sum as f32;
                write_bf16_at(smem, c_base + (m * n_dim + j) * 2, fp_math::f32_to_bf16_rne(prob));
            }
        }
    }

    /// Requant post-pass for out_fmt 0 (fp8), 1 (fp6, via the output LUT), 2 (fp4).
    /// Rewrites the C tile in place with packed codes and writes the per-block e8m0
    /// scale codes to DRAM at `mx_scale_dram`. (spike gemmini.cc:1403/1524/1634)
    ///
    /// Byte layout inside the C region: fp8 packs 1 code per element along N, so byte
    /// `(m*N + j)`. fp6/fp4 pack 2 codes per byte along M ("HW-tiled"), so byte
    /// `((m/2)*N + j)`, low nibble for even m.
    fn mx_requant(
        &self,
        smem: &mut [u8],
        gmem: &mut [u8],
        c_base: usize,
        m_dim: usize,
        n_dim: usize,
    ) {
        const GROUP_OUT: usize = 32;
        let out_fmt = self.st.mx_out_fmt;
        let n_blocks = n_dim / GROUP_OUT;
        let scale_dram = self.st.mx_scale_dram;
        let g = self.st.mx_lut_update_granularity as usize;
        // fp6 projects onto a per-row output LUT (LUT2).
        let lut_c = if out_fmt == 1 {
            read_luts(smem, LUT2_OUT, (m_dim >> g) + 1)
        } else {
            Vec::new()
        };
        // Group max targets: fp8 -> 2^8 (near fp8 max), unless this is a softmax result
        // that feeds a following MX matmul, which must land at 2^4 or it overflows the
        // e4 block accumulator. fp6 emax = 4, fp4 emax = 2.
        let log2_pmax: i32 = match out_fmt {
            0 => {
                if self.st.acc_act == AccAct::Softmax {
                    4
                } else {
                    8
                }
            }
            1 => 4,
            _ => 2,
        };

        for m in 0..m_dim {
            for bi in 0..n_blocks {
                let mut vals = [0.0f32; GROUP_OUT];
                let mut max_abs = 0.0f32;
                for (jj, v) in vals.iter_mut().enumerate() {
                    let j = bi * GROUP_OUT + jj;
                    *v = fp_math::bf16_to_f32(read_bf16_at(smem, c_base + (m * n_dim + j) * 2));
                    if v.abs() > max_abs {
                        max_abs = v.abs();
                    }
                }
                let scale_code = if max_abs == 0.0 {
                    0u8
                } else {
                    // Widen before the arithmetic: a saturated accumulator makes max_abs inf,
                    // and `inf.log2() as i32` saturates to i32::MAX, overflowing the add.
                    // (Spike's `(int)floorf(log2f(inf))` is UB here; clamping is the sane read.)
                    let max_exp = if max_abs.is_finite() {
                        max_abs.log2().floor() as i64
                    } else {
                        i64::from(i32::MAX)
                    };
                    (max_exp - i64::from(log2_pmax) + 127).clamp(0, 254) as u8
                };
                if scale_dram != 0 {
                    let off = gpu_dram(scale_dram) + m * n_blocks + bi;
                    if let Some(b) = gmem.get_mut(off) {
                        *b = scale_code;
                    }
                }
                let scale = fp_math::fpe8m0_decode(scale_code);
                for (jj, &v) in vals.iter().enumerate() {
                    let j = bi * GROUP_OUT + jj;
                    if out_fmt == 0 {
                        let code = fp_math::fp8_e4m3_to_code(v / scale);
                        write_code_byte(smem, c_base + m * n_dim + j, code);
                    } else {
                        let scaled_bf16 = fp_math::f32_to_bf16_rne(v / scale);
                        let code = if out_fmt == 1 {
                            let fp6 = fp_math::bf16_bits_to_fp6_e3m2_code(scaled_bf16);
                            let li = m >> g;
                            let lut = lut_c.get(li * 16..li * 16 + 16).unwrap_or(&[0u8; 16]);
                            fp_math::fp6e3m2_nearest_finder(fp6, lut) as u8
                        } else {
                            fp_math::bf16_bits_to_fp4_e2m1_code(scaled_bf16)
                        };
                        // HW-tiled nibble packing along M
                        let byte_pos = c_base + (m >> 1) * n_dim + j;
                        let cur = smem.get(byte_pos).copied().unwrap_or(0);
                        let byte = if m & 1 == 1 {
                            (cur & 0x0F) | ((code & 0xF) << 4)
                        } else {
                            (cur & 0xF0) | (code & 0xF)
                        };
                        write_code_byte(smem, byte_pos, byte);
                    }
                }
            }
        }
    }
}

// --- SMEM byte access helpers ---
//
// The Radiance shared-ext-mem config maps the Gemmini scratchpad onto cluster
// SMEM: spad row `r` (of DIM elements) begins at SMEM byte `r * DIM` (FP8 = 1
// byte/elem). The bf16 accumulator (`mx_smem`) is addressed by element index into
// the same SMEM, at byte `idx * 2`.

/// Map a host-side DRAM pointer handed to the accelerator into cyclotron's gmem,
/// matching `FlatMemory::copy_elf`'s GPU relocation.
#[inline]
fn gpu_dram(addr: u64) -> usize {
    (if addr >= GPU_DRAM_OFFSET {
        addr - GPU_DRAM_OFFSET
    } else {
        addr
    }) as usize
}

/// Read `count` FP6 LUTs (16 six-bit codes each) staged in SMEM at `base`.
fn read_luts(smem: &[u8], base: usize, count: usize) -> Vec<u8> {
    let mut out = vec![0u8; count * 16];
    for li in 0..count {
        let mut dwords = [0u32; 3];
        for (w, dw) in dwords.iter_mut().enumerate() {
            *dw = read_u32(smem, (base + li * LUT_STRIDE + w * 4) as u64);
        }
        let mut codes = [0u8; 16];
        fp_math::unpack_lut_96bit(dwords, &mut codes);
        out[li * 16..li * 16 + 16].copy_from_slice(&codes);
    }
    out
}

/// Emit a modeling-gap warning exactly once, so an unmodeled datapath is loud rather
/// than silently producing wrong results.
fn warn_once(msg: &str) {
    use std::sync::Mutex;
    use std::sync::OnceLock;
    static SEEN: OnceLock<Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    if seen.lock().unwrap().insert(msg.to_string()) {
        eprintln!("[mxgemmini] WARNING: {msg}");
    }
}

/// Copy `n` bytes gmem[src..] -> smem[dst..], bounds-checked (out-of-range reads as 0).
#[inline]
fn copy_row(smem: &mut [u8], dst: usize, gmem: &[u8], src: usize, n: usize) {
    if dst + n > smem.len() {
        return;
    }
    for i in 0..n {
        smem[dst + i] = gmem.get(src + i).copied().unwrap_or(0);
    }
}

#[inline]
fn spad_u8(bytes: &[u8], row: usize, col: usize) -> u8 {
    let off = row * DIM + col;
    bytes.get(off).copied().unwrap_or(0)
}

/// Write one packed requant code byte into the C region, bounds-checked.
#[inline]
fn write_code_byte(bytes: &mut [u8], off: usize, v: u8) {
    if let Some(b) = bytes.get_mut(off) {
        *b = v;
    }
}

#[inline]
fn read_bf16_at(bytes: &[u8], off: usize) -> u16 {
    if off + 1 < bytes.len() {
        u16::from_le_bytes([bytes[off], bytes[off + 1]])
    } else {
        0
    }
}

#[inline]
fn write_bf16_at(bytes: &mut [u8], off: usize, v: u16) {
    if off + 1 < bytes.len() {
        let le = v.to_le_bytes();
        bytes[off] = le[0];
        bytes[off + 1] = le[1];
    }
}

#[inline]
fn read_u32(bytes: &[u8], addr: u64) -> u32 {
    let a = addr as usize;
    if a + 3 < bytes.len() {
        u32::from_le_bytes([bytes[a], bytes[a + 1], bytes[a + 2], bytes[a + 3]])
    } else {
        0
    }
}

#[inline]
fn read_u64(bytes: &[u8], addr: u64) -> u64 {
    let lo = read_u32(bytes, addr) as u64;
    let hi = read_u32(bytes, addr + 4) as u64;
    lo | (hi << 32)
}

#[inline]
fn write_u32(bytes: &mut [u8], addr: u64, v: u32) {
    let a = addr as usize;
    if a + 3 < bytes.len() {
        let le = v.to_le_bytes();
        bytes[a..a + 4].copy_from_slice(&le);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: u64 = GEMMINI_CTRL_DEFAULT;
    const FP8_ONE: u8 = 0x38; // fp8 E4M3 code for 1.0

    /// Issue one RoCC command through the real MMIO path (RS1, RS2, then INST).
    /// These tests stage operands directly in SMEM, so gmem is unused (empty).
    fn cmd(dev: &mut MxGemminiDevice, bytes: &mut [u8], funct: u32, rs1: u64, rs2: u64) {
        write_u32(bytes, BASE + OFF_RS1, rs1 as u32);
        write_u32(bytes, BASE + OFF_RS1 + 4, (rs1 >> 32) as u32);
        write_u32(bytes, BASE + OFF_RS2, rs2 as u32);
        write_u32(bytes, BASE + OFF_RS2 + 4, (rs2 >> 32) as u32);
        write_u32(bytes, BASE + OFF_INST, funct << 25);
        dev.on_write(BASE + OFF_INST, bytes, &mut [], None);
    }

    /// LOOP_WS rs2: mx_spad marker (bit9) + skip_lda|skip_ldb (operands pre-staged in
    /// SMEM, so no DMA) + C_spad in the high word.
    fn loop_ws_rs2(c_spad: u64) -> u64 {
        const SKIP_LDA_LDB: u64 = (1 << 3) | (1 << 4);
        (1 << 9) | SKIP_LDA_LDB | (c_spad << 32)
    }

    /// Drive a single-PE-tile (TI=TJ=TK=1) FP8 matmul: A at spad 0, B at spad 16,
    /// C at spad 256. Returns the DIM×DIM bf16-decoded C tile.
    fn run_single_tile(a: &[[u8; DIM]; DIM], b: &[[u8; DIM]; DIM]) -> [[f32; DIM]; DIM] {
        let mut bytes = vec![0u8; 0x90000];
        let mut dev = MxGemminiDevice::new(MxConfig::default());
        dev.init_status(&mut bytes);

        // Stage A (spad rows 0..15 -> bytes 0..255) and B (spad rows 16..31).
        let a_sp = 0usize;
        let b_sp = 16usize;
        for row in 0..DIM {
            for col in 0..DIM {
                bytes[(a_sp + row) * DIM + col] = a[row][col];
                bytes[(b_sp + row) * DIM + col] = b[row][col];
            }
        }
        // Identity e8m0 scales (0x7f == 2^0), as the kernel stages into the SF SRAMs.
        // A zeroed SF region would mean code 0 == 2^-127, i.e. results scaled to zero.
        for i in 0..64 {
            bytes[SF_MEM_A + i] = 0x7f;
            bytes[SF_MEM_B + i] = 0x7f;
        }

        let c_spad = 256u64;
        // CONFIG_EX: FP8 (act/wgt/out fmt = 0), WS mode.
        cmd(&mut dev, &mut bytes, F_CONFIG, 0b100, 0);
        // BOUNDS I=J=K=1.
        cmd(&mut dev, &mut bytes, F_LOOP_WS_CONFIG_BOUNDS, 0, 1 | (1 << 16) | (1u64 << 32));
        // SPAD A/B: b_spad passed to config = B_sp + TK*TJ*DIM = 16 + 16 = 32.
        cmd(&mut dev, &mut bytes, F_LOOP_WS_CONFIG_SPAD_AB, a_sp as u64, (b_sp + DIM) as u64);
        // SPAD C.
        cmd(&mut dev, &mut bytes, F_LOOP_WS_CONFIG_SPAD_C, c_spad, 0);
        // LOOP_WS: compute over the pre-staged operands.
        cmd(&mut dev, &mut bytes, F_LOOP_WS, 0, loop_ws_rs2(c_spad));

        let n_dim = DIM; // TJ*DIM
        let c_base = c_spad as usize * DIM; // byte base (matches kernel C_smem)
        let mut out = [[0.0f32; DIM]; DIM];
        for r in 0..DIM {
            for c in 0..DIM {
                let byte = c_base + (r * n_dim + c) * 2;
                out[r][c] = fp_math::bf16_to_f32(read_bf16_at(&bytes, byte));
            }
        }
        out
    }

    #[test]
    fn status_regs_ready_after_init() {
        let mut bytes = vec![0u8; 0x90000];
        let dev = MxGemminiDevice::new(MxConfig::default());
        dev.init_status(&mut bytes);
        assert_eq!(read_u32(&bytes, BASE + OFF_READY), 1, "READY must be 1");
        assert_eq!(read_u32(&bytes, BASE + OFF_BUSY), 0, "BUSY must be 0");
    }

    #[test]
    fn fp8_ones_matmul_k16() {
        // A = B = all 1.0 (fp8). C[r][c] = sum over K=16 of 1*1 = 16.0.
        let ones = [[FP8_ONE; DIM]; DIM];
        let c = run_single_tile(&ones, &ones);
        for r in 0..DIM {
            for col in 0..DIM {
                assert_eq!(c[r][col], 16.0, "C[{r}][{col}] = {}", c[r][col]);
            }
        }
    }

    /// A load-only LOOP_WS (skip_ex) must not compute. In the Radiance deployment the C
    /// region lives in the same SMEM as the operands, so computing here would write C
    /// over the A tile at spad 0 — the kernel issues exactly this call before the real one.
    #[test]
    fn skip_ex_does_not_compute() {
        let mut bytes = vec![0u8; 0x90000];
        let mut dev = MxGemminiDevice::new(MxConfig::default());
        dev.init_status(&mut bytes);
        for row in 0..DIM {
            for col in 0..DIM {
                bytes[row * DIM + col] = FP8_ONE;
                bytes[(16 + row) * DIM + col] = FP8_ONE;
            }
        }
        cmd(&mut dev, &mut bytes, F_CONFIG, 0b100, 0);
        cmd(&mut dev, &mut bytes, F_LOOP_WS_CONFIG_BOUNDS, 0, 1 | (1 << 16) | (1u64 << 32));
        cmd(&mut dev, &mut bytes, F_LOOP_WS_CONFIG_SPAD_AB, 0, (16 + DIM) as u64);

        // skip_lda|skip_ldb|skip_ldd|skip_ex|skip_stc, C_spad = 0 (aliases the A tile).
        let skips = (1u64 << 3) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7);
        cmd(&mut dev, &mut bytes, F_LOOP_WS, 0, (1 << 9) | skips);

        for row in 0..DIM {
            for col in 0..DIM {
                assert_eq!(bytes[row * DIM + col], FP8_ONE, "A clobbered at [{row}][{col}]");
            }
        }
    }

    /// The accelerator must report BUSY until its charged work completes, so the
    /// kernel's gemmini_fence spin turns accelerator latency into SIMT cycles.
    /// In an untimed run (now = None) it always reads idle.
    #[test]
    fn busy_until_gates_fence_poll() {
        let mut bytes = vec![0u8; 0x90000];
        let mut dev = MxGemminiDevice::new(MxConfig::default());
        dev.init_status(&mut bytes);
        assert_eq!(read_u32(&bytes, BASE + OFF_BUSY), 0);

        dev.charge(Some(100), 50); // busy through cycle 149
        dev.refresh_status(&mut bytes, Some(120));
        assert_eq!(read_u32(&bytes, BASE + OFF_BUSY), 1, "busy mid-flight");
        dev.refresh_status(&mut bytes, Some(150));
        assert_eq!(read_u32(&bytes, BASE + OFF_BUSY), 0, "idle after completion");

        // untimed run: never busy
        dev.refresh_status(&mut bytes, None);
        assert_eq!(read_u32(&bytes, BASE + OFF_BUSY), 0, "untimed is always idle");
    }

    /// Work serializes: a second charge queues behind the first.
    #[test]
    fn charges_serialize() {
        let mut dev = MxGemminiDevice::new(MxConfig::default());
        dev.charge(Some(10), 100); // -> 110
        dev.charge(Some(20), 100); // still busy at 20, so -> 210
        assert_eq!(dev.busy_until, 210);
    }

    /// Drive a sub-byte (FP4/FP6) matmul through the full command path -- including the
    /// accelerator's own A/B DMA out of gmem -- and check the C tile bit-for-bit against
    /// spike-validated expected values. 32x32x32, so TI=TJ=1, TK=2 (32-wide PE tiles).
    fn run_subbyte(
        actf: u8,
        a_packed: &[u8],
        b_packed: &[u8],
        sa: &[u8],
        sb: &[u8],
        luts: Option<(&[u32], &[u32])>,
    ) -> Vec<u16> {
        let (c, _) = run_subbyte_full(actf, a_packed, b_packed, sa, sb, luts, 3, None);
        c.chunks(2).map(|w| u16::from_le_bytes([w[0], w[1]])).collect()
    }

    /// As `run_subbyte`, but returns the raw C-region bytes and the e8m0 scale codes the
    /// accelerator wrote to DRAM. `out_fmt` 3 = full bf16, 0/1/2 = requant.
    #[allow(clippy::too_many_arguments)]
    fn run_subbyte_full(
        actf: u8,
        a_packed: &[u8],
        b_packed: &[u8],
        sa: &[u8],
        sb: &[u8],
        luts: Option<(&[u32], &[u32])>,
        out_fmt: u64,
        lut_c: Option<&[u32]>,
    ) -> (Vec<u8>, Vec<u8>) {
        const M: usize = 32;
        const N: usize = 32;
        const K: usize = 32;
        let (ti, tj, tk) = (1u64, 1u64, (K / DIM) as u64);

        let mut smem = vec![0u8; 0x90000];
        let mut gmem = vec![0u8; 0x10000];
        let mut dev = MxGemminiDevice::new(MxConfig::default());
        dev.init_status(&mut smem);

        // operands in gmem: A packed [M/2][K], B packed [K][N/2]
        let (a_dram, b_dram) = (0x1000usize, 0x2000usize);
        gmem[a_dram..a_dram + a_packed.len()].copy_from_slice(a_packed);
        gmem[b_dram..b_dram + b_packed.len()].copy_from_slice(b_packed);
        // e8m0 scales staged in SMEM by the kernel
        smem[SF_MEM_A..SF_MEM_A + sa.len()].copy_from_slice(sa);
        smem[SF_MEM_B..SF_MEM_B + sb.len()].copy_from_slice(sb);
        // fp6 LUTs staged in SMEM as 3 packed LE u32 words each
        if let Some((la, lb)) = luts {
            for (i, w) in la.iter().enumerate() {
                write_u32(&mut smem, (LUT1_ACT + i * 4) as u64, *w);
            }
            for (i, w) in lb.iter().enumerate() {
                write_u32(&mut smem, (LUT0_WGT + i * 4) as u64, *w);
            }
        }

        if let Some(lc) = lut_c {
            for (i, w) in lc.iter().enumerate() {
                write_u32(&mut smem, (LUT2_OUT + i * 4) as u64, *w);
            }
        }

        let use_lut = (actf == 1) as u64;
        let c_spad = 256u64;
        let b_spad_end = 8192u64; // tiles live TK*TJ*DIM rows below this
        let scale_dram = 0x3000u64; // where the accelerator writes e8m0 output scales

        let mut cmd_g = |dev: &mut MxGemminiDevice, smem: &mut [u8], f: u32, rs1: u64, rs2: u64| {
            write_u32(smem, BASE + OFF_RS1, rs1 as u32);
            write_u32(smem, BASE + OFF_RS1 + 4, (rs1 >> 32) as u32);
            write_u32(smem, BASE + OFF_RS2, rs2 as u32);
            write_u32(smem, BASE + OFF_RS2 + 4, (rs2 >> 32) as u32);
            write_u32(smem, BASE + OFF_INST, f << 25);
            dev.on_write(BASE + OFF_INST, smem, &mut gmem, None);
        };

        // CONFIG_EX: WS, use_lut, act/wgt = actf, out = 3 (full bf16)
        let rs1 =
            0b100 | (use_lut << 5) | ((actf as u64) << 10) | ((actf as u64) << 12) | (out_fmt << 14);
        cmd_g(&mut dev, &mut smem, F_CONFIG, rs1, 0);
        cmd_g(&mut dev, &mut smem, F_LOOP_WS_CONFIG_BOUNDS, 0, ti | (tj << 16) | (tk << 32));
        // MXQUANT: output-scale DRAM addr in rs1, lut granularity G=1 in rs2
        cmd_g(&mut dev, &mut smem, F_MXQUANT_CONFIG_MVOUT, scale_dram, 1);
        // DRAM addrs + row strides (A: K bytes, B: N/2 bytes)
        cmd_g(&mut dev, &mut smem, F_LOOP_WS_CONFIG_ADDRS_AB, a_dram as u64, b_dram as u64);
        cmd_g(&mut dev, &mut smem, F_LOOP_WS_CONFIG_STRIDES_AB, K as u64, (N / 2) as u64);
        cmd_g(&mut dev, &mut smem, F_LOOP_WS_CONFIG_SPAD_AB, 0, b_spad_end);
        cmd_g(&mut dev, &mut smem, F_LOOP_WS_CONFIG_SPAD_C, c_spad, 0);

        // load-only loop_ws (skip_ex|skip_stc|skip_ldd), then compute-only (skip_lda|skip_ldb)
        let load_only = (1u64 << 5) | (1 << 6) | (1 << 7);
        cmd_g(&mut dev, &mut smem, F_LOOP_WS, 0, (1 << 9) | load_only);
        let compute = (1u64 << 3) | (1 << 4) | (1 << 5);
        cmd_g(&mut dev, &mut smem, F_LOOP_WS, 0, (1 << 9) | compute | (c_spad << 32));

        let c_base = c_spad as usize * DIM;
        let n_blocks = N / 32;
        // full bf16 extent; requant callers only look at the first M*N (packed) bytes
        let c_bytes = smem[c_base..c_base + M * N * 2].to_vec();
        let scales = gmem[scale_dram as usize..scale_dram as usize + M * n_blocks].to_vec();
        (c_bytes, scales)
    }

    #[test]
    fn fp4_matmul_matches_spike_validated_golden() {
        use mx_fixtures as fx;
        let got = run_subbyte(2, &fx::FP4_A_PACKED, &fx::FP4_B_PACKED, &fx::FP4_SA, &fx::FP4_SB, None);
        let exp = fx::FP4_EXPECTED_C;
        let same = got.iter().zip(exp.iter()).filter(|(a, b)| a == b).count();
        assert_eq!(same, exp.len(), "fp4: {}/{} bit-exact", same, exp.len());
    }

    #[test]
    fn fp6_lut_matmul_matches_spike_validated_golden() {
        use mx_fixtures as fx;
        let got = run_subbyte(
            1,
            &fx::FP6_A_PACKED,
            &fx::FP6_B_PACKED,
            &fx::FP6_SA,
            &fx::FP6_SB,
            Some((&fx::FP6_LUT_A_WORDS, &fx::FP6_LUT_B_WORDS)),
        );
        let exp = fx::FP6_EXPECTED_C;
        let same = got.iter().zip(exp.iter()).filter(|(a, b)| a == b).count();
        assert_eq!(same, exp.len(), "fp6: {}/{} bit-exact", same, exp.len());
    }

    /// Sub-byte requant: packed nibble codes along M + per-row/per-32-col e8m0 scales
    /// written to DRAM. Expectations come from mx_golden (bit-exact vs REAL spike).
    #[test]
    fn fp4_requant_matches_spike_validated_golden() {
        use mx_fixtures as fx;
        let (c, s) = run_subbyte_full(
            2, &fx::FP4_A_PACKED, &fx::FP4_B_PACKED, &fx::FP4_SA, &fx::FP4_SB, None, 2, None,
        );
        let half = fx::FP4_REQ_PACKED.len();
        assert_eq!(&c[..half], &fx::FP4_REQ_PACKED[..], "fp4 requant packed codes");
        assert_eq!(&s[..], &fx::FP4_REQ_SCALES[..], "fp4 requant e8m0 scales");
    }

    #[test]
    fn fp6_requant_matches_spike_validated_golden() {
        use mx_fixtures as fx;
        let (c, s) = run_subbyte_full(
            1,
            &fx::FP6_A_PACKED,
            &fx::FP6_B_PACKED,
            &fx::FP6_SA,
            &fx::FP6_SB,
            Some((&fx::FP6_LUT_A_WORDS, &fx::FP6_LUT_B_WORDS)),
            1,
            Some(&fx::FP6_LUT_C_WORDS),
        );
        let half = fx::FP6_REQ_PACKED.len();
        assert_eq!(&c[..half], &fx::FP6_REQ_PACKED[..], "fp6 requant packed codes");
        assert_eq!(&s[..], &fx::FP6_REQ_SCALES[..], "fp6 requant e8m0 scales");
    }

    /// FLUSH must clear the accumulator, else a software-tiled matmul piles each output
    /// block onto the previous block's residue.
    #[test]
    fn flush_clears_accumulator() {
        let ones = [[FP8_ONE; DIM]; DIM];
        let mut bytes = vec![0u8; 0x90000];
        let mut dev = MxGemminiDevice::new(MxConfig::default());
        dev.init_status(&mut bytes);
        for row in 0..DIM {
            for col in 0..DIM {
                bytes[row * DIM + col] = FP8_ONE;
                bytes[(16 + row) * DIM + col] = FP8_ONE;
            }
        }
        for i in 0..64 {
            bytes[SF_MEM_A + i] = 0x7f;
            bytes[SF_MEM_B + i] = 0x7f;
        }
        let c_spad = 256u64;
        cmd(&mut dev, &mut bytes, F_CONFIG, 0b100 | (3 << 14), 0);
        cmd(&mut dev, &mut bytes, F_LOOP_WS_CONFIG_BOUNDS, 0, 1 | (1 << 16) | (1u64 << 32));
        cmd(&mut dev, &mut bytes, F_LOOP_WS_CONFIG_SPAD_AB, 0, (16 + DIM) as u64);
        cmd(&mut dev, &mut bytes, F_LOOP_WS, 0, loop_ws_rs2(c_spad));

        let c_base = c_spad as usize * DIM;
        assert_eq!(fp_math::bf16_to_f32(read_bf16_at(&bytes, c_base)), 16.0);
        // running it again accumulates (16 + 16)
        cmd(&mut dev, &mut bytes, F_LOOP_WS, 0, loop_ws_rs2(c_spad));
        assert_eq!(fp_math::bf16_to_f32(read_bf16_at(&bytes, c_base)), 32.0);
        // flush clears C, so the next matmul starts from zero -- and A is untouched
        cmd(&mut dev, &mut bytes, F_FLUSH, 0, 0);
        assert_eq!(read_bf16_at(&bytes, c_base), 0, "flush must clear the accumulator");
        assert_eq!(bytes[0], FP8_ONE, "flush must not clobber the A operand");
        cmd(&mut dev, &mut bytes, F_LOOP_WS, 0, loop_ws_rs2(c_spad));
        assert_eq!(fp_math::bf16_to_f32(read_bf16_at(&bytes, c_base)), 16.0);
    }

    /// K-tile scale double-buffering: odd K-tiles stage their e8m0 scales in the second
    /// SF buffer (SF_MEM_* + SF_MEM_BUFFER_OFFSET) and set the mxquant select bits
    /// (rs1[60]/rs1[61]). The accelerator must read from the selected buffer; ignoring
    /// the select made every odd tile read the previous (even) tile's stale scales,
    /// corrupting ~half of a multi-K-tile matmul (prob22 64x64x512 failed 4089/4096
    /// before this fix; prob20's single even tile was unaffected, hiding the bug).
    #[test]
    fn scale_double_buffer_select_is_honored() {
        let mut bytes = vec![0u8; 0x90000];
        let mut dev = MxGemminiDevice::new(MxConfig::default());
        dev.init_status(&mut bytes);
        // 16x16 A at spad 0, B at spad 16, all ones -> raw dot over K=16 is 16.0.
        for row in 0..DIM {
            for col in 0..DIM {
                bytes[row * DIM + col] = FP8_ONE;
                bytes[(16 + row) * DIM + col] = FP8_ONE;
            }
        }
        // base buffer: unit scale (e8m0 0x7f = 2^0). offset buffer: 0x80 each ->
        // combined e = 0x80+0x80-127 = 129 -> 2^2 = 4, so the scaled result is 64.0.
        for i in 0..DIM {
            bytes[SF_MEM_A + i] = 0x7f;
            bytes[SF_MEM_B + i] = 0x7f;
            bytes[SF_MEM_A + SF_MEM_BUFFER_OFFSET + i] = 0x80;
            bytes[SF_MEM_B + SF_MEM_BUFFER_OFFSET + i] = 0x80;
        }
        let c_spad = 256u64;
        let c_base = c_spad as usize * DIM;
        cmd(&mut dev, &mut bytes, F_CONFIG, 0b100 | (3 << 14), 0);
        cmd(&mut dev, &mut bytes, F_LOOP_WS_CONFIG_BOUNDS, 0, 1 | (1 << 16) | (1u64 << 32));
        cmd(&mut dev, &mut bytes, F_LOOP_WS_CONFIG_SPAD_AB, 0, (16 + DIM) as u64);

        // select = 0 -> base buffer (2^0): result 16.0.
        cmd(&mut dev, &mut bytes, F_MXQUANT_CONFIG_MVOUT, 0, 0);
        cmd(&mut dev, &mut bytes, F_LOOP_WS, 0, loop_ws_rs2(c_spad));
        assert_eq!(
            fp_math::bf16_to_f32(read_bf16_at(&bytes, c_base)),
            16.0,
            "select=0 must read the base scale buffer"
        );
        cmd(&mut dev, &mut bytes, F_FLUSH, 0, 0);

        // select = 1 (both act/wgt) -> offset buffer (2^2): result 64.0.
        cmd(&mut dev, &mut bytes, F_MXQUANT_CONFIG_MVOUT, (1u64 << 60) | (1u64 << 61), 0);
        cmd(&mut dev, &mut bytes, F_LOOP_WS, 0, loop_ws_rs2(c_spad));
        assert_eq!(
            fp_math::bf16_to_f32(read_bf16_at(&bytes, c_base)),
            64.0,
            "select=1 must read the offset scale buffer, not stale base scales"
        );
    }

    /// MX_READ_SMEM drains the bf16 accumulator from SMEM to DRAM.
    #[test]
    fn mx_read_smem_drains_to_dram() {
        let mut smem = vec![0u8; 0x90000];
        let mut gmem = vec![0u8; 0x10000];
        let mut dev = MxGemminiDevice::new(MxConfig::default());
        dev.init_status(&mut smem);

        let c_base = 256usize * DIM;
        let vals: [u16; 4] = [0x3f80, 0xc000, 0x4048, 0x0000];
        for (e, v) in vals.iter().enumerate() {
            write_bf16_at(&mut smem, c_base + e * 2, *v);
        }
        let dram = 0x2000u64;
        let rs2 = (c_base as u64) | ((vals.len() as u64) << 32);
        write_u32(&mut smem, BASE + OFF_RS1, dram as u32);
        write_u32(&mut smem, BASE + OFF_RS1 + 4, 0);
        write_u32(&mut smem, BASE + OFF_RS2, rs2 as u32);
        write_u32(&mut smem, BASE + OFF_RS2 + 4, (rs2 >> 32) as u32);
        write_u32(&mut smem, BASE + OFF_INST, F_MX_READ_SMEM << 25);
        dev.on_write(BASE + OFF_INST, &mut smem, &mut gmem, None);

        for (e, v) in vals.iter().enumerate() {
            let got = u16::from_le_bytes([gmem[dram as usize + e * 2], gmem[dram as usize + e * 2 + 1]]);
            assert_eq!(got, *v, "drained element {e}");
        }
    }

    /// The MX softmax (I-BERT) pass, cross-checked against REAL spike: feed spike's
    /// pre-softmax C tile through our pass and expect spike's post-softmax C, bit-for-bit.
    #[test]
    fn mx_softmax_matches_real_spike() {
        use mx_fixtures as fx;
        const M: usize = 64;
        const N: usize = 64;
        let mut smem = vec![0u8; 0x90000];
        let mut dev = MxGemminiDevice::new(MxConfig::default());
        dev.init_status(&mut smem);
        // config_norm with act_msb -> acc_act = SOFTMAX, plus the I-BERT constants
        let norm = |q: u64, ty: u64| -> u64 { (q << 32) | (ty << 18) | (1 << 16) | 0b11 };
        let qbc: u64 = (1_006_165u64 << 32) | 1385;
        cmd(&mut dev, &mut smem, F_CONFIG, norm(710, 0), qbc);
        cmd(&mut dev, &mut smem, F_CONFIG, norm(92, 1), qbc);
        assert_eq!(dev.st.acc_act, AccAct::Softmax, "config_norm must enable softmax");

        let c_base = 256usize * DIM;
        for (e, v) in fx::SOFTMAX_C_PLAIN.iter().enumerate() {
            write_bf16_at(&mut smem, c_base + e * 2, *v);
        }
        dev.mx_softmax(&mut smem, c_base, M, N);
        for e in 0..M * N {
            assert_eq!(
                read_bf16_at(&smem, c_base + e * 2),
                fx::SOFTMAX_C_EXPECTED[e],
                "softmax mismatch at [{}, {}]",
                e / N,
                e % N
            );
        }
    }

    /// act_fmt=3 (FORMAT_FULL) is an output format, never an input dtype. Spike has no
    /// such branch and computes nothing; we must do the same rather than produce garbage.
    #[test]
    fn act_fmt_full_is_illegal_and_computes_nothing() {
        let mut bytes = vec![0u8; 0x90000];
        let mut dev = MxGemminiDevice::new(MxConfig::default());
        dev.init_status(&mut bytes);
        for row in 0..DIM {
            for col in 0..DIM {
                bytes[row * DIM + col] = FP8_ONE;
                bytes[(16 + row) * DIM + col] = FP8_ONE;
            }
        }
        for i in 0..64 {
            bytes[SF_MEM_A + i] = 0x7f;
            bytes[SF_MEM_B + i] = 0x7f;
        }
        let c_spad = 256u64;
        // act_fmt = 3 in rs1[11:10]
        cmd(&mut dev, &mut bytes, F_CONFIG, 0b100 | (3 << 10) | (3 << 14), 0);
        cmd(&mut dev, &mut bytes, F_LOOP_WS_CONFIG_BOUNDS, 0, 1 | (1 << 16) | (1u64 << 32));
        cmd(&mut dev, &mut bytes, F_LOOP_WS_CONFIG_SPAD_AB, 0, (16 + DIM) as u64);
        cmd(&mut dev, &mut bytes, F_LOOP_WS, 0, loop_ws_rs2(c_spad));

        let c_base = c_spad as usize * DIM;
        assert_eq!(read_bf16_at(&bytes, c_base), 0, "act_fmt=3 must not compute");
    }

    #[test]
    fn fp8_row_indexing() {
        // Only A row 0 is nonzero -> only C row 0 should be 16.0, rest 0.
        let mut a = [[0u8; DIM]; DIM];
        for col in 0..DIM {
            a[0][col] = FP8_ONE;
        }
        let ones = [[FP8_ONE; DIM]; DIM];
        let c = run_single_tile(&a, &ones);
        for col in 0..DIM {
            assert_eq!(c[0][col], 16.0, "C[0][{col}]");
        }
        for r in 1..DIM {
            for col in 0..DIM {
                assert_eq!(c[r][col], 0.0, "C[{r}][{col}] should be 0");
            }
        }
    }
}
