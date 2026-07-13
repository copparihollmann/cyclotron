//! MX-Gemmini accelerator functional state.
//!
//! Mirrors the subset of `gemmini_state_t` (spike libgemmini `gemmini.h`) that the
//! MX matmul datapath reads/writes. Only the fields exercised by the Radiance
//! MMIO command stream are modeled; classic Gemmini int8 fields are omitted.

/// Accumulator activation selected via CONFIG (config_mvout rs1[3:2]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccAct {
    None,
    Relu,
    LayerNorm,
    IGelu,
    Softmax,
}

impl Default for AccAct {
    fn default() -> Self {
        AccAct::None
    }
}

#[derive(Debug, Clone, Default)]
pub struct MxGemminiState {
    // --- CONFIG_EX (funct 0, rs1[1:0]==00): MX formats ---
    /// 0 = FP8 E4M3, 1 = FP6 E3M2 (LUT), 2 = FP4 E2M1, 3 = full/bf16.
    pub mx_act_fmt: u8,
    pub mx_wgt_fmt: u8,
    pub mx_out_fmt: u8,
    pub mx_use_lut: bool,
    /// Accumulator activation (config_mvout). Softmax enables the in-loop I-BERT pass.
    pub acc_act: AccAct,

    // --- LOOP_WS_CONFIG_BOUNDS (funct 9) ---
    pub loop_ws_i: u16,
    pub loop_ws_j: u16,
    pub loop_ws_k: u16,
    pub loop_ws_pad_i: u16,
    pub loop_ws_pad_j: u16,
    pub loop_ws_pad_k: u16,

    // --- LOOP_WS_CONFIG_ADDRS_AB / _STRIDES_AB (funct 10 / 12) ---
    // DRAM base + row stride of the A/B operands, used by the loop_ws load phase.
    pub loop_ws_a_dram: u64,
    pub loop_ws_b_dram: u64,
    pub loop_ws_a_stride: u64,
    pub loop_ws_b_stride: u64,

    // --- LOOP_WS_CONFIG_SPAD_AB / _C (funct 24 / 25) ---
    pub mx_loop_a_spad: u32,
    /// NOTE: this is the *end* of the B tile region; tiles start `TK*TJ*DIM` below.
    pub mx_loop_b_spad: u32,
    pub mx_loop_c_spad: u32,

    // --- MXQUANT_CONFIG_MVOUT (funct 26): requant / output scaling ---
    pub mx_scale_dram: u64,
    pub mx_tiles_i: u32,
    pub mx_tiles_j: u32,
    pub mx_tiles_k: u32,
    /// FP6 LUT selection granularity: row/col `x` uses LUT `x >> granularity`.
    pub mx_lut_update_granularity: u32,
    /// A/B scale-SRAM double-buffer selects (rs1[60]/rs1[61]). In the Radiance
    /// shared-ext-mem deployment the kernel stages odd-K-tile e8m0 scales into a
    /// second SMEM buffer at `SF_MEM_* + SF_MEM_BUFFER_OFFSET` and sets these to
    /// tell the accelerator which buffer to read. Ignoring them reads the previous
    /// (even) tile's stale scales on every odd tile.
    pub mx_scale_act_sel: bool,
    pub mx_scale_wgt_sel: bool,

    // --- MX_LOAD_SCALES (funct 27): per-block e8m0 scale codes ---
    pub mx_scale_a_mem: Vec<u8>,
    pub mx_scale_b_mem: Vec<u8>,

    // --- MX_LOAD_LUT (funct 29): FP6 dequant LUTs (16 codes each) ---
    pub mx_lut_a: Vec<u8>,
    pub mx_lut_b: Vec<u8>,
    pub mx_lut_c: Vec<u8>,

    // I-BERT softmax constants (defaults applied at use-site if 0).
    pub qln2: u32,
    pub qln2_inv: u32,
    pub igelu_qb: u32,
    pub igelu_qc: u32,
}

impl MxGemminiState {
    pub fn new() -> Self {
        Self::default()
    }
}
