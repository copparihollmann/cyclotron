use std::io::Write;

use crate::{
    base::mem::HasMemory,
    muon::mxgemmini::{MxConfig, MxGemminiDevice},
    sim::{config::MemConfig, elf::ElfBackedMem},
    timeq::Cycle,
};

/// Gigantic 4 GB vector to model memory space; relies on lazy allocation within OS to avoid actually
/// causing memory pressure. Avoids hash-table lookup for every memory access
#[derive(Debug, Clone)]
pub struct FlatMemory {
    bytes: Vec<u8>,
    config: Option<MemConfig>,
    /// Functional MX-Gemmini accelerator, attached only to the SMEM instance when
    /// enabled. Combined Muon+MX kernels push RoCC commands to the Gemmini MMIO
    /// block via `.shared` stores; this executes the matmul on the SMEM scratchpad.
    mxgemmini: Option<MxGemminiDevice>,
}

impl HasMemory for FlatMemory {
    fn read_impl(&self, addr: usize, n: usize) -> Result<&[u8], anyhow::Error> {
        Ok(self.bytes[addr..addr + n].try_into().unwrap())
    }

    fn write_impl(&mut self, addr: usize, data: &[u8]) -> Result<(), anyhow::Error> {
        // Check if this is a write to the I/O console address range
        if let Some(config) = self.config {
            if addr >= config.io_cout_addr && addr < config.io_cout_addr + config.io_cout_size {
                let bytes = &*data;
                if let Ok(text) = std::str::from_utf8(bytes) {
                    print!("{}", text);
                } else {
                    // Print as hex if not valid UTF-8
                    for &byte in bytes.iter() {
                        print!("{:02x}", byte);
                    }
                }
                std::io::stdout().flush().unwrap();
                return Ok(());
            }
        }

        let bytes = &mut self.bytes[addr..addr + data.len()];
        bytes.copy_from_slice(data);

        Ok(())
    }
}

impl FlatMemory {
    pub fn new(config: Option<MemConfig>) -> Self {
        let bytes = vec![0u8; 1 << 32];
        Self {
            bytes,
            config,
            mxgemmini: None,
        }
    }

    pub fn new_with_size(size: usize, config: Option<MemConfig>) -> Self {
        let bytes = vec![0u8; size];
        Self {
            bytes,
            config,
            mxgemmini: None,
        }
    }

    /// Attach the functional MX-Gemmini co-model (SMEM instance only). Initializes
    /// the accelerator status registers so the kernel's fence polls pass. Gated by
    /// the caller (e.g. an env flag) so SIMT-only runs are unaffected.
    pub fn enable_mxgemmini(&mut self, cfg: MxConfig) {
        let dev = MxGemminiDevice::new(cfg);
        dev.init_status(&mut self.bytes);
        self.mxgemmini = Some(dev);
    }

    /// True when an MX-Gemmini device is attached and `addr` lands in its MMIO window.
    pub fn mx_matches(&self, addr: u64) -> bool {
        self.mxgemmini
            .as_ref()
            .map_or(false, |dev| dev.matches_mmio(addr))
    }

    /// Run the MX-Gemmini command decode for a store that just landed at `addr`.
    /// Driven from the LSU (`Warp::mem_response`) rather than `write_impl` because the
    /// accelerator's `loop_ws` load phase DMAs A/B operands out of global memory, so
    /// it needs both memories in scope. `self` is the SMEM (== the Gemmini scratchpad).
    /// `now` is None in untimed runs (accelerator then completes instantaneously).
    pub fn mx_on_write(&mut self, addr: u64, gmem: &mut FlatMemory, now: Option<Cycle>) {
        // Move the device out so it can take `&mut self.bytes` without aliasing.
        let Some(mut dev) = self.mxgemmini.take() else {
            return;
        };
        dev.on_write(addr, &mut self.bytes, &mut gmem.bytes, now);
        self.mxgemmini = Some(dev);
    }

    /// Republish the accelerator status registers for cycle `now`, so a load of
    /// BUSY/OCCUPANCY (the kernel's `gemmini_fence` spin) observes outstanding work.
    pub fn mx_refresh_status(&mut self, now: Option<Cycle>) {
        let Some(dev) = self.mxgemmini.take() else {
            return;
        };
        dev.refresh_status(&mut self.bytes, now);
        self.mxgemmini = Some(dev);
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

    pub fn copy_elf(&mut self, elf: &ElfBackedMem) {
        for (section, data) in elf.sections.iter() {
            let (start, end) = *section;
            let gpu_offset = 0x100000000;
            let gpu_start: usize;
            let gpu_end: usize;
            if start >= 0x100000000 && end < 0x200000000 {
                // try to detect if this is a CPU-GPU fused ELF, and if so relocate to GPU-private
                // GMEM address space. TODO: hacky and loosey-goosey.
                gpu_start = start - gpu_offset;
                gpu_end = end - gpu_offset;
                println!(
                    "Cyclotron: copy_elf: detected CPU-fused ELF; relocating section [{start:x}..{end:x}] to [{gpu_start:x}..{gpu_end:x}]"
                );
            } else {
                gpu_start = start;
                gpu_end = end;
            }

            let len = self.bytes.len();
            let bytes = &mut self.bytes.get_mut(gpu_start..gpu_end).expect(&format!(
                "copy_elf: copy dest ({gpu_start:x}..{gpu_end:x}) out-of-range of FlatMemory (0..{len:x})"
            ));
            bytes.copy_from_slice(&data);
        }
    }

    pub fn size_bytes(&self) -> usize {
        self.bytes.len()
    }
}
