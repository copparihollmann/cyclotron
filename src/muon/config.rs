use crate::sim::config::Config;
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone, Copy)]
#[serde(default)]
pub struct MuonConfig {
    pub num_lanes: usize,
    pub num_warps: usize,
    pub num_cores: usize, // per-cluster
    pub num_regs: usize,
    // Size of the shared physical register file per core. The RTL Rename stage
    // renames each warp's architectural registers into this pool, shared across all
    // num_warps active warps; if the total distinct registers used exceeds
    // num_phys_regs the RTL asserts globalOverSubscription. Modeled here so cyclotron
    // rejects register-illegal kernels the RTL cannot run. 0 disables the check.
    pub num_phys_regs: usize,
    pub start_pc: u32,
    pub smem_size: usize,
    #[serde(skip)]
    pub lane_config: LaneConfig,
}

impl Config for MuonConfig {}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            num_lanes: 16,
            num_warps: 8,
            num_cores: 1,
            num_regs: 256,
            num_phys_regs: 256,
            start_pc: 0x10000000u32,
            smem_size: 0x10_0000, // includes MMIO space
            lane_config: LaneConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LaneConfig {
    pub lane_id: usize,
    pub warp_id: usize,
    pub core_id: usize,
    pub cluster_id: usize,
}

impl Default for LaneConfig {
    fn default() -> Self {
        Self {
            lane_id: 0,
            warp_id: 0,
            core_id: 0,
            cluster_id: 0,
        }
    }
}
