use crate::muon::config::MuonConfig;
use crate::muon::warp::Writeback;
use std::collections::VecDeque;
use std::fmt;
use std::iter::zip;

/// Queues instruction traces from a core into a buffer, and provides a per-warp consume()
/// interface to dequeue instructions from the buffer in program-order.
pub struct Tracer {
    /// per-warp program-order buffer of lines
    bufs: Vec<VecDeque<Line>>,
    /// round-robin warp selector for consumption
    rr: usize,
}

pub struct MemTracer {
    buf: VecDeque<MemTraceLine>,
}

#[derive(Default, Clone, Debug)]
pub struct Line {
    pub warp_id: u32,
    pub pc: u32,
    pub opcode: u8,
    pub opext: u8,
    pub rd_addr: u8,
    pub rs1_addr: u8,
    pub rs2_addr: u8,
    pub rs3_addr: u8,
    pub rs4_addr: u8,
    pub rd_data: Vec<Option<u32>>,
    pub rs1_data: Vec<Option<u32>>,
    pub rs2_data: Vec<Option<u32>>,
    pub rs3_data: Vec<Option<u32>>,
    pub rs4_data: Vec<Option<u32>>,
    pub f3: u8,
    pub f7: u8,
    pub imm32: u32,
    pub imm24: i32,
    pub csr_imm: u8,
    pub tmask: u32,
    pub raw: u64,
    /// Timing-model cycle at which this instruction issued (0 in functional-only mode,
    /// where it carries the functional step count instead).
    pub cycle: u64,
}

#[derive(Clone, Debug)]
pub struct MemTraceLine {
    pub warp_id: u32,
    pub lane_id: u32,
    pub is_smem: bool,
    pub store: bool,
    pub address: u32,
    pub size: u32,
    pub data: u32,
    /// Timing-model cycle at which this memory access issued.
    pub cycle: u64,
}

impl fmt::Display for Line {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TraceLine: (warp:{}, pc:0x{:x}, opcode:{}, rd:{})",
            self.warp_id, self.pc, self.opcode, self.rd_addr
        )
    }
}

impl Tracer {
    pub fn new(muon_config: &MuonConfig) -> Tracer {
        let mut tracer = Tracer {
            bufs: Vec::new(),
            rr: 0,
        };
        for _ in 0..muon_config.num_warps {
            tracer.bufs.push(VecDeque::new());
        }
        tracer
    }

    pub fn record(&mut self, writebacks: &[Option<Writeback>], now: u64) {
        for (wid, (wb, buf)) in zip(writebacks.iter(), self.bufs.as_mut_slice()).enumerate() {
            if let Some(wb) = wb {
                let line = Line {
                    cycle: now,
                    warp_id: wid as u32,
                    pc: wb.inst.pc,
                    opcode: wb.inst.opcode,
                    opext: wb.inst.opext,
                    rd_addr: wb.inst.rd_addr,
                    rs1_addr: wb.inst.rs1_addr,
                    rs2_addr: wb.inst.rs2_addr,
                    rs3_addr: wb.inst.rs3_addr,
                    rs4_addr: wb.inst.rs4_addr,
                    rd_data: wb.rd_data.clone(),
                    rs1_data: wb.inst.rs1_data.clone(),
                    rs2_data: wb.inst.rs2_data.clone(),
                    rs3_data: wb.inst.rs3_data.clone(),
                    rs4_data: wb.inst.rs4_data.clone(),
                    f3: wb.inst.f3,
                    f7: wb.inst.f7,
                    imm32: wb.inst.imm32,
                    imm24: wb.inst.imm24,
                    csr_imm: wb.inst.csr_imm,
                    tmask: wb.tmask,
                    raw: wb.inst.raw,
                };
                buf.push_back(line);
                // println!("trace: pushed line to buf for warp={}; len={}", wid, buf.len());
            }
        }
    }

    pub fn peek(&self, warp_id: usize) -> Option<&Line> {
        self.bufs[warp_id].front()
    }

    pub fn consume(&mut self, warp_id: usize) -> Option<Line> {
        self.bufs[warp_id].pop_front()
    }

    /// Consume an instruction from any warp's buffer, round-robinning the warp selection.
    pub fn consume_round_robin(&mut self) -> Option<Line> {
        for i in 0..self.bufs.len() {
            let wid = self.rr + i;
            self.rr = (self.rr + 1) % self.bufs.len();
            let buf = &self.bufs[wid];
            if buf.len() == 0 {
                continue;
            }
            return self.consume(wid);
        }
        None
    }

    pub fn len(&self, warp_id: usize) -> usize {
        self.bufs[warp_id].len()
    }

    pub fn all_empty(&self) -> bool {
        self.bufs.iter().all(|buf| buf.len() == 0)
    }

    /// Drop buffered trace records when the simulator was started without a
    /// trace database.  Execution tracing is produced by the core regardless
    /// of the CLI trace setting, so leaving these queues undrained makes a
    /// trace-disabled simulation retain one large record per retired
    /// instruction until the process exits.
    pub fn clear(&mut self) {
        for buf in &mut self.bufs {
            buf.clear();
        }
    }
}

impl MemTracer {
    pub fn new() -> Self {
        Self {
            buf: VecDeque::new(),
        }
    }

    pub fn record(&mut self, lines: &[MemTraceLine], now: u64) {
        self.buf.extend(lines.iter().cloned().map(|mut l| {
            l.cycle = now;
            l
        }));
    }

    pub fn consume(&mut self) -> Option<MemTraceLine> {
        self.buf.pop_front()
    }

    pub fn all_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Drop buffered memory-access traces when no trace database is active.
    pub fn clear(&mut self) {
        self.buf.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_discards_trace_disabled_instruction_and_memory_records() {
        let config = MuonConfig {
            num_warps: 2,
            num_lanes: 1,
            ..MuonConfig::default()
        };
        let mut inst = Tracer::new(&config);
        let mut mem = MemTracer::new();

        inst.bufs[0].push_back(Line::default());
        inst.bufs[1].push_back(Line::default());
        mem.record(
            &[MemTraceLine {
                warp_id: 0,
                lane_id: 0,
                is_smem: false,
                store: false,
                address: 0x1000,
                size: 4,
                data: 0,
                cycle: 0,
            }],
            17,
        );
        assert!(!inst.all_empty());
        assert!(!mem.all_empty());

        inst.clear();
        mem.clear();

        assert!(inst.all_empty());
        assert!(mem.all_empty());
    }
}
