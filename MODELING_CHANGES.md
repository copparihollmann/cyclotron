# Cyclotron modeling changes — PR documentation

Branch: `muon-fidelity-session` (local). Base: `356fa57` ("trace: Expand reg addr to u32 for >256 regs").
Commit: `e12ab42`. 14 files, +288/−48.

This PR makes cyclotron's timing/limits model faithful to the RTL (`RadianceSingleClusterConfig`,
Muon 2-core/8-warp, 500 MHz tile clock, DRAMSim2 backing memory) along three axes — **register-file
pressure**, **DRAM bandwidth**, and a **cross-core SMEM coherence hazard detector** — plus supporting
trace/difftest infrastructure. Every modeling change below was validated against VCS RTL anchors.

---

## 1. Register-file pressure model (RTL-validated)  — #26

**Problem.** The RTL `Rename` stage renames each warp's architectural registers into a shared
physical-register pool per core (`numPhysRegs = 256`); if the total distinct registers used across the
core's warps exceeds it, `Rename.scala:123` asserts `globalOverSubscription` and the kernel cannot run.
Cyclotron did not model this, so it accepted register-illegal kernels the RTL rejects.

**Change.**
- `src/muon/config.rs`: add `num_phys_regs` field (default 256; 0 disables the check).
- `config.toml`: `num_phys_regs = 256`.
- `src/muon/core.rs`: `track_register_pressure` — a runtime model that counts distinct `(warp, rd)`
  first-writes over **actually-executed** instructions (a `HashSet`), monotonic per kernel launch,
  and asserts when the global count reaches `num_phys_regs`, exactly mirroring `Rename.scala:123`.
  Reports `per_warp` distribution on the trip. This is the *runtime* model; an earlier *static* ELF
  symbol scan was found to over-reject (it decoded embedded constants as instructions) and is now only
  used to locate the kernel for context, not to reject.
- `src/sim/elf.rs`: extract the `kernel_body` symbol `(addr, size)` (context for the check).
- `src/sim/top.rs`: comment documenting that register pressure is modeled at runtime in
  `track_register_pressure`, not statically.

**Validation (RTL).** Matches VCS `RadianceSingleClusterConfig` on every gated kernel: flash
`sol7_baseline` → RTL `globalOverSubscription` (Rename.scala:123) ✔ predicted illegal; matmul-lean /
conv (frugal) → RTL legal ✔ predicted legal. The model counts all 8 warps of a single-threadblock
kernel against one core's 256-reg pool (≈32 distinct regs/warp budget; ~16/warp consumed by the
CRT+main+mu_schedule harness floor).

---

## 2. DRAM bandwidth calibration to RTL/DRAMSim2  — #24

**Problem.** Cyclotron's DRAM bandwidth was uncalibrated; absolute cycles for DRAM-bound traffic did
not match the RTL's DRAMSim2 model.

**Change.** `config/timing/gmem.toml`: `[gmem.nodes.dram] bytes_per_cycle 8 → 4` (also retains the
structural calibration `l0_sets=64` = 4 KB L0d and DRAM `base_latency=40`).

**Validation (RTL).** A pure streaming probe (single-pass read of a 1 MB BSS region > L2, so every
access misses to DRAM) on VCS/DRAMSim2 = **543,711 tile cycles** (`$finish`/2000; see §5 clock note).
At `bytes_per_cycle=4` cyclotron reports **534,744** (1.6 %). Cache-resident target kernels are
unaffected (they only pay cold-miss DRAM traffic): matmul-naive +0.9 %, matmul-lean/conv/softmax
unchanged → performance ranking preserved.

**Known limitation.** RTL DRAM cost is *super-linear* with working-set size (1 MB→2 MB = 2.245× vs
cyclotron's linear 1.98×), the signature of row-buffer/bank/queueing effects. A constant-bandwidth
knob matches one size only; a DRAMSim-style row-buffer model (tRCD/tRP) would capture the scaling but
is **low ROI** here — no target kernel streams >1 MB from DRAM (all are cache-resident with reuse, and
a DRAM-bound matmul is infeasible to RT-simulate). Deferred with evidence.

---

## 3. Per-request rate-limit field on timing servers  — supports #24

**Change.**
- `src/timeq.rs`: add `cycles_per_request` to `ServerConfig` — minimum service cycles per request
  regardless of size: `service = max(ceil(bytes/bytes_per_cycle), cycles_per_request)`. Models a
  per-request rate limit independent of byte bandwidth. Default 0 (no-op).
- `src/timeflow/execute.rs`, `src/timeflow/icache.rs`: default the new field to 0 in their server configs.

**Rationale.** Physical knob for DRAM-node request throttling. NB: a blanket `cpr` on cache levels
penalizes cache *hits* and does not generalize (it cannot fit both memory-bound and compute-bound
kernels) — so it is kept at 0 by default and only intended for the DRAM node when a row-buffer model
is added.

---

## 4. Cross-core SMEM coherence hazard detector (warning-only)  — #32

**Problem.** Cyclotron models SMEM as one instantly-coherent `FlatMemory` shared by all cores, and
treats `fence.s` functionally as a no-op (`timeflow/execute.rs` MISC_MEM was `// TODO fence`). The RTL
has per-core SMEM write buffers that become cross-core-visible only after `fence.s`. So cyclotron
silently accepted kernels that read another core's SMEM writes without an intervening `fence.s` — which
read **stale** data on RTL (the cross-core staging bug).

**Change.** `src/muon/warp.rs`: a process-global per-address tracker `addr → (writer_core, visible)`.
`store_shared` records the writer; `fence.s` (detected in `execute_nomem`) marks that core's writes
visible; a cross-core SMEM **load** of an unfenced write emits a `warn!("[SMEM-COHERENCE #32] …")`.
**Warning-only** — it does not change any value, so it cannot affect functional/difftest correctness;
surfaced only at `RUST_LOG=warn` (the eval runs at `RUST_LOG=error`, so it is invisible to the normal
loop). Single-cluster, single-process assumption.

**Correct kernel pattern (learned from RTL + reference kernels gemm_simt/softmax):**
`stage → mu_fence_smem() → mu_barrier(0, total_warps)` (barrier **ID 0** = cross-core; ID 1 does not
sync across cores). The detector immediately caught a latent hazard in the 2.52× matmul-lean kernel.

---

## 5. Trace / difftest infrastructure (supporting)

- `src/sim/trace.rs`: add a `cycle` field to the instruction trace line (timing-model issue cycle;
  carries functional step count in functional-only mode).
- `src/sim/trace_db.rs`: trace-DB schema carries `cycle` + per-source (`rs1/rs2`) columns.
- `src/dpi/mod.rs`: default the DPI/difftest log level to `Error` (DIFFTEST PASS/fail lines are
  `println!`, unaffected); add `lanes_or_empty` null/alignment-safe wrappers for the DPI mem-trace
  signal slices (robustness for the cyclotron↔RTL lockstep difftest mem-trace).
- `.gitignore`: ignore `*.sqlite` (trace-DB artifacts from `--gen-trace` / `+trace-db`).

---

## 6. Important downstream note: vcs_gate clock period (NOT in this repo)

The harness gate script (`autocomp/scripts/muon/vcs_gate.sh`, separate repo) converted RTL `$finish`
ps→cycles by **/500** (assuming 2 GHz); the tile clock is **500 MHz = 2000 ps**
(`TestHarness.sv: ClockSourceAtFreqMHz #(.PERIOD(2.0))`). Fixed to `/2000`. Prior absolute cycle anchors
quoted from the gate were 4× inflated; **speedup ratios, register legality, and PASS/FAIL verdicts were
all unaffected** (ratios cancel; asserts are structural). All cyclotron-vs-RTL absolute comparisons in
this PR use the corrected /2000 conversion.

---

## 7. Intra-warp ILP scoreboard (#23 / #33) — partial, in this PR

`src/muon/gmem/{mod,core}.rs` + `issue.rs` + `warp.rs`: per-warp register scoreboard `[warp][reg]
-> ready_at`. `issue_execute` now stalls only on a true RAW dependency (a source register still in
flight) instead of the blanket `pending_execute` one-in-flight gate, so independent ops overlap (unit
throughput still bounded by the execute queue). Loads register `scoreboard[rd]=ready_at` (so dependent
ops wait under real/DRAM latency; no-op for L1 hits). Result: matmul-naive 633k→**395k** vs RTL 294k —
compute over-count **2.16×→1.34×**; memory-bound bw unchanged; all kernels isa-test pass; ranking
preserved. RAW-only is correct for in-order issue (WAW = last writer; WAR needs no stall). VALIDATED: 2-core lockstep difftest (vecadd) 0 fails / 700 insts -> timing-only change, functional values unaffected.

**Remaining (~1.34×, ~1 IPC/core vs RTL 1.34):** a further per-core issue / shared-resource limit
(empirically NOT execute-latency, execute-throughput, load latency, or icache fetch bandwidth) — needs
profiling. Candidate: the `allow_fetch` one-fetch-in-flight-per-warp gate or a per-core dispatch limit.

## (prior) Known fidelity gap — #23 / #33 background

With the corrected clock, the clean RTL anchor is matmul-naive = **293,521 tile cycles** vs cyclotron
~644k → **cyclotron OVER-counts compute-bound kernels ~2.2×** (it under-counts nothing; the earlier
"under-count" was the /500 clock-bug artifact). Root cause: cyclotron's timing model is **warp+resource
granular with no register scoreboard** — it serializes per-warp via one-in-flight gates
(`issue.rs:278 pending_execute`, `allow_fetch` icache, `add_gmem_pending`/`add_smem_pending`), so it has
**no ILP** (independent ops don't overlap) vs RTL's reservation-station backend. The fix is a unified
per-warp register scoreboard (producers — execute *and* gmem/smem loads — record `rd → ready_at`;
consumers stall to the max in-flight source). There is **no correct shortcut** (issue-width,
completions_per_cycle, icache bandwidth, and execute-latency reduction were all ruled out empirically).
Ranking is preserved, so the optimization loop is unaffected; this is an absolute-fidelity improvement
deferred to a focused effort. Fully designed in autocomp `MUON_RESULTS.md` / task #33.
