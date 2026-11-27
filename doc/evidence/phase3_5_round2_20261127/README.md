# Phase 3.5 round 2 evidence (2026-11-27)

Report: [`../../SM120_V2_PHASE3_5_TIER1_ZH.md`](../../SM120_V2_PHASE3_5_TIER1_ZH.md) §8; ledger §10.13.
All files are sanitized (host names, user names, other users' processes and private paths redacted).

| File | Content |
|---|---|
| `vllm_mp_summary.txt`, `vllm_mp_final.jsonl` | Two vLLM processes under XSched (GLB + automatic XQueues), DEF-37 build: HP latency, LP throughput, exactness per mode |
| `vllm_mp_cnt.jsonl` | DEF-37 diagnosis: LP commands on the legacy stream / unmanaged streams / XQueues (`XCudaShimStats`) |
| `def38_baseline.txt` | DEF-38 before the fix: 4-queue hang rate (5/8), 2 queues 0/8, the q=3 run with another user's job (Xid 109) |
| `def38_hang_backtrace.txt` | gdb thread stacks of a hung 4-queue process (all threads in `cuStreamSynchronize`) |
| `def38_gate_ab.txt` | CE-vs-SM probe 1; gate off vs on (same build); watchdog dump of a hang (stage 4: mirror memset blocked) |
| `def39_def40_unguarded.txt` | CE-vs-SM probe 2 (pitched HtoD copies use the copy engine); DEF-39 gate off/on; foreign-context test WITHOUT the DEF-40 guard (native victim killed 2/2) |
| `def38_def40_guarded.txt` | Gate-off watchdog dumps (one stuck CTA, 8 recovery pulses do not heal it), serialize-only gate 10/10, gate on, S7 on the 5080, foreign-context test WITH the guard (victim unaffected) |
| `regress_5060_round2.txt` | RTX 5060 full regressions of the round (per build), reruns of the display-GPU failures, OBS-20 data |
| `regress_5080_final.txt` | RTX 5080 final validation + full regression (297/297, no Xid) |
| `round3_evidence.txt` | Round 3: injected pulses (OBS-21), DEF-40 synchronous check, DEF-41 peer rotation, corrected multi-queue test (T-30), OBS-23 characterization and starvation guard, co-running foreign context experiments, OBS-20 / T-29 reproduction, DEF-42 deadlock stacks |
| `regress_round3.txt` | Round 3 final regressions: RTX 5080 288/288 (pulses active), RTX 5060 287/288 (pulses withheld on the desktop GPU; hazard demo reports SKIP) |
