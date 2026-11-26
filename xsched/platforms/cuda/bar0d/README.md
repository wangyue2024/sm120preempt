# xsched-bar0d — non-root Level-3 preemption (DEF-10)

SM120 Level-3 preemption needs one hardware action that ordinary users cannot perform: the
evict pulse on the GPU's trap register in BAR0. Everything else (trap-handler patching,
Guardian instrumentation, control memory) works without root. `xsched-bar0d` is a small proxy
that performs only that pulse on behalf of authorized processes.

## Security model

| Property | How |
|---|---|
| Who may preempt | Members of one group (`--group`, default `xsched`): the socket is `root:<group> 0660` |
| What a client can do | One operation, `PULSE` (plus `HELLO`). No register address or value comes from the client |
| What the daemon can touch | Two 4 KiB BAR0 pages (trap register at `0x424684`, flush register at `0x9400`); the rest of BAR0 is never mapped |
| Daemon privileges | Maps the pages and binds the socket as root, then `setgroups(0)`, `setgid/setuid(nobody)`, `no_new_privs`, not dumpable |
| Abuse limit | Per-connection token bucket (`--rate 5000 --burst 64` pulses/s by default); refusals are logged |
| Cross-process safety | Single-threaded event loop: pulse sequences of different processes never interleave |
| Wrong GPU | Clients send `HELLO` and refuse a proxy that serves a different PCI device |

The pulse is a GPU-wide broadcast. An earlier cross-process test (60 pulses while an unrelated
compute process without XSched ran kernels) saw no errors and no slowdown in that process. On an
RTX 5060 that also drives the desktop, however, two multi-wave stress runs coincided with MMU
faults (Xid 31) in desktop graphics processes, which recovered (OBS-8 in
`doc/SM120_V2_PHASE3_4_MULTIPROCESS_ZH.md`); prefer GPUs without a display for Level-3. And a group
member can make the GPU take trap entries as often as the rate limit allows. Treat group
membership as a scheduling privilege: on a shared GPU, grant it only to users whose jobs are
scheduled together.

## Use

```bash
sudo xsched-bar0d --group xsched                 # one GPU: discovers it
sudo xsched-bar0d --bdf 0000:01:00.0 --group xsched   # several GPUs: one daemon each
```

Or install `xsched-bar0d@.service` (instance = BDF). XSched picks the proxy up automatically.

| Variable | Default | Effect |
|---|---|---|
| `XSCHED_SM120_BAR0_PROXY` | `/run/xsched/bar0-<bdf>.sock` | Socket path; `0` disables the proxy |
| `XSCHED_SM120_BAR0_FORCE_PROXY` | `0` | `1`: use the proxy even when BAR0 can be mapped directly (testing) |
| `XSCHED_SM120_BAR0_PROXY_TIMEOUT_MS` | `200` | Reply timeout per pulse |

Root processes map BAR0 directly and do not use the proxy. If neither path works, Level-3 still
runs correctly but a suspend waits for the running kernel to finish (logged once per second).

Cost: one Unix-socket round trip per pulse, 4.5 µs (RTX 5080 host) to 12 µs (RTX 5060 host)
P50; measured suspend latency +2–15 µs versus direct mapping. Protocol checks:
`test/sm120_v2/eval/test_bar0_proxy_p34.py <socket>` (never touches hardware).
