| suite | pass / total | failing tests |
|---|---:|---|
| SYN | 16/16 | - |
| S1 | 45/45 | - |
| S4 | 6/6 | - |
| S5 | 8/8 | - |
| S6 | 9/9 | - |
| S7 | 145/145 | - |
| S8 | 15/16 | `S8-prefill-lv3-slo`(rc=0) |
| S9 | 6/6 | - |
| ACC | 28/28 | - |
| LL | 16/16 | - |
| **all** | **294/295** | |

Failure details:

- `S8-prefill-lv3-slo`: rc=0 [SERVE mode=lv3 policy=slo lp=prefill] FAIL TTFT p50/p99=7.79/50.26ms TPOT p50/p99=6.59/9.75ms hp_exact=20/20 lp_iters=156 bad=1 lp_tput=70.1% hp_duty=25%
