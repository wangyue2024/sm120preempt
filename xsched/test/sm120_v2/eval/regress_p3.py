#!/usr/bin/env python3
"""
One-command SM120 regression pipeline (Phase 3.2).

  1. build native test binaries:   bash build_regress_bins.sh <env_file>
  2. generate a test plan:         python3 regress_p3.py plan --env <env_file> --suites quick > plan.txt
  3. run it (root, GPU-guarded):   printf '%s\\n' "$PW" | sudo -S -p '' bash p3_guarded_run.sh plan.txt run.log
  4. summarize:                    python3 regress_p3.py summarize run.log [--jsonl-dir DIR] [--md out.md]

<env_file> is a bash file exporting: XSCHED (install prefix), OUT (binary dir), PY (python), EVAL (this
directory), CUDA_HOME_LOCAL, LD_LIBRARY_PATH (XSched lib first), CUDA_MPS_PIPE_DIRECTORY, optional
HF_HOME / TRITON_CACHE_DIR. See regress_env.example.sh.

Suite groups:
  syn    synthetic trap-frame tests (partial exit, convergence/register sweep, DEF-21 stress)
  s1     Suite 1 closed-source libs + Blackwell HW features (NP=2) + LIB-08 x10
  s4     Suite 4 extended CUDA libraries
  s5     Suite 5 production DNNs        (python)
  s6     Suite 6 production LLMs        (python, Mode 5)
  s7     Suite 7 production operators   (lv3 + lv3m5; s7full = native/shim/lv1/lv3/lv3m5)
         lv3m5 = Level-3 with the DEF-17 short-CTA router off (pure Mode 5)
  s8     Suite 8 real-model serving     (needs HF weights; native/solo/lv1/lv3 x policies)
  acc    Phase 3.2 acceptance repeats (C1 x10 np=2/3, 12-kernel chain, HW-06 x20, dnn06 x5)
  quick  = syn,s1,s4,s7          full = syn,s1,s4,s5,s6,s7full,s8,acc
"""
import argparse
import json
import os
import re
import sys
from collections import OrderedDict

LIBS = ["LIB-01", "LIB-02", "LIB-03", "LIB-04", "LIB-05", "LIB-06", "LIB-07",
        "HW-01", "HW-02", "HW-03", "HW-04", "HW-05", "HW-06", "HW-07"]
LIBS4 = ["LIB-09", "LIB-10", "LIB-11", "LIB-12", "LIB-13", "LIB-14"]
DNN = ["dnn01", "dnn02", "dnn03", "dnn04", "dnn05", "dnn06", "dnn07", "dnn08"]
LLM = ["llm01", "llm02", "llm03", "llm04", "llm05", "llm08", "llm06:m5", "llm07:m5"]
GROUPS = {"quick": ["syn", "s1", "s4", "s7"],
          "full": ["syn", "s1", "s4", "s5", "s6", "s7full", "s8", "acc"]}


def ops_list():
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    src = open(os.path.join(here, "bench_prod_ops_p3.py")).read()
    return re.findall(r'^\s*@register\("([a-z0-9_]+)"|^\s*register\("([a-z0-9_]+)"', src, re.M)


def op_names():
    names = []
    for a, b in ops_list():
        names.append(a or b)
    return names


def plan(args):
    env = os.path.abspath(args.env)
    E = f"source {env} &&"
    nat = "LD_LIBRARY_PATH=$CUDA_HOME_LOCAL/lib64:/usr/lib/x86_64-linux-gnu"
    out_dir = args.result_dir
    groups = []
    for g in args.suites.split(","):
        groups += GROUPS.get(g, [g])
    lines = [f"# regress_p3 plan: suites={args.suites} env={env}"]

    def L(label, tmo, cmd):
        lines.append(f"{label} | {tmo} | {E} {cmd}")

    # Mechanism suites must exercise the Mode 5 trap path on every kernel, so the DEF-17 short-CTA
    # Level-1 router (a production performance policy) is disabled for them.
    M5 = "XSCHED_SM120_SHORT_CTA_LV1=0"

    for g in OrderedDict.fromkeys(groups):
        if g == "syn":
            L("SYN-partial", 300, f"{M5} $OUT/test_partial_exit_p3 all 5 3")
            L("SYN-convbar-np3", 600, f"{M5} $OUT/test_convbar_p3 all 3 3")
            L("SYN-stress-g2", 900, f"{M5} GRID_MULT=2 CHAIN=6 GAP_US=1000 TARGET_MS=4 $OUT/test_convbar_p3 all 3 10")
            L("SYN-stress-g8", 900, f"{M5} GRID_MULT=8 CHAIN=6 GAP_US=1000 TARGET_MS=4 $OUT/test_convbar_p3 all 3 10")
            L("SYN-stress-gap300", 900, f"{M5} GRID_MULT=8 CHAIN=4 GAP_US=300 $OUT/test_convbar_p3 all 3 8")
            for m in ("lv1", "lv3"):
                L(f"SYN-percall-{m}", 120, f"$OUT/test_percall_p3 {m} 2000")
        elif g == "acc":
            L("ACC-C1-np2", 600, f"{M5} $OUT/test_convbar_p3 C1 10 2")
            L("ACC-C1-np3", 600, f"{M5} $OUT/test_convbar_p3 C1 10 3")
            L("ACC-chain12", 900, f"{M5} CHAIN=12 $OUT/test_convbar_p3 C1 3 3")
            for i in range(1, 21):
                L(f"ACC-HW06-r{i}", 60, f"{M5} TEST_NP=2 $OUT/test_libs_and_features HW-06")
            for i in range(1, 6):
                L(f"ACC-dnn06-r{i}", 240, "$PY $EVAL/test_real_dnn_prod_p3.py dnn06")
        elif g == "s1":
            for c in LIBS + ["LIB-08"]:
                L(f"S1-{c}", 60, f"{M5} TEST_NP=2 $OUT/test_libs_and_features {c}")
            for i in range(1, args.lib08_reps + 1):
                L(f"S1-LIB08-r{i}", 60, f"{M5} TEST_NP=2 $OUT/test_libs_and_features LIB-08")
        elif g == "s4":
            for c in LIBS4:
                L(f"S4-{c}", 90, f"{M5} TEST_NP=2 $OUT/test_real_cuda_libs_p3 {c}")
        elif g == "s5":
            for c in DNN:
                L(f"S5-{c}", 240, f"$PY $EVAL/test_real_dnn_prod_p3.py {c}")
        elif g == "s6":
            for c in LLM:
                L(f"S6-{c}", 300, f"$PY $EVAL/test_real_llm_prod_p3.py {c}")
        elif g in ("s7", "s7full"):
            res = os.path.join(out_dir, "suite7.jsonl")
            for op in op_names():
                modes = ["native", "shim", "lv1", "lv3", "lv3m5"] if g == "s7full" else ["lv3", "lv3m5"]
                for m in modes:
                    # lv3m5 = Level-3 with the short-CTA router off (Mode 5 on every kernel); its
                    # results go to suite7_m5.jsonl to keep the mode column of suite7.jsonl intact.
                    pre = nat + " " if m == "native" else (M5 + " " if m == "lv3m5" else "")
                    mm = "lv3" if m == "lv3m5" else m
                    out = res.replace("suite7.jsonl", "suite7_m5.jsonl") if m == "lv3m5" else res
                    L(f"S7-{op}-{m}", 300, f"{pre}$PY $EVAL/bench_prod_ops_p3.py run --op {op} --mode {mm} --out {out}")
        elif g == "s8":
            res = os.path.join(out_dir, "suite8.jsonl")
            for lpk in ("ppl", "prefill"):
                L(f"S8-{lpk}-solo", 400, f"{nat} HF_HUB_OFFLINE=1 $PY $EVAL/bench_real_serving_p3.py run --mode solo --lp {lpk} --out {res}")
                L(f"S8-{lpk}-native", 400, f"{nat} HF_HUB_OFFLINE=1 $PY $EVAL/bench_real_serving_p3.py run --mode native --lp {lpk} --out {res}")
                for m in ("lv1", "lv3"):
                    for pol in ("request", "prefill", "slo"):
                        L(f"S8-{lpk}-{m}-{pol}", 400, f"HF_HUB_OFFLINE=1 $PY $EVAL/bench_real_serving_p3.py run --mode {m} --policy {pol} --lp {lpk} --out {res}")
        else:
            sys.exit(f"unknown suite group: {g}")
    print("\n".join(lines))


RES_PAT = re.compile(r"^\[(?:P3-CONVBAR|P3-PARTIAL|P3-PERCALL|LIB-\d+|HW-\d+|DNN-\d+|LLM-\d+|OPS|SERVE)[^\]]*\].*?\b(PASS|FAIL|ERR)\b")


def summarize(args):
    text = open(args.log, errors="replace").read().splitlines()
    tests = OrderedDict()
    cur = None
    for line in text:
        m = re.match(r"^>>> \[(.+?)\]", line)
        if m:
            cur = m.group(1)
            tests[cur] = {"rc": None, "fails": 0, "passes": 0, "notes": []}
            continue
        m = re.match(r"^<<< \[(.+?)\] rc=(-?\d+)", line)
        if m and m.group(1) in tests:
            tests[m.group(1)]["rc"] = int(m.group(2))
            cur = None
            continue
        if re.match(r"^>>> \[(.+?)\] SKIPPED_GPU_BUSY", line):
            lab = re.match(r"^>>> \[(.+?)\]", line).group(1)
            tests[lab] = {"rc": "skipped", "fails": 0, "passes": 0, "notes": ["GPU busy"]}
            continue
        if cur is None:
            continue
        r = RES_PAT.search(line)
        if r:
            if r.group(1) == "PASS":
                tests[cur]["passes"] += 1
            else:
                tests[cur]["fails"] += 1
                tests[cur]["notes"].append(line.strip()[:160])
        elif "cuda error" in line or "Traceback" in line:
            tests[cur]["notes"].append(line.strip()[:160])

    def ok(t):
        return t["rc"] == 0 and t["fails"] == 0

    suites = OrderedDict()
    for lab, t in tests.items():
        key = lab.split("-")[0]
        s = suites.setdefault(key, {"pass": 0, "total": 0, "bad": []})
        s["total"] += 1
        if ok(t):
            s["pass"] += 1
        else:
            s["bad"].append((lab, t))
    out = ["| suite | pass / total | failing tests |", "|---|---:|---|"]
    for k, s in suites.items():
        bad = ", ".join(f"`{lab}`(rc={t['rc']})" for lab, t in s["bad"]) or "-"
        out.append(f"| {k} | {s['pass']}/{s['total']} | {bad} |")
    total = sum(s["total"] for s in suites.values())
    passed = sum(s["pass"] for s in suites.values())
    out.append(f"| **all** | **{passed}/{total}** | |")
    details = []
    for k, s in suites.items():
        for lab, t in s["bad"]:
            details.append(f"- `{lab}`: rc={t['rc']} " + " / ".join(t["notes"][:3]))
    if details:
        out += ["", "Failure details:", ""] + details
    text_out = "\n".join(out)
    print(text_out)
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(text_out + "\n")
    return 0 if passed == total else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--env", required=True)
    p.add_argument("--suites", default="quick")
    p.add_argument("--result-dir", default=".")
    p.add_argument("--lib08-reps", type=int, default=30)
    s = sp.add_parser("summarize")
    s.add_argument("log")
    s.add_argument("--md")
    a = ap.parse_args()
    if a.cmd == "plan":
        plan(a)
    else:
        sys.exit(summarize(a))


if __name__ == "__main__":
    main()
