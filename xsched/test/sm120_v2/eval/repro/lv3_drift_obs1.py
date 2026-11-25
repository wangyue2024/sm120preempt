"""Find the first call at which an in-place op diverges under an Lv3 XQueue (no preemption)."""
import sys, importlib.util, ctypes, torch
spec = importlib.util.spec_from_file_location("b", sys.argv[1]); b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
op_name, n, level = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
torch.manual_seed(1234); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
lp = torch.cuda.Stream()
with torch.cuda.stream(lp): op = b.OPS[op_name][2]()
torch.cuda.synchronize()
def seq():
    snaps = []
    with torch.cuda.stream(lp):
        op.reset()
        for _ in range(n):
            op.run(); snaps.append([t.detach().clone() for t in op.state()])
    lp.synchronize(); return snaps
gold = seq(); gold2 = seq()
print("native self-consistent:", all(all(torch.equal(x, y) for x, y in zip(a, c)) for a, c in zip(gold, gold2)))
xs = b.XSchedBridge(); hwq, xq = xs.create_xqueue(lp, level)
got = seq()
first = next((i for i, (a, c) in enumerate(zip(got, gold)) if not all(torch.equal(x, y) for x, y in zip(a, c))), None)
print(f"{op_name} level={level}: first divergent call = {first} of {n}")
if first is not None:
    for k, (x, y) in enumerate(zip(got[first], gold[first])):
        d = (x.float() - y.float()).abs(); print(f"  state[{k}] max_err={d.max().item():.3e} n_diff={(d > 0).sum().item()}/{d.numel()}")
    rs = xs.route_stats(); print("  route stats:", rs)
xs.destroy(hwq, xq)
