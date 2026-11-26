// DEF-09 (tier-1 item 1): preemption inside CUDA graphs by graph flattening.
//
// A CUDA graph is launched by the driver as one unit, so XSched could only preempt at the graph
// boundary (Level-1): a suspend waited for the whole graph (10-31 ms for torch.compile
// reduce-overhead graphs). Flattening replays a graph on an XQueue as its individual commands, in
// a topological order of the graph, so every kernel goes through the normal Level-3 path (Entry
// Guardian instrumentation, trap eviction, restore) and the queue can be preempted between and
// inside the graph's kernels.
//
// - At instantiation (cuGraphInstantiate*), while the CUgraph still exists (PyTorch destroys it
//   right after instantiating), the graph is walked: kernel nodes keep function, geometry, dynamic
//   shared memory and a byte copy of their argument values; memset / memcpy nodes keep their
//   parameters; empty nodes are dropped; child graphs are inlined. Any other node type (host, event,
//   external semaphore, memory alloc / free, conditional, batch mem-op), launches that pass
//   arguments through `extra`, and cluster / cooperative kernels make the graph non-flattenable.
// - Any later modification of the executable graph (cuGraphExecUpdate, cuGraphExec*SetParams,
//   cuGraphNodeSetEnabled) marks it non-flattenable: it is then launched as a whole, as before.
// - At cuGraphLaunch on an XQueue, the plan is submitted node by node when the policy allows:
//   XSCHED_CUDA_GRAPH_FLATTEN = 0 (off), 1 (default: XQueues at Level-3 or above), 2 (all XQueues).
//   Parallel branches of the graph run sequentially on the stream (a valid schedule).
// - DEF-32: every instantiated graph also records the CUfunctions of all its kernel nodes (whatever
//   the flattening policy). A kernel node runs with the Guardian parameter window its function holds
//   when the graph is launched; whole-graph launches (on a Level-3 XQueue, or pass-through on an
//   ordinary stream) clear those windows first, so a node never runs as an earlier Mode 5 launch.
// - Duration-aware flattening: flattening costs CPU per kernel (a decode-step graph of a small LLM
//   has hundreds of microsecond-scale kernels), and a graph shorter than the latency target gains
//   little from it. With XSCHED_CUDA_GRAPH_FLATTEN_MIN_US > 0, the first launch of a graph on an
//   XQueue is a whole-graph launch timed with CUDA events; the graph is flattened from then on only
//   if it ran for at least that long, otherwise it keeps being launched as a whole. A timing inflated
//   by a preemption classifies the graph as long (the more preemptible side).

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuxtra/cuxtra.h>

#include "xsched/utils/log.h"
#include "xsched/utils/common.h"
#include "xsched/cuda/shim/shim.h"
#include "xsched/preempt/xqueue/xqueue.h"
#include "xsched/cuda/hal/level3/sm120_trap_engine.h"

namespace xsched::cuda
{

namespace
{

struct NodePlan
{
    enum Kind { kKernel, kMemset, kMemcpy } kind = kKernel;
    // kernel
    CUfunction func = nullptr;
    unsigned int gdx = 1, gdy = 1, gdz = 1, bdx = 1, bdy = 1, bdz = 1, shm = 0;
    std::vector<char> arg_blob;          // argument values, laid out at their parameter offsets
    std::vector<size_t> arg_offsets;     // one per parameter
    // memset / memcpy
    CUDA_MEMSET_NODE_PARAMS memset{};
    CUDA_MEMCPY3D memcpy{};
};

struct GraphPlan
{
    std::shared_ptr<std::vector<CUfunction>> funcs;   // all kernel-node functions (DEF-32)
    bool flattenable = true;
    std::string reason;
    std::vector<NodePlan> nodes;
    size_t kernels = 0;
    bool logged = false;
    // duration-aware flattening (XSCHED_CUDA_GRAPH_FLATTEN_MIN_US)
    std::mutex probe_mtx;
    int duration = 0;               // 0 unknown, 1 long (flatten), 2 short (launch whole)
    bool probing = false;
    CUevent beg = nullptr, end = nullptr;
    std::shared_ptr<CudaEventRecordCommand> end_cmd;   // the probe's end record, launched by the XQueue
    float measured_ms = 0.0f;
    ~GraphPlan()
    {
        if (beg != nullptr) Driver::EventDestroy_v2(beg);
        if (end != nullptr) Driver::EventDestroy_v2(end);
    }
};

std::mutex g_mtx;
// Leaked (DEF-33): GraphPlan destructors destroy CUDA events; never run them during static
// destruction at process exit.
std::unordered_map<CUgraphExec, std::shared_ptr<GraphPlan>> &g_plans = *new std::unordered_map<CUgraphExec, std::shared_ptr<GraphPlan>>();
std::atomic<uint64_t> g_flat_launches{0}, g_whole_launches{0}, g_flat_cmds{0}, g_unflattenable{0};
std::atomic<uint64_t> g_short_launches{0};

int64_t MinUs()
{
    static const int64_t v = [] {
        const char *e = std::getenv("XSCHED_CUDA_GRAPH_FLATTEN_MIN_US");
        return (e == nullptr || e[0] == '\0') ? int64_t(0) : static_cast<int64_t>(std::atoll(e));
    }();
    return v;
}

int Policy()
{
    static const int p = [] {
        const char *v = std::getenv("XSCHED_CUDA_GRAPH_FLATTEN");
        if (v == nullptr || v[0] == '\0') return 1;
        return std::atoi(v);
    }();
    return p;
}

bool Fail(GraphPlan &plan, const std::string &why)
{
    plan.flattenable = false;
    if (plan.reason.empty()) plan.reason = why;
    return false;
}

bool AddKernel(GraphPlan &plan, CUgraphNode node)
{
    CUDA_KERNEL_NODE_PARAMS p{};
    if (Driver::GraphKernelNodeGetParams_v2(node, &p) != CUDA_SUCCESS) return Fail(plan, "kernel params unreadable");
    if (p.extra != nullptr) return Fail(plan, "kernel arguments passed through `extra`");
    CUfunction f = p.func != nullptr ? p.func : reinterpret_cast<CUfunction>(p.kern);
    if (f == nullptr) return Fail(plan, "kernel node without function");

    CUkernelNodeAttrValue v{};
    if (Driver::GraphKernelNodeGetAttribute(node, CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, &v) == CUDA_SUCCESS &&
        v.clusterDim.x * v.clusterDim.y * v.clusterDim.z > 1) {
        return Fail(plan, "thread-block cluster kernel");
    }
    std::memset(&v, 0, sizeof(v));
    if (Driver::GraphKernelNodeGetAttribute(node, CU_LAUNCH_ATTRIBUTE_COOPERATIVE, &v) == CUDA_SUCCESS &&
        v.cooperative != 0) {
        return Fail(plan, "cooperative kernel");
    }

    NodePlan n;
    n.kind = NodePlan::kKernel;
    n.func = f;
    n.gdx = p.gridDimX; n.gdy = p.gridDimY; n.gdz = p.gridDimZ;
    n.bdx = p.blockDimX; n.bdy = p.blockDimY; n.bdz = p.blockDimZ;
    n.shm = p.sharedMemBytes;
    CUfunction fh = cuXtraKernelGetFunction(reinterpret_cast<CUkernel>(f));
    const size_t cnt = fh != nullptr ? cuXtraGetParamCount(fh) : 0;
    if (cnt > 0) {
        if (p.kernelParams == nullptr) return Fail(plan, "kernel node without argument pointers");
        size_t total = 0;
        std::vector<size_t> sizes(cnt);
        n.arg_offsets.resize(cnt);
        for (size_t i = 0; i < cnt; ++i) {
            size_t off = 0, sz = 0;
            cuXtraGetParamInfo(fh, i, &off, &sz, nullptr);
            n.arg_offsets[i] = off;
            sizes[i] = sz;
            if (off + sz > total) total = off + sz;
        }
        n.arg_blob.assign(total == 0 ? 1 : total, 0);
        for (size_t i = 0; i < cnt; ++i) {
            std::memcpy(n.arg_blob.data() + n.arg_offsets[i], p.kernelParams[i], sizes[i]);
        }
    }
    plan.nodes.push_back(std::move(n));
    plan.kernels++;
    return true;
}

void CollectFuncs(CUgraph graph, int depth, std::unordered_set<CUfunction> &out)
{
    if (depth > 8) return;
    size_t n = 0;
    if (Driver::GraphGetNodes(graph, nullptr, &n) != CUDA_SUCCESS || n == 0) return;
    std::vector<CUgraphNode> nodes(n);
    if (Driver::GraphGetNodes(graph, nodes.data(), &n) != CUDA_SUCCESS) return;
    for (size_t i = 0; i < n; ++i) {
        CUgraphNodeType type;
        if (Driver::GraphNodeGetType(nodes[i], &type) != CUDA_SUCCESS) continue;
        if (type == CU_GRAPH_NODE_TYPE_KERNEL) {
            CUDA_KERNEL_NODE_PARAMS p{};
            if (Driver::GraphKernelNodeGetParams_v2(nodes[i], &p) != CUDA_SUCCESS) continue;
            CUfunction f = p.func != nullptr ? p.func : reinterpret_cast<CUfunction>(p.kern);
            CUfunction fh = f != nullptr ? cuXtraKernelGetFunction(reinterpret_cast<CUkernel>(f)) : nullptr;
            if (fh != nullptr) out.insert(fh);
        } else if (type == CU_GRAPH_NODE_TYPE_GRAPH) {
            CUgraph child = nullptr;
            if (Driver::GraphChildGraphNodeGetGraph(nodes[i], &child) == CUDA_SUCCESS) CollectFuncs(child, depth + 1, out);
        }
    }
}

bool Walk(GraphPlan &plan, CUgraph graph, int depth)
{
    if (depth > 8) return Fail(plan, "child graphs nested too deeply");
    size_t n = 0;
    if (Driver::GraphGetNodes(graph, nullptr, &n) != CUDA_SUCCESS) return Fail(plan, "cannot list nodes");
    std::vector<CUgraphNode> nodes(n);
    if (n > 0 && Driver::GraphGetNodes(graph, nodes.data(), &n) != CUDA_SUCCESS) return Fail(plan, "cannot list nodes");
    size_t e = 0;
    if (Driver::GraphGetEdges(graph, nullptr, nullptr, &e) != CUDA_SUCCESS) return Fail(plan, "cannot list edges");
    std::vector<CUgraphNode> from(e), to(e);
    if (e > 0 && Driver::GraphGetEdges(graph, from.data(), to.data(), &e) != CUDA_SUCCESS) {
        return Fail(plan, "cannot list edges");
    }

    // Kahn's algorithm; ties broken by the driver's node order (the capture order).
    std::unordered_map<CUgraphNode, size_t> index;
    for (size_t i = 0; i < n; ++i) index[nodes[i]] = i;
    std::vector<std::vector<size_t>> succ(n);
    std::vector<size_t> indeg(n, 0);
    for (size_t i = 0; i < e; ++i) {
        auto a = index.find(from[i]), b = index.find(to[i]);
        if (a == index.end() || b == index.end()) return Fail(plan, "edge to unknown node");
        succ[a->second].push_back(b->second);
        indeg[b->second]++;
    }
    std::map<size_t, bool> ready;          // ordered by node index
    for (size_t i = 0; i < n; ++i) if (indeg[i] == 0) ready[i] = true;
    size_t emitted = 0;
    while (!ready.empty()) {
        const size_t i = ready.begin()->first;
        ready.erase(ready.begin());
        emitted++;
        CUgraphNodeType type;
        if (Driver::GraphNodeGetType(nodes[i], &type) != CUDA_SUCCESS) return Fail(plan, "unknown node type");
        switch (type) {
        case CU_GRAPH_NODE_TYPE_KERNEL:
            if (!AddKernel(plan, nodes[i])) return false;
            break;
        case CU_GRAPH_NODE_TYPE_MEMSET: {
            NodePlan m;
            m.kind = NodePlan::kMemset;
            if (Driver::GraphMemsetNodeGetParams(nodes[i], &m.memset) != CUDA_SUCCESS) return Fail(plan, "memset params");
            if (m.memset.elementSize != 1 && m.memset.elementSize != 2 && m.memset.elementSize != 4) {
                return Fail(plan, "memset element size");
            }
            plan.nodes.push_back(std::move(m));
            break;
        }
        case CU_GRAPH_NODE_TYPE_MEMCPY: {
            NodePlan m;
            m.kind = NodePlan::kMemcpy;
            if (Driver::GraphMemcpyNodeGetParams(nodes[i], &m.memcpy) != CUDA_SUCCESS) return Fail(plan, "memcpy params");
            plan.nodes.push_back(std::move(m));
            break;
        }
        case CU_GRAPH_NODE_TYPE_EMPTY:
            break;
        case CU_GRAPH_NODE_TYPE_GRAPH: {
            CUgraph child = nullptr;
            if (Driver::GraphChildGraphNodeGetGraph(nodes[i], &child) != CUDA_SUCCESS) return Fail(plan, "child graph");
            if (!Walk(plan, child, depth + 1)) return false;
            break;
        }
        default:
            return Fail(plan, "node type " + std::to_string(static_cast<int>(type)) + " cannot be replayed");
        }
        for (size_t s : succ[i]) {
            if (--indeg[s] == 0) ready[s] = true;
        }
    }
    if (emitted != n) return Fail(plan, "graph has a cycle");
    return true;
}

void Submit(const NodePlan &n, const std::shared_ptr<preempt::XQueue> &xq)
{
    switch (n.kind) {
    case NodePlan::kKernel: {
        std::vector<void *> args(n.arg_offsets.size());
        for (size_t i = 0; i < args.size(); ++i) {
            args[i] = const_cast<char *>(n.arg_blob.data()) + n.arg_offsets[i];
        }
        xq->Submit(std::make_shared<CudaKernelLaunchCommand>(n.func, n.gdx, n.gdy, n.gdz, n.bdx, n.bdy, n.bdz,
                                                             n.shm, args.empty() ? nullptr : args.data(),
                                                             nullptr, true));
        break;
    }
    case NodePlan::kMemset: {
        const auto &m = n.memset;
        const size_t pitch = m.height > 1 ? m.pitch : m.width * m.elementSize;
        if (m.elementSize == 1) {
            xq->Submit(std::make_shared<CudaMemsetD2D8Command>(m.dst, pitch, static_cast<unsigned char>(m.value), m.width, m.height));
        } else if (m.elementSize == 2) {
            xq->Submit(std::make_shared<CudaMemsetD2D16Command>(m.dst, pitch, static_cast<unsigned short>(m.value), m.width, m.height));
        } else {
            xq->Submit(std::make_shared<CudaMemsetD2D32Command>(m.dst, pitch, m.value, m.width, m.height));
        }
        break;
    }
    case NodePlan::kMemcpy:
        xq->Submit(std::make_shared<CudaMemcpy3DV2Command>(&n.memcpy));
        break;
    }
}

} // namespace

void GraphFlattenOnInstantiate(CUgraphExec exec, CUgraph graph)
{
    if (exec == nullptr || graph == nullptr) return;
    auto plan = std::make_shared<GraphPlan>();
    std::unordered_set<CUfunction> funcs;
    CollectFuncs(graph, 0, funcs);
    plan->funcs = std::make_shared<std::vector<CUfunction>>(funcs.begin(), funcs.end());
    if (Policy() == 0) Fail(*plan, "flattening disabled (XSCHED_CUDA_GRAPH_FLATTEN=0)");
    else Walk(*plan, graph, 0);
    if (!plan->flattenable) g_unflattenable.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(g_mtx);
    g_plans[exec] = plan;
}

void GraphFlattenInvalidate(CUgraphExec exec, const char *why, CUgraph new_graph, CUfunction new_func)
{
    std::unordered_set<CUfunction> added;
    if (new_graph != nullptr) CollectFuncs(new_graph, 0, added);
    if (new_func != nullptr) {
        CUfunction fh = cuXtraKernelGetFunction(reinterpret_cast<CUkernel>(new_func));
        if (fh != nullptr) added.insert(fh);
    }
    std::lock_guard<std::mutex> lock(g_mtx);
    auto it = g_plans.find(exec);
    if (it == g_plans.end()) return;
    // Replace rather than mutate: a launch may be iterating over the old plan / function list.
    auto plan = std::make_shared<GraphPlan>();
    plan->flattenable = false;
    plan->reason = it->second->flattenable ? why : it->second->reason;
    std::unordered_set<CUfunction> funcs(added);
    if (it->second->funcs) funcs.insert(it->second->funcs->begin(), it->second->funcs->end());
    plan->funcs = std::make_shared<std::vector<CUfunction>>(funcs.begin(), funcs.end());
    it->second = plan;
}

std::shared_ptr<const std::vector<CUfunction>> GraphFuncs(CUgraphExec exec)
{
    std::lock_guard<std::mutex> lock(g_mtx);
    auto it = g_plans.find(exec);
    return it == g_plans.end() ? nullptr : it->second->funcs;
}

std::shared_ptr<CudaGraphLaunchCommand> WholeGraphCommand(CUgraphExec exec)
{
    auto cmd = std::make_shared<CudaGraphLaunchCommand>(exec);
    cmd->graph_funcs = GraphFuncs(exec);
    return cmd;
}

void GraphFlattenOnDestroy(CUgraphExec exec)
{
    std::lock_guard<std::mutex> lock(g_mtx);
    g_plans.erase(exec);
}

bool GraphFlattenSubmit(CUgraphExec exec, const std::shared_ptr<preempt::XQueue> &xq)
{
    const int policy = Policy();
    if (policy == 0) return false;
    if (policy == 1 && xq->GetPreemptLevel() < kPreemptLevelInterrupt) return false;
    std::shared_ptr<GraphPlan> plan;
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        auto it = g_plans.find(exec);
        if (it != g_plans.end()) plan = it->second;
    }
    if (plan == nullptr) return false;
    if (!plan->logged) {
        plan->logged = true;
        if (plan->flattenable) {
            XINFO("CUDA graph %p flattened for in-graph preemption: %zu commands (%zu kernels)",
                  (void *)exec, plan->nodes.size(), plan->kernels);
        } else {
            XINFO("CUDA graph %p launched as a whole (level-1 boundary): %s", (void *)exec, plan->reason.c_str());
        }
    }
    if (!plan->flattenable) {
        g_whole_launches.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (MinUs() > 0) {
        std::lock_guard<std::mutex> lk(plan->probe_mtx);
        // An event that was never recorded reads as complete: query only after the XQueue launched
        // the end record (dropped commands also reach "completed" without recording: timed as long).
        if (plan->probing && plan->end_cmd->GetState() >= preempt::kCommandStateInFlight &&
            Driver::EventQuery(plan->end) == CUDA_SUCCESS) {
            float ms = 0.0f;
            const CUresult er = Driver::EventElapsedTime(&ms, plan->beg, plan->end);
            const bool ok = er == CUDA_SUCCESS;
            if (!ok) XWARN("CUDA graph %p: timing the probe launch failed (CUresult %d)", (void *)exec, (int)er);
            plan->measured_ms = ms;
            plan->duration = (!ok || ms * 1000.0f >= static_cast<float>(MinUs())) ? 1 : 2;
            plan->probing = false;
            plan->end_cmd = nullptr;
            XINFO("CUDA graph %p ran %.3f ms as a whole: %s", (void *)exec, ms,
                  plan->duration == 1 ? "flattened from now on" : "kept as a whole (below XSCHED_CUDA_GRAPH_FLATTEN_MIN_US)");
        }
        if (plan->duration != 1) {
            if (plan->duration == 0 && !plan->probing) {
                // Timed whole-graph launch (the probe).
                if (plan->beg == nullptr) Driver::EventCreate(&plan->beg, CU_EVENT_DEFAULT);
                if (plan->end == nullptr) Driver::EventCreate(&plan->end, CU_EVENT_DEFAULT);
                plan->probing = true;
                xq->Submit(std::make_shared<CudaEventRecordCommand>(plan->beg));
                auto whole = std::make_shared<CudaGraphLaunchCommand>(exec);
                whole->graph_funcs = plan->funcs;
                xq->Submit(whole);
                plan->end_cmd = std::make_shared<CudaEventRecordCommand>(plan->end);
                xq->Submit(plan->end_cmd);
                g_short_launches.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            g_short_launches.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
    }
    for (const auto &n : plan->nodes) Submit(n, xq);
    g_flat_launches.fetch_add(1, std::memory_order_relaxed);
    g_flat_cmds.fetch_add(plan->nodes.size(), std::memory_order_relaxed);
    return true;
}

} // namespace xsched::cuda

/// Counters for tests: [flattened launches, whole-graph launches of non-flattenable graphs on XQueues,
/// commands submitted by flattening, non-flattenable graphs instantiated, whole-graph launches of
/// flattenable graphs kept whole by XSCHED_CUDA_GRAPH_FLATTEN_MIN_US (incl. the timed probe)].
/// Returns the number of counters written.
EXPORT_C_FUNC int XCudaGraphFlattenStats(uint64_t *out, int n, int reset)
{
    using namespace xsched::cuda;
    std::atomic<uint64_t> *c[5] = {&g_flat_launches, &g_whole_launches, &g_flat_cmds, &g_unflattenable,
                                   &g_short_launches};
    int k = n < 5 ? n : 5;
    for (int i = 0; i < k; ++i) out[i] = reset ? c[i]->exchange(0) : c[i]->load();
    return k;
}

// ------------------------------------------------------------------------------- driver entry points
namespace xsched::cuda
{

#define XSCHED_GF_RECORD(call) \
    CUresult r = call; \
    if (r == CUDA_SUCCESS && phGraphExec != nullptr) GraphFlattenOnInstantiate(*phGraphExec, hGraph); \
    return r;

CUresult XGraphInstantiateWithFlags(CUgraphExec *phGraphExec, CUgraph hGraph, unsigned long long flags)
{ XSCHED_GF_RECORD(Driver::GraphInstantiateWithFlags(phGraphExec, hGraph, flags)) }

CUresult XGraphInstantiateWithParams(CUgraphExec *phGraphExec, CUgraph hGraph, CUDA_GRAPH_INSTANTIATE_PARAMS *p)
{ XSCHED_GF_RECORD(Driver::GraphInstantiateWithParams(phGraphExec, hGraph, p)) }

CUresult XGraphInstantiateWithParams_ptsz(CUgraphExec *phGraphExec, CUgraph hGraph, CUDA_GRAPH_INSTANTIATE_PARAMS *p)
{ XSCHED_GF_RECORD(Driver::GraphInstantiateWithParams_ptsz(phGraphExec, hGraph, p)) }

CUresult XGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph, CUgraphNode *err, char *log, size_t size)
{ XSCHED_GF_RECORD(Driver::GraphInstantiate(phGraphExec, hGraph, err, log, size)) }

CUresult XGraphInstantiate_v2(CUgraphExec *phGraphExec, CUgraph hGraph, CUgraphNode *err, char *log, size_t size)
{ XSCHED_GF_RECORD(Driver::GraphInstantiate_v2(phGraphExec, hGraph, err, log, size)) }

#undef XSCHED_GF_RECORD

CUresult XGraphExecDestroy(CUgraphExec exec)
{
    GraphFlattenOnDestroy(exec);
    return Driver::GraphExecDestroy(exec);
}

CUresult XGraphExecUpdate_v2(CUgraphExec exec, CUgraph g, CUgraphExecUpdateResultInfo *info)
{
    GraphFlattenInvalidate(exec, "updated with cuGraphExecUpdate", g);
    return Driver::GraphExecUpdate_v2(exec, g, info);
}

CUresult XGraphExecKernelNodeSetParams_v2(CUgraphExec exec, CUgraphNode node, const CUDA_KERNEL_NODE_PARAMS *p)
{
    GraphFlattenInvalidate(exec, "kernel node parameters changed after instantiation", nullptr,
                           p != nullptr ? (p->func != nullptr ? p->func : reinterpret_cast<CUfunction>(p->kern)) : nullptr);
    return Driver::GraphExecKernelNodeSetParams_v2(exec, node, p);
}

CUresult XGraphExecMemcpyNodeSetParams(CUgraphExec exec, CUgraphNode node, const CUDA_MEMCPY3D *p, CUcontext ctx)
{
    GraphFlattenInvalidate(exec, "memcpy node parameters changed after instantiation");
    return Driver::GraphExecMemcpyNodeSetParams(exec, node, p, ctx);
}

CUresult XGraphExecMemsetNodeSetParams(CUgraphExec exec, CUgraphNode node, const CUDA_MEMSET_NODE_PARAMS *p, CUcontext ctx)
{
    GraphFlattenInvalidate(exec, "memset node parameters changed after instantiation");
    return Driver::GraphExecMemsetNodeSetParams(exec, node, p, ctx);
}

CUresult XGraphNodeSetEnabled(CUgraphExec exec, CUgraphNode node, unsigned int enabled)
{
    GraphFlattenInvalidate(exec, "node enabled state changed after instantiation");
    return Driver::GraphNodeSetEnabled(exec, node, enabled);
}

CUresult XGraphExecNodeSetParams(CUgraphExec exec, CUgraphNode node, CUgraphNodeParams *p)
{
    GraphFlattenInvalidate(exec, "node parameters changed after instantiation", nullptr,
                           (p != nullptr && p->type == CU_GRAPH_NODE_TYPE_KERNEL)
                               ? (p->kernel.func != nullptr ? p->kernel.func : reinterpret_cast<CUfunction>(p->kernel.kern))
                               : nullptr);
    return Driver::GraphExecNodeSetParams(exec, node, p);
}

static CUresult PassThroughGraphLaunch(CUgraphExec exec, CUstream stream)
{
    // DEF-32 for graphs on ordinary streams (e.g. a high-priority stream in a process whose LP queue
    // runs Mode 5): clear the graph's Guardian windows under the LaunchMutex before launching.
    if (!sm120::Sm120StreamContext::AnyParamsSet()) return Driver::GraphLaunch(exec, stream);
    auto funcs = GraphFuncs(exec);
    if (funcs == nullptr || funcs->empty()) return Driver::GraphLaunch(exec, stream);
    CUcontext ctx = nullptr;
    if (Driver::CtxGetCurrent(&ctx) != CUDA_SUCCESS || ctx == nullptr) {
        if (stream == nullptr || stream == CU_STREAM_LEGACY || stream == CU_STREAM_PER_THREAD ||
            Driver::StreamGetCtx(stream, &ctx) != CUDA_SUCCESS || ctx == nullptr) {
            return Driver::GraphLaunch(exec, stream);
        }
    }
    std::lock_guard<std::mutex> lock(sm120::Sm120TrapEngine::LaunchMutex(ctx));
    for (CUfunction f : *funcs) sm120::Sm120StreamContext::ClearDebuggerParams(f);
    return Driver::GraphLaunch(exec, stream);
}

static CUresult XGraphLaunchImpl(CUgraphExec exec, CUstream stream)
{
    auto xq = preempt::HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) return PassThroughGraphLaunch(exec, stream);
    ScopedStreamContext scoped_ctx(stream);
    if (GraphFlattenSubmit(exec, xq)) return CUDA_SUCCESS;
    xq->Submit(WholeGraphCommand(exec));
    return CUDA_SUCCESS;
}

CUresult XGraphLaunch(CUgraphExec exec, CUstream stream)
{
    XDEBG("XGraphLaunch(exec: %p, stream: %p)", (void *)exec, (void *)stream);
    CHECK_STREAM(stream, PassThroughGraphLaunch(exec, stream));
    return XGraphLaunchImpl(exec, stream);
}

CUresult XGraphLaunch_ptsz(CUgraphExec exec, CUstream stream)
{
    XDEBG("XGraphLaunch_ptsz(exec: %p, stream: %p)", (void *)exec, (void *)stream);
    CHECK_STREAM_PTSZ(stream, PassThroughGraphLaunch(exec, stream));
    return XGraphLaunchImpl(exec, stream);
}

} // namespace xsched::cuda
