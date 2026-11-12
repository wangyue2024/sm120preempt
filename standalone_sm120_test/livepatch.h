// Install a cave into the live sm_120 .text.traphandler of the current context
// (hook at +0x390, cave in the NOP-filled slot-20 region at tools_pc+0xa00).
#pragma once
#include "arena.h"
#include "sass120.h"
#include <cstring>

struct LivePatch {
    sm120arena::Info I;
    uint64_t cave = 0, orig[2] = {0, 0};
    size_t cave_bytes = 0;
    std::vector<uint64_t> cave_backup;
    bool installed = false;
    CUcontext ctx = nullptr;

    bool probe() {
        cuCtxGetCurrent(&ctx);
        I = sm120arena::probe(ctx, 0);
        if (!I.ok || !I.handler_is_memobj || I.handler.size() < 0x3b0 || I.tools_size < 0xb00) {
            printf("[patch] arena/handler unavailable\n"); return false;
        }
        for (int i = 0; i < 10; i++)
            if (sm120arena::opcode_at(&I.handler[i * 16]) != sm120arena::kTrapHandlerFp[i]) {
                printf("[patch] fingerprint mismatch at +%#x\n", i * 16); return false;
            }
        cave = I.tools_pc + 0xa00;
        memcpy(orig, &I.handler[0x390], 16);
        printf("[patch] handler=%#llx cave=%#llx preempt_buf=%#llx\n", (unsigned long long)I.handler_va,
               (unsigned long long)cave, (unsigned long long)I.preempt_buf);
        return true;
    }
    // max 0x880 bytes of cave (0xa00..0x1280 is NOP-filled)
    bool install(const std::vector<s120::I> &code) {
        auto w = s120::flat(code);
        cave_bytes = w.size() * 8;
        if (cave_bytes > 0x880) { printf("[patch] cave too large\n"); return false; }
        cave_backup.resize(w.size());
        cuXtraMemcpyDtoH(cave_backup.data(), cave, cave_bytes, 0);
        uint64_t o0, o1; memcpy(&o0, &I.handler[0x380], 8); memcpy(&o1, &I.handler[0x388], 8);
        uint64_t enc = (cave - 0x100) >> 8, call[2];
        call[0] = ((enc & 0xffffffffULL) << 32) | (o0 & 0xffffffffULL);
        call[1] = (o1 & ~0xffULL) | ((enc >> 32) & 0xffULL);
        size_t a = cuXtraMemcpyHtoD(cave, w.data(), cave_bytes, 0);
        size_t b = cuXtraMemcpyHtoD(I.handler_va + 0x390, call, 16, 0);
        cuXtraInvalInstrCache(ctx);
        uint64_t rb[2]; cuXtraMemcpyDtoH(rb, I.handler_va + 0x390, 16, 0);
        installed = a == cave_bytes && b == 16 && rb[0] == call[0] && rb[1] == call[1];
        printf("[patch] install cave=%zu hook=%zu verify=%d\n", a, b, (int)installed);
        return installed;
    }
    // rewrite cave body only (hook stays); only safe while no kernel runs
    bool rewrite(const std::vector<s120::I> &code) {
        auto w = s120::flat(code);
        size_t a = cuXtraMemcpyHtoD(cave, w.data(), w.size() * 8, 0);
        cuXtraInvalInstrCache(ctx);
        return a == w.size() * 8;
    }
    void restore() {
        if (!installed) return;
        cuXtraMemcpyHtoD(I.handler_va + 0x390, orig, 16, 0);
        cuXtraMemcpyHtoD(cave, cave_backup.data(), cave_bytes, 0);
        cuXtraInvalInstrCache(ctx);
        installed = false;
        printf("[patch] restored\n");
    }
};
