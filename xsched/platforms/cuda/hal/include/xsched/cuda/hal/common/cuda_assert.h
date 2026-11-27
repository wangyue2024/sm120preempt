#pragma once

#include <atomic>
#include <cstdlib>

#include "xsched/utils/log.h"
#include "xsched/utils/common.h"
#include "xsched/cuda/hal/common/driver.h"

// DEF-42: CUDA_ERROR_DEINITIALIZED only happens while the process exits (driver already torn down);
// exiting again from whichever thread hit it deadlocks against the exiting main thread. Ignore it.
#define CUDA_ASSERT(cmd) \
    do { \
        CUresult result = cmd; \
        if (UNLIKELY(result == CUDA_ERROR_DEINITIALIZED)) { \
            static std::atomic<bool> warned_deinit__{false}; \
            if (!warned_deinit__.exchange(true)) XWARN("cuda driver deinitialized (process exiting): ignoring"); \
        } else if (UNLIKELY(result != CUDA_SUCCESS)) { \
            const char *str = nullptr; \
            if (xsched::cuda::Driver::GetErrorString(result, &str) != CUDA_SUCCESS || str == nullptr) \
                str = "(no error string: driver deinitialized?)"; \
            XERRO("cuda error %d: %s", result, str); \
        } \
    } while (0);

#define CUPTI_ASSERT(cmd) \
    do { \
        CUptiResult result = cmd; \
        if (UNLIKELY(result != CUPTI_SUCCESS)) { \
            const char *str = nullptr; \
            if (xsched::cuda::PTI::GetResultString(result, &str) != CUPTI_SUCCESS || str == nullptr) \
                str = "(no error string)"; \
            XERRO("cupti error %d: %s", result, str); \
        } \
    } while (0);
