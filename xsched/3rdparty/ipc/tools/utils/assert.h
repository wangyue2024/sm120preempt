#pragma once

#include "log.h"
#include "common.h"

#define XASSERT(expr, format, ...) \
    do { \
        bool val = static_cast<bool>(expr); \
        if (UNLIKELY(!val)) { \
            XERRO_NOEXIT("Assertion failed: " format  __VA_OPT__(,) __VA_ARGS__); \
            XERRO("Error code: " FMT_ERRNO ", Error message: %s", GetErrNo(), GetErrStr().c_str()); \
        } \
    } while (0);

#define XASSERT_RET(expr, ret, format, ...) \
    do { \
        bool val = static_cast<bool>(expr); \
        if (UNLIKELY(!val)) { \
            XWARN("Assertion failed: " format  __VA_OPT__(,) __VA_ARGS__); \
            XWARN("Error code: " FMT_ERRNO ", Error message: %s", GetErrNo(), GetErrStr().c_str());\
            return ret; \
        } \
    } while (0);

#define XASSERT_NOEXIT(expr, format, ...) \
    do { \
        bool val = static_cast<bool>(expr); \
        if (UNLIKELY(!val)) { \
            XWARN("Assertion failed: " format  __VA_OPT__(,) __VA_ARGS__); \
            XWARN("Error code: " FMT_ERRNO ", Error message: %s", GetErrNo(), GetErrStr().c_str());\
        } \
    } while (0);
