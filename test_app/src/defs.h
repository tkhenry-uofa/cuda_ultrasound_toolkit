#ifndef DEFS_H
#define DEFS_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <iostream>
#include <stdexcept>
#include <string>

#include "parameter_defs.h"

typedef void* Handle;

typedef unsigned int uint;

typedef char      c8;
typedef uint8_t   u8;
typedef int16_t   i16;
typedef uint16_t  u16;
typedef int32_t   i32;
typedef uint32_t  u32;
typedef int64_t   i64;
typedef uint64_t  u64;
typedef uint32_t  b32;
typedef float     f32;
typedef double    f64;
typedef ptrdiff_t size;
typedef ptrdiff_t iptr;

#ifdef _DEBUG
#include <assert.h>
#define ASSERT(x) assert(x);
#else
#define ASSERT(x)
#endif // _DEBUG

#define INPUT_MAX_BUFFER 1000000000 // 1 Gb

#define IS_HANDLE_INVALID(h) ((h) == INVALID_HANDLE_VALUE || (h) == NULL)

inline const char* format_windows_error_message(DWORD code) {
    static char message[512];
    DWORD size = FormatMessageA(
        FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        code,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        message,
        sizeof(message),
        NULL
    );

    if (size == 0) {
        snprintf(message, sizeof(message), "Unknown error code: %lu", (unsigned long)code);
    }
    else {
        // Remove trailing newline if present
        size_t len = strlen(message);
        if (len > 0 && message[len - 1] == '\n') {
            message[len - 1] = '\0';
        }
    }

    return message;
}


#define WINDOWS_ERROR_MESSAGE(MSG, CODE)                                                        \
do                                                                                              \
{                                                                                               \
    std::cerr << MSG << ", Error code: " << CODE << " (" << format_windows_error_message(CODE) << ")" << std::endl; \
} while (0);                                                                                    \


#endif // !DEFS_H