#include <stdint.h>

#if defined(_WIN32)
#define TURING_VALIDATION_EXPORT __declspec(dllexport)
#else
#define TURING_VALIDATION_EXPORT __attribute__((visibility("default")))
#endif

static _Thread_local uint32_t turing_validation_code = 0;

/* Windows parity receipt (2026-10-24): this runtime is compiled as its own
 * translation unit, so it cannot inherit the emitter's TURING_EXPORT macro.
 * MinGW linked the DLL but omitted these three host-facing symbols; ctypes
 * then failed before the authored program ran.  Export the runtime ABI at
 * its writer, for every platform on which the DLL/shared object is loaded. */
TURING_VALIDATION_EXPORT void turing_validation_error(uint32_t error_code) {
    if (turing_validation_code == 0) {
        turing_validation_code = error_code == 0 ? UINT32_MAX : error_code;
    }
}

TURING_VALIDATION_EXPORT void turing_validation_error_reset(void) {
    turing_validation_code = 0;
}

TURING_VALIDATION_EXPORT uint32_t turing_validation_error_take(void) {
    uint32_t result = turing_validation_code;
    turing_validation_code = 0;
    return result;
}
