#include "turing_validation_runtime.h"
#include <stddef.h>

#if defined(_WIN32)
#define TURING_VALIDATION_EXPORT __declspec(dllexport)
#else
#define TURING_VALIDATION_EXPORT __attribute__((visibility("default")))
#endif

static _Thread_local turing_error_packet active_error = {0};
static _Thread_local turing_error_packet reported_error = {0};
static _Thread_local uint64_t current_worker = 0;
static _Thread_local const char *current_worker_function = NULL;

TURING_VALIDATION_EXPORT void turing_validation_error_set_worker(uint64_t token, const char *function) {
    current_worker = token;
    current_worker_function = function;
}

TURING_VALIDATION_EXPORT void turing_validation_error_named(uint32_t code, const char *type, const char *message) {
    if (!active_error.code) {
        active_error.code = code ? code : UINT32_MAX;
        active_error.worker_token = current_worker;
        active_error.worker_function = current_worker_function;
        active_error.exception_type = type;
        active_error.message = message;
    }
}

TURING_VALIDATION_EXPORT void turing_validation_error(uint32_t code) {
    const char *type = NULL;
    /* These are the existing native_faults.py ABI codes. */
    switch (code) {
        case 71: type = "builtins.IndexError"; break;
        case 84: type = "builtins.KeyError"; break;
        case 86: case 95: case 96: type = "builtins.OverflowError"; break;
        case 87: case 90: case 94: type = "builtins.ValueError"; break;
        case 88: case 92: type = "builtins.ZeroDivisionError"; break;
        case 97: type = "builtins.RuntimeError"; break;
    }
    turing_validation_error_named(code, type, NULL);
}

TURING_VALIDATION_EXPORT void turing_validation_error_reset(void) {
    active_error = (turing_error_packet){0};
    reported_error = (turing_error_packet){0};
}

TURING_VALIDATION_EXPORT void turing_validation_error_report(const turing_error_packet *packet) {
    if (packet && packet->code && !reported_error.code) reported_error = *packet;
}

TURING_VALIDATION_EXPORT uint32_t turing_validation_error_pending(void) {
    return active_error.code;
}

TURING_VALIDATION_EXPORT int turing_validation_error_take_packet(turing_error_packet *packet) {
    if (!packet) return 0;
    if (active_error.code) {
        *packet = active_error;
        active_error = (turing_error_packet){0};
    } else {
        *packet = reported_error;
        reported_error = (turing_error_packet){0};
    }
    return packet->code != 0;
}

TURING_VALIDATION_EXPORT uint32_t turing_validation_error_take(void) {
    turing_error_packet packet;
    return turing_validation_error_take_packet(&packet) ? packet.code : 0;
}

TURING_VALIDATION_EXPORT void turing_error_packet_init(turing_error_packet *packet, uint32_t code, const char *type, const char *message) {
    *packet = (turing_error_packet){code ? code : UINT32_MAX, current_worker, current_worker_function, type, message};
}

TURING_VALIDATION_EXPORT int turing_validation_error_take_active(turing_error_packet *packet) {
    if (!packet) return 0;
    *packet = active_error;
    active_error = (turing_error_packet){0};
    return packet->code != 0;
}

TURING_VALIDATION_EXPORT void turing_validation_error_raise_packet(const turing_error_packet *packet) {
    if (packet && packet->code) active_error = *packet;
}

#include <string.h>
TURING_VALIDATION_EXPORT int turing_error_packet_matches(const turing_error_packet *packet, const char *type) {
    const char *actual;
    if (!packet || !packet->code || !type) return 0;
    actual = packet->exception_type;
    /* Unknown numeric faults have no declared Python exception identity. */
    if (!actual) return 0;
    if (!strcmp(type, "builtins.BaseException")) return 1;
    if (!strcmp(type, "builtins.Exception"))
        return strcmp(actual, "builtins.BaseException") != 0;
    if (!strcmp(actual, type)) return 1;
    if (!strcmp(type, "builtins.ArithmeticError"))
        return !strcmp(actual, "builtins.ZeroDivisionError") || !strcmp(actual, "builtins.OverflowError") || !strcmp(actual, "builtins.FloatingPointError");
    if (!strcmp(type, "builtins.RuntimeError"))
        return !strcmp(actual, "threading.BrokenBarrierError");
    if (!strcmp(type, "builtins.LookupError"))
        return !strcmp(actual, "builtins.IndexError") || !strcmp(actual, "builtins.KeyError");
    return 0;
}
