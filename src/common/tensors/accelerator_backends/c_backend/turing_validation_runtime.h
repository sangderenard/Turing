#ifndef TURING_VALIDATION_RUNTIME_H
#define TURING_VALIDATION_RUNTIME_H
#include <stdint.h>
#if defined(_WIN32)
#define TURING_VALIDATION_EXPORT __declspec(dllexport)
#else
#define TURING_VALIDATION_EXPORT __attribute__((visibility("default")))
#endif
/* First runtime failure, with identity/payload fields when lowering supplies
 * them. NULL means unavailable, never an invented exception or message. */
typedef struct {
    uint32_t code;
    uint64_t worker_token;
    const char *worker_function;
    const char *exception_type;
    const char *message;
    uint8_t suppress_context;
} turing_error_packet;
TURING_VALIDATION_EXPORT void turing_validation_error(uint32_t code);
TURING_VALIDATION_EXPORT void turing_validation_error_named(uint32_t code, const char *type, const char *message);
TURING_VALIDATION_EXPORT void turing_validation_error_reset(void);
TURING_VALIDATION_EXPORT uint32_t turing_validation_error_take(void);
TURING_VALIDATION_EXPORT int turing_validation_error_take_packet(turing_error_packet *packet);
TURING_VALIDATION_EXPORT void turing_validation_error_set_worker(uint64_t token, const char *function);
/* Shell reporting is separate from a synchronous error in the current path. */
TURING_VALIDATION_EXPORT void turing_validation_error_report(const turing_error_packet *packet);
TURING_VALIDATION_EXPORT uint32_t turing_validation_error_pending(void);
TURING_VALIDATION_EXPORT void turing_error_packet_init(turing_error_packet *packet, uint32_t code, const char *type, const char *message, uint8_t suppress_context);
TURING_VALIDATION_EXPORT int turing_validation_error_take_active(turing_error_packet *packet);
TURING_VALIDATION_EXPORT void turing_validation_error_raise_packet(const turing_error_packet *packet);
TURING_VALIDATION_EXPORT int turing_error_packet_matches(const turing_error_packet *packet, const char *type);
#endif
