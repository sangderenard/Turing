// Text stream buffer ABI for compiled programs.
//
// Shell-class targets link ``turing_stream_buffer.c`` and gain a text sink
// for every publication the program makes. Targets without a sink elide
// publications at emission instead; publication is never load-bearing for
// numerics, so both dispositions run the same SSA.
#ifndef TURING_STREAM_BUFFER_H
#define TURING_STREAM_BUFFER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Append a payload to one stream. ``count`` < 0 means NUL-terminated text.
void turing_stream_publish(uint32_t stream_id, const void* value,
                           int64_t count, int final);

// Append one numeric value, rendered.
void turing_stream_publish_double(uint32_t stream_id, double value,
                                  int final);

// Read the accumulated text; the caller may ignore it entirely.
const char* turing_stream_text(uint32_t stream_id, uint64_t* out_length);

// Payloads dropped because the buffer could not grow.
uint64_t turing_stream_dropped(uint32_t stream_id);

void turing_stream_reset(uint32_t stream_id);

#ifdef __cplusplus
}
#endif

#endif  // TURING_STREAM_BUFFER_H
