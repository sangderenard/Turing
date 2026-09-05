// Universal text stream buffer for compiled programs.
//
// A compiled program publishes text through ``turing_stream_publish`` -- the
// operation ingestion produces for ``print`` and any other publication. This
// file is the sink shell-class targets link: a growable byte buffer per
// stream id, appended to as the program runs and handed back at the end as
// one descriptor the caller may read or ignore entirely.
//
// Nothing here is required for numerics. A target without a text sink (a
// bare native artifact, a shader) does not link this file, and its emitter
// elides publications instead -- publication is never load-bearing.

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef TURING_STREAM_COUNT
#define TURING_STREAM_COUNT 8
#endif

// One growable buffer per stream. Capacity doubles on demand; a failed
// growth drops the payload and records the overflow rather than aborting a
// computation for the sake of text.
typedef struct {
    char* bytes;
    size_t length;
    size_t capacity;
    uint64_t dropped;
} TuringStream;

static TuringStream turing_streams[TURING_STREAM_COUNT];

static int turing_stream_reserve(TuringStream* stream, size_t additional) {
    size_t needed = stream->length + additional + 1;
    if (needed <= stream->capacity) {
        return 1;
    }
    size_t capacity = stream->capacity ? stream->capacity : 1024;
    while (capacity < needed) {
        capacity *= 2;
    }
    char* grown = (char*)realloc(stream->bytes, capacity);
    if (grown == NULL) {
        stream->dropped += 1;
        return 0;
    }
    stream->bytes = grown;
    stream->capacity = capacity;
    return 1;
}

// The publish entry the emitters already call. ``value`` is the payload
// address, ``count`` its element count (-1 meaning "NUL-terminated text"),
// and ``final`` marks the last publication on that stream.
void turing_stream_publish(uint32_t stream_id, const void* value,
                           int64_t count, int final) {
    (void)final;
    if (stream_id >= TURING_STREAM_COUNT || value == NULL) {
        return;
    }
    TuringStream* stream = &turing_streams[stream_id];
    size_t length = (count < 0)
        ? strlen((const char*)value)
        : (size_t)count;
    if (!turing_stream_reserve(stream, length + 1)) {
        return;
    }
    memcpy(stream->bytes + stream->length, value, length);
    stream->length += length;
    stream->bytes[stream->length++] = '\n';
    stream->bytes[stream->length] = '\0';
}

// Publish one double, which is what a numeric ``print`` carries.
void turing_stream_publish_double(uint32_t stream_id, double value,
                                  int final) {
    char rendered[32];
    int written = snprintf(rendered, sizeof(rendered), "%.17g", value);
    if (written > 0) {
        turing_stream_publish(stream_id, rendered, written, final);
    }
}

// The drain: hand back the accumulated text and its length. The caller may
// ignore both -- an ignored buffer costs one allocation and nothing else.
const char* turing_stream_text(uint32_t stream_id, uint64_t* out_length) {
    if (stream_id >= TURING_STREAM_COUNT) {
        if (out_length) *out_length = 0;
        return NULL;
    }
    TuringStream* stream = &turing_streams[stream_id];
    if (out_length) {
        *out_length = (uint64_t)stream->length;
    }
    return stream->bytes ? stream->bytes : "";
}

uint64_t turing_stream_dropped(uint32_t stream_id) {
    if (stream_id >= TURING_STREAM_COUNT) {
        return 0;
    }
    return turing_streams[stream_id].dropped;
}

void turing_stream_reset(uint32_t stream_id) {
    if (stream_id >= TURING_STREAM_COUNT) {
        return;
    }
    TuringStream* stream = &turing_streams[stream_id];
    stream->length = 0;
    stream->dropped = 0;
    if (stream->bytes) {
        stream->bytes[0] = '\0';
    }
}
