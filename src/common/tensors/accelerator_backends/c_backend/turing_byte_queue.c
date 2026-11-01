#ifndef _WIN32
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#endif
#include "turing_byte_queue.h"
#include "turing_pool.h"
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

struct turing_byte_queue {
    turing_dispatch_condition* changed;
    unsigned char* bytes;
    turing_byte_record* records;
    size_t slots, slot_bytes, head, count, readers, writers;
    int closed;
};

static int byte_queue_now(uint64_t* result) {
#ifdef _WIN32
    *result = (uint64_t)GetTickCount64();
#else
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now)) return -1;
    *result = (uint64_t)now.tv_sec * 1000 + (uint64_t)now.tv_nsec / 1000000;
#endif
    return 0;
}

static int byte_queue_deadline(long timeout_ms, uint64_t* deadline) {
    uint64_t now;
    if (timeout_ms < -1) return -1;
    *deadline = 0;
    if (timeout_ms <= 0) return 0;
    if (byte_queue_now(&now) || (uint64_t)timeout_ms > UINT64_MAX - now) return -1;
    *deadline = now + (uint64_t)timeout_ms;
    return 0;
}

// Called while owning changed. Condition.wait atomically registers and releases
// ownership, then reacquires it; there is no predicate/register lost-wakeup gap.
static int byte_queue_wait(turing_byte_queue* q, long timeout_ms, uint64_t deadline) {
    uint64_t now;
    long remaining = timeout_ms;
    if (!timeout_ms) return TURING_QUEUE_WOULD_BLOCK;
    if (timeout_ms > 0) {
        if (byte_queue_now(&now)) return TURING_QUEUE_INVALID;
        if (now >= deadline) return TURING_QUEUE_WOULD_BLOCK;
        remaining = (long)(deadline - now); // bounded by original positive long
    }
    return turing_dispatch_condition_wait(q->changed, remaining);
}

static int byte_queue_unlock(turing_byte_queue* q, int status) {
    return turing_dispatch_condition_release(q->changed) ? TURING_QUEUE_INVALID : status;
}

static int byte_queue_endpoint_valid(const turing_byte_endpoint* endpoint) {
    return endpoint && endpoint->queue &&
        (endpoint->end == TURING_QUEUE_LEFT || endpoint->end == TURING_QUEUE_RIGHT);
}

// Overflow-free wrapped slot arithmetic: head + offset can exceed SIZE_MAX.
static size_t byte_queue_slot(const turing_byte_queue* q, size_t offset) {
    size_t until_wrap = q->slots - q->head;
    return offset >= until_wrap ? offset - until_wrap : q->head + offset;
}

turing_byte_queue* turing_byte_queue_create(size_t slots, size_t slot_bytes) {
    turing_byte_queue* q;
    if (!slots || slots > SIZE_MAX / sizeof(turing_byte_record) ||
        (slot_bytes && slots > SIZE_MAX / slot_bytes)) return NULL;
    q = calloc(1, sizeof(*q));
    if (!q) return NULL;
    q->slots = slots;
    q->slot_bytes = slot_bytes;
    q->records = calloc(slots, sizeof(*q->records));
    q->bytes = malloc(slot_bytes ? slots * slot_bytes : 1);
    q->changed = turing_dispatch_condition_create();
    if (!q->records || !q->bytes || !q->changed) {
        if (q->changed) turing_dispatch_condition_destroy(q->changed);
        free(q->records); free(q->bytes); free(q); return NULL;
    }
    return q;
}

int turing_byte_queue_anchor(turing_byte_queue* q, int end, uint64_t worker,
                             turing_byte_endpoint* endpoint) {
    if (!q || !endpoint || (end != TURING_QUEUE_LEFT && end != TURING_QUEUE_RIGHT))
        return TURING_QUEUE_INVALID;
    endpoint->queue = q; endpoint->end = end; endpoint->worker = worker;
    return TURING_QUEUE_OK;
}

int turing_byte_queue_push(const turing_byte_endpoint* endpoint, const void* payload,
                           size_t length, uint64_t type_tag, uint64_t contract, long timeout_ms) {
    turing_byte_queue* q;
    uint64_t deadline;
    size_t slot;
    int status;
    if (!byte_queue_endpoint_valid(endpoint) || (length && !payload) ||
        byte_queue_deadline(timeout_ms, &deadline)) return TURING_QUEUE_INVALID;
    q = endpoint->queue;
    if (length > q->slot_bytes) return TURING_QUEUE_TOO_LARGE;
    if (turing_dispatch_condition_acquire(q->changed) != 1) return TURING_QUEUE_INVALID;
    while (q->count == q->slots && !q->closed) {
        q->writers++;
        status = byte_queue_wait(q, timeout_ms, deadline);
        q->writers--;
        if (status < 0) return byte_queue_unlock(q, TURING_QUEUE_INVALID);
        // Recheck under the lock even on timeout: readiness can race expiry.
        if (!status && q->count == q->slots && !q->closed)
            return byte_queue_unlock(q, TURING_QUEUE_WOULD_BLOCK);
    }
    if (q->closed) return byte_queue_unlock(q, TURING_QUEUE_CLOSED);
    if (endpoint->end == TURING_QUEUE_LEFT) {
        q->head = q->head ? q->head - 1 : q->slots - 1;
        slot = q->head;
    } else slot = byte_queue_slot(q, q->count);
    if (length) memcpy(q->bytes + slot * q->slot_bytes, payload, length);
    q->records[slot] = (turing_byte_record){length, type_tag, contract, endpoint->worker};
    q->count++;
    status = turing_dispatch_condition_notify_all(q->changed);
    return byte_queue_unlock(q, status ? TURING_QUEUE_INVALID : TURING_QUEUE_OK);
}

int turing_byte_queue_pop(const turing_byte_endpoint* endpoint, void* payload,
                          size_t capacity, turing_byte_record* record, long timeout_ms) {
    turing_byte_queue* q;
    uint64_t deadline;
    size_t slot;
    int status;
    if (!byte_queue_endpoint_valid(endpoint) || !record || (capacity && !payload) ||
        byte_queue_deadline(timeout_ms, &deadline)) return TURING_QUEUE_INVALID;
    q = endpoint->queue;
    if (turing_dispatch_condition_acquire(q->changed) != 1) return TURING_QUEUE_INVALID;
    while (!q->count && !q->closed) {
        q->readers++;
        status = byte_queue_wait(q, timeout_ms, deadline);
        q->readers--;
        if (status < 0) return byte_queue_unlock(q, TURING_QUEUE_INVALID);
        if (!status && !q->count && !q->closed)
            return byte_queue_unlock(q, TURING_QUEUE_WOULD_BLOCK);
    }
    if (!q->count) return byte_queue_unlock(q, TURING_QUEUE_CLOSED);
    slot = endpoint->end == TURING_QUEUE_LEFT ? q->head : byte_queue_slot(q, q->count - 1);
    *record = q->records[slot];
    if (record->length > capacity) return byte_queue_unlock(q, TURING_QUEUE_BUFFER_SMALL);
    if (record->length) memcpy(payload, q->bytes + slot * q->slot_bytes, record->length);
    if (endpoint->end == TURING_QUEUE_LEFT) q->head = byte_queue_slot(q, 1);
    q->count--;
    status = turing_dispatch_condition_notify_all(q->changed);
    return byte_queue_unlock(q, status ? TURING_QUEUE_INVALID : TURING_QUEUE_OK);
}

int turing_byte_queue_close(turing_byte_queue* q) {
    int status;
    if (!q || turing_dispatch_condition_acquire(q->changed) != 1) return TURING_QUEUE_INVALID;
    q->closed = 1;
    status = turing_dispatch_condition_notify_all(q->changed);
    return byte_queue_unlock(q, status ? TURING_QUEUE_INVALID : TURING_QUEUE_OK);
}

int turing_byte_queue_size(turing_byte_queue* q, size_t* count) {
    if (!q || !count || turing_dispatch_condition_acquire(q->changed) != 1)
        return TURING_QUEUE_INVALID;
    *count = q->count;
    return byte_queue_unlock(q, TURING_QUEUE_OK);
}

int turing_byte_queue_destroy(turing_byte_queue* q) {
    if (!q || turing_dispatch_condition_destroy(q->changed)) return TURING_QUEUE_INVALID;
    free(q->records); free(q->bytes); free(q);
    return TURING_QUEUE_OK;
}

int turing_byte_queue_waiters(turing_byte_queue* q, size_t* readers, size_t* writers) {
    if (!q || !readers || !writers || turing_dispatch_condition_acquire(q->changed) != 1)
        return TURING_QUEUE_INVALID;
    *readers = q->readers; *writers = q->writers;
    return byte_queue_unlock(q, TURING_QUEUE_OK);
}
