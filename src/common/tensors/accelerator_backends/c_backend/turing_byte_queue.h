// Copied byte-record deque. No queue-management thread or payload interpreter.
#ifndef TURING_BYTE_QUEUE_H
#define TURING_BYTE_QUEUE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct turing_byte_queue turing_byte_queue;
enum { TURING_QUEUE_LEFT = 0, TURING_QUEUE_RIGHT = 1 };
enum {
    TURING_QUEUE_OK = 1, TURING_QUEUE_WOULD_BLOCK = 0,
    TURING_QUEUE_INVALID = -1, TURING_QUEUE_CLOSED = -2,
    TURING_QUEUE_BUFFER_SMALL = -3, TURING_QUEUE_TOO_LARGE = -4
};

typedef struct {
    turing_byte_queue* queue;
    uint64_t worker; // logical context; never an OS-thread ownership claim
    int end;
} turing_byte_endpoint;

typedef struct {
    size_t length;
    uint64_t type_tag;
    uint64_t operation_contract;
    uint64_t producer_worker;
} turing_byte_record;

// Fixed slot count and maximum bytes per record; each record has its own length.
// Zero-length records are valid, including in a queue with slot_bytes == 0.
turing_byte_queue* turing_byte_queue_create(size_t slots, size_t slot_bytes);
// An endpoint is a caller-owned view, not an exclusive claim on an end. It may
// be passed across physical threads. The queue must outlive all endpoint uses.
int turing_byte_queue_anchor(turing_byte_queue*, int end, uint64_t worker,
                             turing_byte_endpoint*);
// timeout_ms: 0 = try, -1 = wait, positive = bounded wait; other negatives invalid.
// Publication copies payload and metadata atomically with respect to queue ops.
int turing_byte_queue_push(const turing_byte_endpoint*, const void*, size_t length,
                           uint64_t type_tag, uint64_t operation_contract, long timeout_ms);
// BUFFER_SMALL reports required length/metadata and consumes nothing. Empty or
// closed leaves outputs untouched. Retrying after BUFFER_SMALL is not a lease:
// another consumer can remove the record. payload may be NULL for an empty record.
int turing_byte_queue_pop(const turing_byte_endpoint*, void*, size_t capacity,
                          turing_byte_record*, long timeout_ms);
// Close refuses future pushes, wakes waiters, and permits draining queued records.
int turing_byte_queue_close(turing_byte_queue*);
// Snapshot only; not a reservation guaranteeing a later operation's success.
int turing_byte_queue_size(turing_byte_queue*, size_t*);
// Readiness diagnostics for scheduling/testing; counts native pending waits.
// This snapshot does not grant ownership of a waiter or reserve any storage.
int turing_byte_queue_waiters(turing_byte_queue*, size_t* readers, size_t* writers);
// Requires exclusive lifetime ownership: no active operations or future endpoint
// uses. Closing is not destruction; callers join all participating tasks first.
int turing_byte_queue_destroy(turing_byte_queue*);
#ifdef __cplusplus
}
#endif
#endif
