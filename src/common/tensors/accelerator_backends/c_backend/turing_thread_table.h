#ifndef TURING_THREAD_TABLE_H
#define TURING_THREAD_TABLE_H
#include <stdint.h>
#include "turing_pool.h"
#include "turing_validation_runtime.h"
/* Authored thread handles: construction does not start the native task. */
typedef struct turing_source_thread turing_source_thread;
typedef struct turing_scope_resource turing_scope_resource;
typedef struct { turing_source_thread *head; turing_scope_resource *resources; } turing_thread_scope;
turing_dispatch_condition *turing_source_mutex_create(turing_thread_scope*, int reentrant);
turing_dispatch_condition *turing_source_wait_queue_create(turing_thread_scope*, turing_dispatch_condition*);
turing_sync_latch *turing_source_latch_create(turing_thread_scope*, uint64_t participants, uint64_t generation);
turing_sync_latch *turing_source_barrier_create(turing_thread_scope*, uint64_t participants, double timeout, int finite);
int turing_source_latch_destroy(turing_thread_scope*, turing_sync_latch*);
turing_source_thread *turing_source_thread_create(turing_thread_scope *scope,
    turing_task_fn entry, void *owned_context, turing_error_packet *fault);
int turing_source_thread_start(turing_source_thread *thread);
/* 0 completion, negative lifecycle/OS error. Worker errors remain packets. */
int turing_source_thread_join(turing_source_thread *thread);
const turing_error_packet *turing_source_thread_error(turing_source_thread *thread);
int turing_source_thread_is_alive(turing_source_thread *thread);
void turing_thread_scope_clear(turing_thread_scope *scope);
#endif
