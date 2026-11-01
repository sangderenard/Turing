#include "turing_thread_table.h"
#include <stdlib.h>
#include <stdatomic.h>
static _Atomic uint64_t next_worker = 1;

/* The same application/activation scope owns tasks and synchronization rows.
 * Join captured workers before disposing any object they can still reach. */
struct turing_scope_resource {
    void *object;
    int (*dispose)(void*);
    struct turing_scope_resource *next;
};
static int dispose_mutex(void *p) {
    turing_dispatch_condition *mutex = turing_sync_wait_queue_mutex(p);
    if (mutex == p) {
        uint64_t worker = turing_sync_worker_token();
        int64_t token = turing_sync_mutex_token(mutex, worker);
        if (token) turing_sync_mutex_suspend(mutex, worker, token);
    }
    return turing_dispatch_condition_destroy(p);
}
static int dispose_latch(void *p) { return turing_sync_latch_destroy(p) == 1 ? 0 : -1; }
static int own_resource(turing_thread_scope *scope, void *object, int (*dispose)(void*)) {
    turing_scope_resource *row = malloc(sizeof(*row));
    if (!row) { dispose(object); return 0; }
    row->object = object; row->dispose = dispose;
    row->next = scope->resources; scope->resources = row; return 1;
}
turing_dispatch_condition *turing_source_mutex_create(turing_thread_scope *scope, int reentrant) {
    turing_dispatch_condition *mutex = turing_sync_mutex_create(reentrant);
    return mutex && own_resource(scope, mutex, dispose_mutex) ? mutex : NULL;
}
turing_dispatch_condition *turing_source_wait_queue_create(turing_thread_scope *scope, turing_dispatch_condition *mutex) {
    turing_dispatch_condition *queue = turing_sync_wait_queue_create(mutex);
    return queue && own_resource(scope, queue, dispose_mutex) ? queue : NULL;
}
turing_sync_latch *turing_source_latch_create(turing_thread_scope *scope, uint64_t participants, uint64_t generation) {
    turing_sync_latch *latch = turing_sync_latch_create(participants, generation);
    return latch && own_resource(scope, latch, dispose_latch) ? latch : NULL;
}
turing_sync_latch *turing_source_barrier_create(turing_thread_scope *scope, uint64_t participants, double timeout, int finite) {
    if (!participants) return NULL;
    turing_sync_latch *latch = turing_sync_latch_create(participants, 0);
    if (!latch) return NULL;
    if (turing_sync_barrier_configure(latch, timeout, finite) != 1) { turing_sync_latch_destroy(latch); return NULL; }
    return own_resource(scope, latch, dispose_latch) ? latch : NULL;
}
int turing_source_latch_destroy(turing_thread_scope *scope, turing_sync_latch *latch) {
    turing_scope_resource **link = &scope->resources;
    while (*link && (*link)->object != latch) link = &(*link)->next;
    if (!*link || (*link)->dispose != dispose_latch) return -1;
    turing_scope_resource *row = *link;
    if (row->dispose(row->object)) return -1;
    *link = row->next; free(row); return 1;
}

struct turing_source_thread {
    turing_task_fn entry;
    void *context;
    turing_error_packet *fault;
    int reported;
    turing_dispatch_task *task;
    struct turing_source_thread *next;
};

turing_source_thread *turing_source_thread_create(turing_thread_scope *scope,
    turing_task_fn entry, void *owned_context, turing_error_packet *fault) {
    turing_source_thread *thread;
    if (!scope || !entry || !owned_context || !fault) return NULL;
    thread = (turing_source_thread *)calloc(1, sizeof(*thread));
    if (!thread) return NULL;
    thread->entry = entry;
    thread->context = owned_context;
    thread->fault = fault;
    uint64_t token = atomic_load(&next_worker);
    do {
        if (!token || token > (UINT64_MAX >> 1)) { free(thread); return NULL; }
    } while (!atomic_compare_exchange_weak(&next_worker, &token, token + 1));
    fault->worker_token = token;
    thread->next = scope->head;
    scope->head = thread;
    return thread;
}

int turing_source_thread_start(turing_source_thread *thread) {
    if (!thread || thread->task) return -1;
    thread->task = turing_task_start(thread->entry, thread->context);
    return thread->task ? 0 : -2;
}

int turing_source_thread_join(turing_source_thread *thread) {
    int status;
    if (!thread || !thread->task) return -1;
    status = turing_task_wait(thread->task, -1);
    if (status != 1) return -1;
    if (thread->fault->code && !thread->reported) {
        turing_validation_error_report(thread->fault);
        thread->reported = 1;
    }
    return 0;
}

const turing_error_packet *turing_source_thread_error(turing_source_thread *thread) {
    if (!thread || !thread->task || turing_task_is_alive(thread->task)) return NULL;
    return thread->fault->code ? thread->fault : NULL;
}

int turing_source_thread_is_alive(turing_source_thread *thread) {
    return thread && thread->task ? turing_task_is_alive(thread->task) : 0;
}

void turing_thread_scope_clear(turing_thread_scope *scope) {
    turing_source_thread *thread, *next;
    if (!scope) return;
    for (thread = scope->head; thread; thread = next) {
        next = thread->next;
        /* Normal admitted paths already joined. Fault unwinding must also
         * finish tasks before their captured activation storage is freed. */
        if (thread->task && turing_task_destroy(thread->task) != 0) abort();
        free(thread->context);
        free(thread);
    }
    scope->head = NULL;
    while (scope->resources) {
        turing_scope_resource *row = scope->resources;
        scope->resources = row->next;
        if (row->dispose(row->object)) {
            /* A caller abandoned an acquired mutex. Report the lifecycle
             * failure rather than free live synchronization state. */
            turing_validation_error_named(97, "builtins.RuntimeError", "synchronization object is still in use");
        }
        free(row);
    }
}
