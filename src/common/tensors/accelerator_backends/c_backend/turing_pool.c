// Persistent worker pool: the native ``pool`` lowering of a deployment
// frame.  See turing_pool.h for the ABI contract.
//
// The algorithm is deliberately identical to the Python reference pool
// (src/compiler/deployment_host_pool.py): one lock, one condition, a shared
// cursor claimed under the lock, a completion count for the barrier, and
// the deploying thread draining jobs alongside the workers.  Keeping the
// two implementations line-for-line parallel makes the Python pool the
// executable specification of this one -- a property worth more than the
// nanoseconds a lock-free deque would save at chunk granularity.
//
// Two lifetime rules the Python version gets from garbage collection must
// be explicit here, because the frame lives on the deploying thread's
// stack:
//
// 1. ``observers`` counts workers currently inside the frame; the deployer
//    retires the frame only when the work is complete AND no worker still
//    holds the pointer, so a worker can never dereference a dead stack.
// 2. ``g_generation`` increases once per installed frame; workers park on
//    the generation, never on the frame address, so a recycled stack slot
//    (same address, next deploy) cannot ABA a parked worker into sleeping
//    through a live frame.
//
// Windows uses SRWLOCK/CONDITION_VARIABLE, POSIX uses pthreads; both are
// the platform's native parking primitive, so parked workers cost nothing
// between frames.

#include "turing_pool.h"

#include <stddef.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
typedef SRWLOCK turing_pool_lock;
typedef CONDITION_VARIABLE turing_pool_cond;
typedef HANDLE turing_pool_thread;
#define TURING_POOL_LOCK_INIT SRWLOCK_INIT
#define TURING_POOL_COND_INIT CONDITION_VARIABLE_INIT
#if defined(_MSC_VER)
#define TURING_POOL_TLS __declspec(thread)
#else
#define TURING_POOL_TLS __thread
#endif
static void pool_lock(turing_pool_lock* lock) { AcquireSRWLockExclusive(lock); }
static void pool_unlock(turing_pool_lock* lock) { ReleaseSRWLockExclusive(lock); }
static void pool_wait(turing_pool_cond* cond, turing_pool_lock* lock) {
    SleepConditionVariableSRW(cond, lock, INFINITE, 0);
}
static void pool_notify_all(turing_pool_cond* cond) {
    WakeAllConditionVariable(cond);
}
#else
#include <pthread.h>
typedef pthread_mutex_t turing_pool_lock;
typedef pthread_cond_t turing_pool_cond;
typedef pthread_t turing_pool_thread;
#define TURING_POOL_LOCK_INIT PTHREAD_MUTEX_INITIALIZER
#define TURING_POOL_COND_INIT PTHREAD_COND_INITIALIZER
#define TURING_POOL_TLS __thread
static void pool_lock(turing_pool_lock* lock) { pthread_mutex_lock(lock); }
static void pool_unlock(turing_pool_lock* lock) { pthread_mutex_unlock(lock); }
static void pool_wait(turing_pool_cond* cond, turing_pool_lock* lock) {
    pthread_cond_wait(cond, lock);
}
static void pool_notify_all(turing_pool_cond* cond) {
    pthread_cond_broadcast(cond);
}
#endif

#define TURING_POOL_MAX_WORKERS 64

// Numerical determinism across threads: on x86 the MXCSR control register
// (flush-to-zero / denormals-are-zero, rounding mode) is PER THREAD, and a
// host process often runs with FTZ set on its main thread while fresh
// workers get the architectural default.  Denormal-range arithmetic then
// differs by which thread claimed a lane -- observed as run-to-run
// non-reproducibility at ~1e-309 magnitudes.  Every frame therefore
// carries the DEPLOYING thread's control word, and each drainer adopts it
// for the duration of the frame, so pooled execution is bitwise identical
// to the deploying thread running the same schedule serially.
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <xmmintrin.h>
#define TURING_POOL_HAS_FPENV 1
typedef unsigned int turing_fp_env;
static turing_fp_env turing_fp_env_get(void) { return _mm_getcsr(); }
static void turing_fp_env_set(turing_fp_env env) { _mm_setcsr(env); }
#else
#define TURING_POOL_HAS_FPENV 0
typedef int turing_fp_env;
static turing_fp_env turing_fp_env_get(void) { return 0; }
static void turing_fp_env_set(turing_fp_env env) { (void)env; }
#endif

struct turing_dispatch_task {
    turing_task_fn fn;
    void* context;
    turing_pool_thread thread;
    turing_pool_lock lock;
    turing_pool_cond completed;
    int done;
    turing_fp_env fp_env;
};
static TURING_POOL_TLS turing_dispatch_task* t_current_task = NULL;

#ifdef _WIN32
static DWORD WINAPI dispatch_task_main(LPVOID raw)
#else
static void* dispatch_task_main(void* raw)
#endif
{
    turing_dispatch_task* task = (turing_dispatch_task*)raw;
    t_current_task = task;
    turing_fp_env_set(task->fp_env);
    task->fn(task->context);
    pool_lock(&task->lock);
    task->done = 1;
    pool_notify_all(&task->completed);
    pool_unlock(&task->lock);
    t_current_task = NULL;
    return 0;
}

turing_dispatch_task* turing_task_start(turing_task_fn fn, void* context) {
    turing_dispatch_task* task;
    if (!fn) return NULL;
    task = (turing_dispatch_task*)calloc(1, sizeof(*task));
    if (!task) return NULL;
    task->fn = fn;
    task->context = context;
    task->fp_env = turing_fp_env_get();
#ifdef _WIN32
    InitializeSRWLock(&task->lock);
    InitializeConditionVariable(&task->completed);
    task->thread = CreateThread(NULL, 0, dispatch_task_main, task, 0, NULL);
    if (!task->thread) { free(task); return NULL; }
#else
    if (pthread_mutex_init(&task->lock, NULL)) { free(task); return NULL; }
    if (pthread_cond_init(&task->completed, NULL)) {
        pthread_mutex_destroy(&task->lock); free(task); return NULL;
    }
    if (pthread_create(&task->thread, NULL, dispatch_task_main, task)) {
        pthread_cond_destroy(&task->completed);
        pthread_mutex_destroy(&task->lock); free(task); return NULL;
    }
#endif
    return task;
}

int turing_task_wait(turing_dispatch_task* task, long timeout_ms) {
    int result = 1;
#ifdef _WIN32
    ULONGLONG deadline = timeout_ms < 0 ? 0 : GetTickCount64() + (ULONGLONG)timeout_ms;
#else
    struct timespec deadline;
    if (timeout_ms >= 0) {
        if (clock_gettime(CLOCK_REALTIME, &deadline)) return -1;
        deadline.tv_sec += timeout_ms / 1000;
        deadline.tv_nsec += (timeout_ms % 1000) * 1000000L;
        if (deadline.tv_nsec >= 1000000000L) {
            deadline.tv_sec++; deadline.tv_nsec -= 1000000000L;
        }
    }
#endif
    if (!task) return -1;
    if (task == t_current_task) return -2;
    pool_lock(&task->lock);
    while (!task->done) {
#ifdef _WIN32
        DWORD remaining = INFINITE;
        if (timeout_ms >= 0) {
            ULONGLONG now = GetTickCount64();
            if (now >= deadline) { result = 0; break; }
            remaining = (DWORD)(deadline - now);
        }
        if (!SleepConditionVariableSRW(&task->completed, &task->lock, remaining, 0)) {
            result = GetLastError() == ERROR_TIMEOUT ? 0 : -1;
            if (task->done) result = 1;
            break;
        }
#else
        int status = timeout_ms < 0
            ? pthread_cond_wait(&task->completed, &task->lock)
            : pthread_cond_timedwait(&task->completed, &task->lock, &deadline);
        if (status) {
            result = task->done ? 1 : status == ETIMEDOUT ? 0 : -1;
            break;
        }
#endif
    }
    pool_unlock(&task->lock);
    return result;
}

int turing_task_destroy(turing_dispatch_task* task) {
    int status = turing_task_wait(task, -1);
    if (status != 1) return status;
#ifdef _WIN32
    if (WaitForSingleObject(task->thread, INFINITE) != WAIT_OBJECT_0) return -1;
    if (!CloseHandle(task->thread)) return -1;
#else
    if (pthread_join(task->thread, NULL)) return -1;
    pthread_cond_destroy(&task->completed);
    pthread_mutex_destroy(&task->lock);
#endif
    free(task);
    return 0;
}

typedef struct turing_condition_waiter {
    struct turing_condition_waiter* next;
    turing_pool_cond changed;
    int notified;
} turing_condition_waiter;

struct turing_dispatch_condition {
    turing_pool_lock monitor;
    turing_pool_cond available;
    void* owner;
    unsigned long depth;
    unsigned long acquiring;
    turing_condition_waiter* waiters;
};
static TURING_POOL_TLS int t_condition_owner;

turing_dispatch_condition* turing_dispatch_condition_create(void) {
    turing_dispatch_condition* condition =
        (turing_dispatch_condition*)calloc(1, sizeof(*condition));
    if (!condition) return NULL;
#ifdef _WIN32
    InitializeSRWLock(&condition->monitor);
    InitializeConditionVariable(&condition->available);
#else
    if (pthread_mutex_init(&condition->monitor, NULL)) { free(condition); return NULL; }
    if (pthread_cond_init(&condition->available, NULL)) {
        pthread_mutex_destroy(&condition->monitor); free(condition); return NULL;
    }
#endif
    return condition;
}

int turing_dispatch_condition_acquire(turing_dispatch_condition* condition) {
    if (!condition) return -1;
    pool_lock(&condition->monitor);
    condition->acquiring++;
    while (condition->owner && condition->owner != &t_condition_owner)
        pool_wait(&condition->available, &condition->monitor);
    condition->acquiring--;
    condition->owner = &t_condition_owner;
    condition->depth++;
    pool_unlock(&condition->monitor);
    return 1;
}

int turing_dispatch_condition_release(turing_dispatch_condition* condition) {
    if (!condition) return -1;
    pool_lock(&condition->monitor);
    if (condition->owner != &t_condition_owner) {
        pool_unlock(&condition->monitor); return -1;
    }
    if (--condition->depth == 0) {
        condition->owner = NULL;
        pool_notify_all(&condition->available);
    }
    pool_unlock(&condition->monitor);
    return 0;
}

int turing_dispatch_condition_wait(turing_dispatch_condition* condition, long timeout_ms) {
    turing_condition_waiter waiter;
    turing_condition_waiter** link;
    unsigned long depth;
    int result = 1;
#ifdef _WIN32
    ULONGLONG deadline = timeout_ms < 0 ? 0 : GetTickCount64() + (ULONGLONG)timeout_ms;
    InitializeConditionVariable(&waiter.changed);
#else
    struct timespec deadline;
    if (timeout_ms >= 0) {
        if (clock_gettime(CLOCK_REALTIME, &deadline)) return -1;
        deadline.tv_sec += timeout_ms / 1000;
        deadline.tv_nsec += (timeout_ms % 1000) * 1000000L;
        if (deadline.tv_nsec >= 1000000000L) {
            deadline.tv_sec++; deadline.tv_nsec -= 1000000000L;
        }
    }
    if (pthread_cond_init(&waiter.changed, NULL)) return -1;
#endif
    if (!condition) {
#ifndef _WIN32
        pthread_cond_destroy(&waiter.changed);
#endif
        return -1;
    }
    pool_lock(&condition->monitor);
    if (condition->owner != &t_condition_owner) {
        pool_unlock(&condition->monitor);
#ifndef _WIN32
        pthread_cond_destroy(&waiter.changed);
#endif
        return -1;
    }
    waiter.next = NULL;
    waiter.notified = 0;
    link = &condition->waiters;
    while (*link) link = &(*link)->next;
    *link = &waiter;
    depth = condition->depth;
    condition->depth = 0;
    condition->owner = NULL;
    pool_notify_all(&condition->available);
    while (!waiter.notified) {
#ifdef _WIN32
        DWORD remaining = INFINITE;
        if (timeout_ms >= 0) {
            ULONGLONG now = GetTickCount64();
            if (now >= deadline) { result = 0; break; }
            remaining = (DWORD)(deadline - now);
        }
        if (!SleepConditionVariableSRW(&waiter.changed, &condition->monitor, remaining, 0)) {
            result = waiter.notified ? 1 : GetLastError() == ERROR_TIMEOUT ? 0 : -1;
            break;
        }
#else
        int status = timeout_ms < 0
            ? pthread_cond_wait(&waiter.changed, &condition->monitor)
            : pthread_cond_timedwait(&waiter.changed, &condition->monitor, &deadline);
        if (status) {
            result = waiter.notified ? 1 : status == ETIMEDOUT ? 0 : -1;
            break;
        }
#endif
    }
    link = &condition->waiters;
    while (*link != &waiter) link = &(*link)->next;
    *link = waiter.next;
    // Reacquisition is intentionally unbounded, even after a timed wait:
    // the authored caller resumes only while owning the same recursive lock.
    condition->acquiring++;
    while (condition->owner)
        pool_wait(&condition->available, &condition->monitor);
    condition->acquiring--;
    condition->owner = &t_condition_owner;
    condition->depth = depth;
    pool_unlock(&condition->monitor);
#ifndef _WIN32
    pthread_cond_destroy(&waiter.changed);
#endif
    return result;
}

int turing_dispatch_condition_notify(turing_dispatch_condition* condition, long count) {
    turing_condition_waiter* waiter;
    if (!condition) return -1;
    pool_lock(&condition->monitor);
    if (condition->owner != &t_condition_owner) {
        pool_unlock(&condition->monitor); return -1;
    }
    for (waiter = condition->waiters; waiter && count > 0; waiter = waiter->next) {
        if (waiter->notified) continue;
        waiter->notified = 1;
        count--;
        pool_notify_all(&waiter->changed);
    }
    pool_unlock(&condition->monitor);
    return 0;
}

int turing_dispatch_condition_notify_all(turing_dispatch_condition* condition) {
    return turing_dispatch_condition_notify(condition, LONG_MAX);
}

int turing_dispatch_condition_destroy(turing_dispatch_condition* condition) {
    if (!condition) return -1;
    pool_lock(&condition->monitor);
    if (condition->owner || condition->waiters || condition->acquiring) {
        pool_unlock(&condition->monitor); return -1;
    }
    pool_unlock(&condition->monitor);
#ifndef _WIN32
    pthread_cond_destroy(&condition->available);
    pthread_mutex_destroy(&condition->monitor);
#endif
    free(condition);
    return 0;
}

struct turing_dispatch_event {
    turing_dispatch_condition* condition;
    int flag;
};

turing_dispatch_event* turing_dispatch_event_create(void) {
    turing_dispatch_event* event = calloc(1, sizeof(*event));
    if (!event) return NULL;
    event->condition = turing_dispatch_condition_create();
    if (!event->condition) { free(event); return NULL; }
    return event;
}

int turing_dispatch_event_set(turing_dispatch_event* event) {
    int status, released;
    if (!event || turing_dispatch_condition_acquire(event->condition) != 1) return -1;
    event->flag = 1;
    status = turing_dispatch_condition_notify_all(event->condition);
    released = turing_dispatch_condition_release(event->condition);
    return status < 0 ? status : released;
}

int turing_dispatch_event_clear(turing_dispatch_event* event) {
    if (!event || turing_dispatch_condition_acquire(event->condition) != 1) return -1;
    event->flag = 0;
    return turing_dispatch_condition_release(event->condition);
}

int turing_dispatch_event_is_set(turing_dispatch_event* event) {
    int flag, released;
    if (!event || turing_dispatch_condition_acquire(event->condition) != 1) return -1;
    flag = event->flag;
    released = turing_dispatch_condition_release(event->condition);
    return released < 0 ? released : flag;
}

int turing_dispatch_event_wait(turing_dispatch_event* event, long timeout_ms) {
    int signaled, released;
    if (!event || turing_dispatch_condition_acquire(event->condition) != 1) return -1;
    signaled = event->flag;
    if (!signaled)
        signaled = turing_dispatch_condition_wait(event->condition, timeout_ms);
    // Keep the wait's notification result: another thread may clear the flag
    // before this waiter reacquires the lock. That does not undo its signal.
    released = turing_dispatch_condition_release(event->condition);
    return signaled < 0 ? signaled : released < 0 ? released : signaled;
}

int turing_dispatch_event_destroy(turing_dispatch_event* event) {
    int status;
    if (!event) return -1;
    status = turing_dispatch_condition_destroy(event->condition);
    if (status) return status;
    free(event);
    return 0;
}

typedef struct {
    turing_lane_fn fn;
    void* context;
    long chunks_per_lane;
    long total;
    long cursor;
    long completed;
    long observers;
    turing_fp_env fp_env;
} turing_pool_frame;

static turing_pool_lock g_lock = TURING_POOL_LOCK_INIT;
static turing_pool_cond g_cond = TURING_POOL_COND_INIT;
static turing_pool_thread g_threads[TURING_POOL_MAX_WORKERS];
static int g_worker_count = 0;
static int g_closing = 0;
static turing_pool_frame* g_frame = NULL;
static long g_generation = 0;
static TURING_POOL_TLS int t_in_lane = 0;

// Claim-and-run until the frame is exhausted.  Callable from a worker or
// from the deploying thread; with zero workers this loop on the deploying
// thread IS the serial fallback.
static void pool_drain(turing_pool_frame* frame) {
    turing_fp_env entry_env = turing_fp_env_get();
    turing_fp_env_set(frame->fp_env);
    for (;;) {
        long index;
        pool_lock(&g_lock);
        if (frame->cursor >= frame->total) {
            pool_unlock(&g_lock);
            turing_fp_env_set(entry_env);
            return;
        }
        index = frame->cursor++;
        pool_unlock(&g_lock);

        t_in_lane = 1;
        frame->fn(frame->context, index / frame->chunks_per_lane,
                  index % frame->chunks_per_lane, frame->chunks_per_lane);
        t_in_lane = 0;

        pool_lock(&g_lock);
        frame->completed++;
        if (frame->completed == frame->total) {
            pool_notify_all(&g_cond);
        }
        pool_unlock(&g_lock);
    }
}

#ifdef _WIN32
static DWORD WINAPI pool_worker_main(LPVOID unused)
#else
static void* pool_worker_main(void* unused)
#endif
{
    long seen_generation = 0;
    (void)unused;
    for (;;) {
        turing_pool_frame* frame;
        pool_lock(&g_lock);
        while (!g_closing
               && (g_frame == NULL || g_generation == seen_generation)) {
            pool_wait(&g_cond, &g_lock);
        }
        if (g_closing) {
            pool_unlock(&g_lock);
#ifdef _WIN32
            return 0;
#else
            return NULL;
#endif
        }
        seen_generation = g_generation;
        frame = g_frame;
        frame->observers++;
        pool_unlock(&g_lock);

        pool_drain(frame);

        pool_lock(&g_lock);
        frame->observers--;
        // The deployer may be waiting for the last observer to leave.
        pool_notify_all(&g_cond);
        pool_unlock(&g_lock);
    }
}

int turing_pool_start(int workers) {
    int target = workers;
    if (target < 0) {
        return -1;
    }
    if (target > TURING_POOL_MAX_WORKERS) {
        target = TURING_POOL_MAX_WORKERS;
    }
    pool_lock(&g_lock);
    g_closing = 0;
    while (g_worker_count < target) {
#ifdef _WIN32
        HANDLE thread = CreateThread(NULL, 0, pool_worker_main, NULL, 0, NULL);
        if (thread == NULL) {
            pool_unlock(&g_lock);
            return -1;
        }
        g_threads[g_worker_count] = thread;
#else
        pthread_t thread;
        if (pthread_create(&thread, NULL, pool_worker_main, NULL) != 0) {
            pool_unlock(&g_lock);
            return -1;
        }
        g_threads[g_worker_count] = thread;
#endif
        g_worker_count++;
    }
    {
        int count = g_worker_count;
        pool_unlock(&g_lock);
        return count;
    }
}

int turing_pool_workers(void) {
    int count;
    pool_lock(&g_lock);
    count = g_worker_count;
    pool_unlock(&g_lock);
    return count;
}

int turing_pool_deploy(turing_lane_fn fn, void* context, long lane_count,
                       long chunks_per_lane) {
    turing_pool_frame frame;
    if (fn == NULL || lane_count < 1 || chunks_per_lane < 1) {
        return -1;
    }
    if (t_in_lane) {
        // A nested deploy from inside a lane would wait on its own frame.
        return -2;
    }
    frame.fn = fn;
    frame.context = context;
    frame.chunks_per_lane = chunks_per_lane;
    frame.total = lane_count * chunks_per_lane;
    frame.cursor = 0;
    frame.completed = 0;
    frame.observers = 0;
    frame.fp_env = turing_fp_env_get();

    pool_lock(&g_lock);
    while (g_frame != NULL) {
        // One frame at a time; a second deployer queues here.
        pool_wait(&g_cond, &g_lock);
    }
    g_frame = &frame;
    g_generation++;
    pool_notify_all(&g_cond);
    pool_unlock(&g_lock);

    pool_drain(&frame);

    pool_lock(&g_lock);
    while (frame.completed < frame.total || frame.observers > 0) {
        pool_wait(&g_cond, &g_lock);
    }
    g_frame = NULL;
    pool_notify_all(&g_cond);
    pool_unlock(&g_lock);
    return 0;
}

typedef struct {
    turing_span_fn fn;
    void* context;
    long item_count;
    long chunk_size;
} turing_span_context;

static void turing_span_lane(void* raw, long lane, long chunk,
                             long chunks_per_lane) {
    turing_span_context* span = (turing_span_context*)raw;
    long start = lane * span->chunk_size;
    long stop = start + span->chunk_size;
    (void)chunk;
    (void)chunks_per_lane;
    if (stop > span->item_count) {
        stop = span->item_count;
    }
    if (start < stop) {
        span->fn(span->context, start, stop);
    }
}

int turing_pool_deploy_span(turing_span_fn fn, void* context, long item_count,
                            long chunk_size) {
    turing_span_context span;
    long claims;
    if (fn == NULL || item_count < 1 || chunk_size < 1) {
        return -1;
    }
    claims = (item_count + chunk_size - 1) / chunk_size;
    span.fn = fn;
    span.context = context;
    span.item_count = item_count;
    span.chunk_size = chunk_size;
    return turing_pool_deploy(turing_span_lane, &span, claims, 1);
}

// Dedicated lock for order-insensitive lane effects (see turing_pool.h).
// Never shared with g_lock: effect sections run inside lanes, and lanes
// must be free to take this lock while the scheduler lock cycles around
// claim bookkeeping on other threads.
static turing_pool_lock g_effect_lock = TURING_POOL_LOCK_INIT;

void turing_pool_effect_lock(void) { pool_lock(&g_effect_lock); }
void turing_pool_effect_unlock(void) { pool_unlock(&g_effect_lock); }

void turing_pool_stop(void) {
    int count;
    int index;
    pool_lock(&g_lock);
    g_closing = 1;
    count = g_worker_count;
    g_worker_count = 0;
    pool_notify_all(&g_cond);
    pool_unlock(&g_lock);
    for (index = 0; index < count; index++) {
#ifdef _WIN32
        WaitForSingleObject(g_threads[index], INFINITE);
        CloseHandle(g_threads[index]);
#else
        pthread_join(g_threads[index], NULL);
#endif
    }
}
