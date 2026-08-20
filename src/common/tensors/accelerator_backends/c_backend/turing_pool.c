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

#ifdef _WIN32
#include <windows.h>
typedef SRWLOCK turing_pool_lock;
typedef CONDITION_VARIABLE turing_pool_cond;
typedef HANDLE turing_pool_thread;
#define TURING_POOL_LOCK_INIT SRWLOCK_INIT
#define TURING_POOL_COND_INIT CONDITION_VARIABLE_INIT
#define TURING_POOL_TLS __declspec(thread)
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

typedef struct {
    turing_lane_fn fn;
    void* context;
    long chunks_per_lane;
    long total;
    long cursor;
    long completed;
    long observers;
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
    for (;;) {
        long index;
        pool_lock(&g_lock);
        if (frame->cursor >= frame->total) {
            pool_unlock(&g_lock);
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
