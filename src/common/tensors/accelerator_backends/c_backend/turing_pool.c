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

#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
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

int turing_task_is_alive(turing_dispatch_task* task) {
    int alive;
    if (!task) return 0;
    pool_lock(&task->lock);
    alive = !task->done;
    pool_unlock(&task->lock);
    return alive;
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

/* One native state implementation, shared by SSA primitives and the historical
 * Condition/Event ABI. Logical mutex ownership is distinct from the short
 * monitor critical sections used to update it. */
typedef struct turing_sync_epoch {
    uint64_t generation;
    unsigned long waiters;
    int broken;
    struct turing_sync_epoch *next;
} turing_sync_epoch;
struct turing_sync_latch {
    turing_pool_lock monitor;
    turing_pool_cond changed;
    uint64_t generation, remaining, participants;
    unsigned long active_waiters, generation_waiters;
    turing_sync_epoch *completed;
    int released, cyclic, broken, default_finite;
    double default_timeout;
    struct turing_sync_latch *next;
    turing_dispatch_condition *registration;
};
struct turing_dispatch_condition {
    turing_pool_lock monitor;
    turing_pool_cond available;
    uint64_t owner, depth, generation, held_token;
    unsigned long acquiring;
    int reentrant;
    turing_sync_latch *waiters, *waiter_tail;
    turing_dispatch_condition *guard;
};
static TURING_POOL_TLS _Alignas(8) int t_condition_owner;
static TURING_POOL_TLS uint64_t t_source_worker_context;
void turing_sync_set_worker_context(uint64_t source_worker) {
    if (source_worker > (UINT64_MAX >> 1)) abort();
    /* Reuse the management table's logical instance token. Odd encoded
     * source IDs cannot collide with aligned legacy caller addresses. */
    t_source_worker_context = source_worker ? (source_worker << 1) | 1 : 0;
}
uint64_t turing_sync_worker_token(void) {
    return t_source_worker_context ? t_source_worker_context : (uint64_t)(uintptr_t)&t_condition_owner;
}

static int sync_init(turing_pool_lock *lock, turing_pool_cond *cond) {
#ifdef _WIN32
    InitializeSRWLock(lock); InitializeConditionVariable(cond); return 0;
#else
    pthread_condattr_t attributes;
    if (pthread_mutex_init(lock, NULL)) return -1;
    if (pthread_condattr_init(&attributes)) { pthread_mutex_destroy(lock); return -1; }
#if defined(__APPLE__)
    /* Apple's pthread conditions retain their default real-time clock. */
#else
    if (pthread_condattr_setclock(&attributes, CLOCK_MONOTONIC)) {
        pthread_condattr_destroy(&attributes); pthread_mutex_destroy(lock); return -1;
    }
#endif
    int status = pthread_cond_init(cond, &attributes);
    pthread_condattr_destroy(&attributes);
    if (status) { pthread_mutex_destroy(lock); return -1; }
    return 0;
#endif
}
static void sync_dispose(turing_pool_lock *lock, turing_pool_cond *cond) {
#ifndef _WIN32
    pthread_cond_destroy(cond); pthread_mutex_destroy(lock);
#else
    (void)lock; (void)cond;
#endif
}
typedef struct {
    int finite;
#ifdef _WIN32
    LONGLONG ticks, frequency;
#else
    struct timespec at;
#endif
} sync_deadline;
static int sync_deadline_init(sync_deadline *deadline, double seconds, int finite) {
    deadline->finite = finite;
    if (!finite) return 0;
    if (seconds != seconds) return -3;
    if (seconds < 0) seconds = 0;
    /* Refuse an unrepresentable platform interval; never shorten a wait. */
#ifdef _WIN32
    if (seconds > 4294967.0) return -2;
    LARGE_INTEGER frequency, now;
    if (!QueryPerformanceFrequency(&frequency) || !frequency.QuadPart || !QueryPerformanceCounter(&now)) return -1;
    double interval = seconds * (double)frequency.QuadPart;
    if (interval >= (double)(LLONG_MAX - now.QuadPart)) return -2;
    LONGLONG ticks = (LONGLONG)interval;
    if ((double)ticks < interval) ticks++;
    deadline->frequency = frequency.QuadPart;
    deadline->ticks = now.QuadPart + ticks;
#else
    if (seconds > 9223372036.0) return -2;
#if defined(__APPLE__)
    if (clock_gettime(CLOCK_REALTIME, &deadline->at)) return -1;
#else
    if (clock_gettime(CLOCK_MONOTONIC, &deadline->at)) return -1;
#endif
    if (sizeof(time_t) < 8 && seconds > (double)INT32_MAX - (double)deadline->at.tv_sec) return -2;
    time_t whole = (time_t)seconds;
    deadline->at.tv_sec += whole;
    double fractional = (seconds - (double)whole) * 1000000000.0;
    long nanoseconds = (long)fractional;
    if ((double)nanoseconds < fractional) nanoseconds++;
    deadline->at.tv_nsec += nanoseconds;
    if (deadline->at.tv_nsec >= 1000000000L) {
        deadline->at.tv_sec++; deadline->at.tv_nsec -= 1000000000L;
    }
#endif
    return 0;
}
/* 1 signaled/spurious, 0 timeout, -1 platform error. Caller loops on state. */
static int sync_park(turing_pool_cond *cond, turing_pool_lock *lock, sync_deadline *deadline) {
#ifdef _WIN32
    DWORD remaining = INFINITE;
    if (deadline->finite) {
        LARGE_INTEGER now;
        if (!QueryPerformanceCounter(&now)) return -1;
        if (now.QuadPart >= deadline->ticks) return 0;
        double milliseconds = ((double)(deadline->ticks - now.QuadPart) / (double)deadline->frequency) * 1000.0;
        remaining = milliseconds >= (double)(INFINITE - 1) ? INFINITE - 1 : (DWORD)milliseconds;
        if ((double)remaining < milliseconds && remaining < INFINITE - 1) remaining++;
    }
    if (SleepConditionVariableSRW(cond, lock, remaining, 0)) return 1;
    return GetLastError() == ERROR_TIMEOUT ? 0 : -1;
#else
    int status = deadline->finite ? pthread_cond_timedwait(cond, lock, &deadline->at)
                                  : pthread_cond_wait(cond, lock);
    return status == 0 ? 1 : status == ETIMEDOUT ? 0 : -1;
#endif
}
turing_dispatch_condition* turing_sync_mutex_create(int reentrant) {
    turing_dispatch_condition *mutex = calloc(1, sizeof(*mutex));
    if (!mutex) return NULL;
    if (sync_init(&mutex->monitor, &mutex->available)) { free(mutex); return NULL; }
    mutex->reentrant = !!reentrant;
    return mutex;
}
turing_dispatch_condition* turing_sync_wait_queue_mutex(turing_dispatch_condition *queue) {
    return queue && queue->guard ? queue->guard : queue;
}
turing_dispatch_condition* turing_sync_wait_queue_create(turing_dispatch_condition *mutex) {
    if (!mutex) return NULL;
    turing_dispatch_condition *queue = calloc(1, sizeof(*queue));
    if (queue) queue->guard = turing_sync_wait_queue_mutex(mutex);
    return queue;
}
int64_t turing_sync_mutex_acquire(turing_dispatch_condition *mutex, uint64_t worker, int blocking, double timeout) {
    sync_deadline deadline;
    if (!mutex || !worker) return -1;
    if ((!blocking && timeout != -1.0) || timeout < -1.0 || timeout != timeout) return -3;
    int deadline_status = sync_deadline_init(&deadline, timeout, timeout >= 0);
    if (deadline_status) return deadline_status;
    pool_lock(&mutex->monitor);
    mutex->acquiring++;
    while (mutex->owner && !(mutex->reentrant && mutex->owner == worker)) {
        int status;
        if (!blocking) { mutex->acquiring--; pool_unlock(&mutex->monitor); return 0; }
        status = sync_park(&mutex->available, &mutex->monitor, &deadline);
        if (status != 1) { mutex->acquiring--; pool_unlock(&mutex->monitor); return status < 0 ? -1 : 0; }
    }
    mutex->acquiring--;
    if (!mutex->owner) {
        mutex->generation++;
        if (!mutex->generation || mutex->generation > INT64_MAX) abort();
        mutex->held_token = mutex->generation;
        mutex->owner = worker;
    }
    mutex->depth++;
    int64_t token = (int64_t)mutex->held_token;
    pool_unlock(&mutex->monitor);
    return token;
}
int64_t turing_sync_mutex_token(turing_dispatch_condition *mutex, uint64_t worker) {
    if (!mutex) return 0;
    pool_lock(&mutex->monitor);
    int64_t token = mutex->owner == worker ? (int64_t)mutex->held_token : 0;
    pool_unlock(&mutex->monitor);
    return token;
}
static int sync_owns(turing_dispatch_condition *mutex, uint64_t worker, uint64_t token) {
    return worker && mutex->owner == worker && mutex->depth && token && mutex->held_token == token;
}
int turing_sync_mutex_release(turing_dispatch_condition *mutex, uint64_t worker, uint64_t token) {
    if (!mutex) return -1;
    pool_lock(&mutex->monitor);
    if (!sync_owns(mutex, worker, token)) { pool_unlock(&mutex->monitor); return -1; }
    if (--mutex->depth == 0) {
        mutex->owner = 0; mutex->held_token = 0;
        pool_notify_all(&mutex->available);
    }
    pool_unlock(&mutex->monitor); return 1;
}
int64_t turing_sync_mutex_suspend(turing_dispatch_condition *mutex, uint64_t worker, uint64_t token) {
    if (!mutex) return 0;
    pool_lock(&mutex->monitor);
    if (!sync_owns(mutex, worker, token)) { pool_unlock(&mutex->monitor); return 0; }
    uint64_t depth = mutex->depth;
    mutex->depth = 0; mutex->owner = 0; mutex->held_token = 0;
    pool_notify_all(&mutex->available);
    pool_unlock(&mutex->monitor); return (int64_t)depth;
}
int64_t turing_sync_mutex_restore(turing_dispatch_condition *mutex, uint64_t worker, uint64_t depth) {
    if (!depth || depth > INT64_MAX) return 0;
    int64_t token = turing_sync_mutex_acquire(mutex, worker, 1, -1.0);
    if (token <= 0) return 0;
    pool_lock(&mutex->monitor);
    mutex->depth = depth;
    pool_unlock(&mutex->monitor); return token;
}
int turing_sync_mutex_locked(turing_dispatch_condition *mutex) {
    if (!mutex) return -1;
    pool_lock(&mutex->monitor); int held = mutex->depth != 0;
    pool_unlock(&mutex->monitor); return held;
}
turing_sync_latch* turing_sync_latch_create(uint64_t participants, uint64_t generation) {
    turing_sync_latch *latch = calloc(1, sizeof(*latch));
    if (!latch) return NULL;
    if (sync_init(&latch->monitor, &latch->changed)) { free(latch); return NULL; }
    latch->generation = generation; latch->remaining = participants;
    latch->participants = participants;
    latch->released = participants == 0;
    return latch;
}
int turing_sync_latch_signal(turing_sync_latch *latch, uint64_t generation) {
    if (!latch) return -1;
    pool_lock(&latch->monitor);
    if (latch->cyclic || latch->broken || latch->generation != generation) { pool_unlock(&latch->monitor); return -1; }
    latch->released = 1; latch->remaining = 0;
    pool_notify_all(&latch->changed); pool_unlock(&latch->monitor); return 1;
}
/* Completed epochs remain addressable until exclusive latch destruction.
 * Reset cannot make a released old-generation waiter sleep on a new epoch. */
static int sync_latch_state(turing_sync_latch *latch, uint64_t generation) {
    if (generation == latch->generation) return latch->broken ? -4 : latch->released;
    for (turing_sync_epoch *epoch = latch->completed; epoch; epoch = epoch->next)
        if (epoch->generation == generation) return epoch->broken ? -4 : 1;
    return -1;
}
static int sync_latch_arrive_locked(turing_sync_latch *latch, uint64_t generation, uint64_t count) {
    if (generation != latch->generation || !count || count > latch->remaining || latch->released || latch->broken) return -1;
    latch->remaining -= count;
    if (!latch->remaining) {
        latch->released = 1;
        pool_notify_all(&latch->changed);
    }
    return 1;
}
int turing_sync_latch_arrive(turing_sync_latch *latch, uint64_t generation, uint64_t count) {
    if (!latch) return -1;
    pool_lock(&latch->monitor);
    int status = latch->cyclic ? -1 : sync_latch_arrive_locked(latch, generation, count);
    pool_unlock(&latch->monitor); return status;
}
static int sync_latch_rearm_locked(turing_sync_latch *latch, uint64_t generation, uint64_t participants, int broken) {
    if (generation <= latch->generation) return -2;
    if (!latch->cyclic || latch->generation_waiters) {
        turing_sync_epoch *epoch = malloc(sizeof(*epoch));
        if (!epoch) return -5;
        epoch->generation = latch->generation; epoch->waiters = latch->generation_waiters;
        epoch->broken = broken; epoch->next = latch->completed; latch->completed = epoch;
    }
    latch->generation = generation; latch->remaining = participants; latch->participants = participants;
    latch->generation_waiters = 0; latch->released = participants == 0; latch->broken = 0;
    pool_notify_all(&latch->changed); return 1;
}
int turing_sync_latch_reset(turing_sync_latch *latch, uint64_t previous, uint64_t generation, uint64_t participants) {
    if (!latch) return -1;
    pool_lock(&latch->monitor);
    int status = (latch->cyclic || previous != latch->generation || generation <= previous
                  || (!latch->released && !latch->broken) || latch->registration) ? -1 :
                  sync_latch_rearm_locked(latch, generation, participants, latch->broken);
    pool_unlock(&latch->monitor); return status;
}
static int sync_latch_wait_locked(turing_sync_latch *latch, uint64_t generation, sync_deadline *deadline) {
    int result;
    latch->active_waiters++;
    if (generation == latch->generation) latch->generation_waiters++;
    else {
        for (turing_sync_epoch *epoch = latch->completed; epoch; epoch = epoch->next)
            if (epoch->generation == generation) { epoch->waiters++; break; }
    }
    while ((result = sync_latch_state(latch, generation)) == 0) {
        int status = sync_park(&latch->changed, &latch->monitor, deadline);
        if (status != 1) {
            result = sync_latch_state(latch, generation);
            if (!result) result = status;
            break;
        }
    }
    latch->active_waiters--;
    if (generation == latch->generation) latch->generation_waiters--;
    else {
        turing_sync_epoch **link = &latch->completed;
        while (*link && (*link)->generation != generation) link = &(*link)->next;
        if (*link) {
            turing_sync_epoch *epoch = *link;
            epoch->waiters--;
            if (latch->cyclic && !epoch->waiters) { *link = epoch->next; free(epoch); }
        }
    }
    return result;
}
int turing_sync_latch_wait(turing_sync_latch *latch, uint64_t generation, double timeout, int finite) {
    sync_deadline deadline;
    if (!latch) return -1;
    int status = sync_deadline_init(&deadline, timeout, finite);
    if (status) return status;
    pool_lock(&latch->monitor);
    status = latch->cyclic ? -1 : sync_latch_wait_locked(latch, generation, &deadline);
    pool_unlock(&latch->monitor); return status;
}
int turing_sync_control_barrier(turing_sync_latch *latch, uint64_t generation, uint64_t participants) {
    sync_deadline deadline;
    if (!latch || !participants) return -1;
    sync_deadline_init(&deadline, 0, 0);
    pool_lock(&latch->monitor);
    int status = !latch->cyclic && latch->participants == participants ? sync_latch_arrive_locked(latch, generation, 1) : -1;
    if (status == 1) status = sync_latch_wait_locked(latch, generation, &deadline);
    if (status < 0 && !latch->cyclic && generation == latch->generation) {
        latch->broken = 1; pool_notify_all(&latch->changed);
    }
    pool_unlock(&latch->monitor); return status;
}
int turing_sync_barrier_configure(turing_sync_latch *latch, double timeout, int finite) {
    if (!latch || !latch->participants || latch->participants > INT64_MAX) return -1;
    latch->cyclic = 1; latch->default_timeout = timeout; latch->default_finite = finite;
    return 1;
}
int64_t turing_sync_barrier_wait(turing_sync_latch *latch, double timeout, int finite) {
    sync_deadline deadline;
    if (!latch) return -1;
    pool_lock(&latch->monitor);
    if (!latch->cyclic || latch->broken) { pool_unlock(&latch->monitor); return -4; }
    if (!finite) { timeout = latch->default_timeout; finite = latch->default_finite; }
    int status = sync_deadline_init(&deadline, timeout, finite);
    if (status) {
        latch->broken = 1; pool_notify_all(&latch->changed);
        pool_unlock(&latch->monitor); return status;
    }
    uint64_t generation = latch->generation;
    int64_t index = (int64_t)(latch->participants - latch->remaining);
    status = sync_latch_arrive_locked(latch, generation, 1);
    if (status == 1 && latch->released) {
        status = sync_latch_rearm_locked(latch, generation + 1, latch->participants, 0);
        if (status < 0) { latch->broken = 1; pool_notify_all(&latch->changed); }
    } else if (status == 1) {
        status = sync_latch_wait_locked(latch, generation, &deadline);
        if (status <= 0 && generation == latch->generation) {
            latch->broken = 1; pool_notify_all(&latch->changed);
        }
        if (status == 0) status = -4;
    }
    pool_unlock(&latch->monitor); return status < 0 ? status : index;
}
int turing_sync_barrier_abort(turing_sync_latch *latch) {
    if (!latch) return -1;
    pool_lock(&latch->monitor);
    if (!latch->cyclic) { pool_unlock(&latch->monitor); return -1; }
    /* Abort also faults released participants still parked in an older
     * cyclic epoch. Explicit latch reset retains completed epochs instead. */
    for (turing_sync_epoch *epoch = latch->completed; epoch; epoch = epoch->next) epoch->broken = 1;
    latch->broken = 1; pool_notify_all(&latch->changed);
    pool_unlock(&latch->monitor); return 1;
}
int turing_sync_barrier_reset(turing_sync_latch *latch) {
    if (!latch) return -1;
    pool_lock(&latch->monitor);
    int status = latch->cyclic ? sync_latch_rearm_locked(latch, latch->generation + 1, latch->participants, 1) : -1;
    if (status < 0 && latch->cyclic) { latch->broken = 1; pool_notify_all(&latch->changed); }
    pool_unlock(&latch->monitor); return status;
}
int turing_sync_latch_destroy(turing_sync_latch *latch) {
    if (!latch) return -1;
    pool_lock(&latch->monitor);
    if (latch->registration || latch->active_waiters) { pool_unlock(&latch->monitor); return -1; }
    turing_sync_epoch *epoch = latch->completed;
    while (epoch) { turing_sync_epoch *next = epoch->next; free(epoch); epoch = next; }
    pool_unlock(&latch->monitor);
    sync_dispose(&latch->monitor, &latch->changed); free(latch); return 1;
}
int turing_sync_waiter_register(turing_dispatch_condition *queue, uint64_t worker, uint64_t token, turing_sync_latch *latch) {
    turing_dispatch_condition *mutex = turing_sync_wait_queue_mutex(queue);
    if (!mutex || !latch) return -1;
    pool_lock(&mutex->monitor);
    if (!sync_owns(mutex, worker, token) || latch->registration) { pool_unlock(&mutex->monitor); return -1; }
    latch->registration = queue;
    if (queue->waiter_tail) queue->waiter_tail->next = latch;
    else queue->waiters = latch;
    queue->waiter_tail = latch;
    pool_unlock(&mutex->monitor); return 1;
}
turing_sync_latch* turing_sync_waiter_select(turing_dispatch_condition *queue, uint64_t worker, uint64_t token) {
    turing_dispatch_condition *mutex = turing_sync_wait_queue_mutex(queue);
    turing_sync_latch *latch;
    if (!mutex) return NULL;
    pool_lock(&mutex->monitor);
    if (!sync_owns(mutex, worker, token)) { pool_unlock(&mutex->monitor); return NULL; }
    for (latch = queue->waiters; latch; latch = latch->next) {
        pool_lock(&latch->monitor); int released = latch->released;
        pool_unlock(&latch->monitor);
        if (!released) break;
    }
    pool_unlock(&mutex->monitor); return latch;
}
int turing_sync_waiter_retire(turing_dispatch_condition *queue, uint64_t worker, turing_sync_latch *latch) {
    turing_dispatch_condition *mutex = turing_sync_wait_queue_mutex(queue);
    if (!mutex || !latch) return -1;
    pool_lock(&mutex->monitor);
    if (mutex->owner != worker || latch->registration != queue) { pool_unlock(&mutex->monitor); return -1; }
    turing_sync_latch **link = &queue->waiters, *previous = NULL;
    while (*link && *link != latch) { previous = *link; link = &(*link)->next; }
    if (!*link) { pool_unlock(&mutex->monitor); return -1; }
    *link = latch->next;
    if (queue->waiter_tail == latch) queue->waiter_tail = previous;
    latch->registration = NULL; latch->next = NULL;
    pool_unlock(&mutex->monitor); return 1;
}
/* Compatibility ABI uses exactly the primitives SSA uses, not a second
 * condition engine. Native SSA's notify loop lives in its actual CFG. */
turing_dispatch_condition* turing_dispatch_condition_create(void) { return turing_sync_mutex_create(1); }
int turing_dispatch_condition_acquire(turing_dispatch_condition *m) {
    return turing_sync_mutex_acquire(turing_sync_wait_queue_mutex(m), turing_sync_worker_token(), 1, -1) > 0 ? 1 : -1;
}
int turing_dispatch_condition_release(turing_dispatch_condition *m) {
    uint64_t worker = turing_sync_worker_token();
    m = turing_sync_wait_queue_mutex(m);
    return turing_sync_mutex_release(m, worker, turing_sync_mutex_token(m, worker)) == 1 ? 0 : -1;
}
int turing_dispatch_condition_wait(turing_dispatch_condition *m, long timeout_ms) {
    uint64_t worker = turing_sync_worker_token();
    turing_dispatch_condition *mutex = turing_sync_wait_queue_mutex(m);
    int64_t token = turing_sync_mutex_token(mutex, worker);
    if (!token) return -1;
    turing_sync_latch *latch = turing_sync_latch_create(1, 0);
    if (!latch) return -1;
    if (turing_sync_waiter_register(m, worker, token, latch) != 1) { turing_sync_latch_destroy(latch); return -1; }
    int64_t depth = turing_sync_mutex_suspend(mutex, worker, token);
    int result = depth ? turing_sync_latch_wait(latch, 0, (double)timeout_ms / 1000.0, timeout_ms >= 0) : -1;
    if (depth && !turing_sync_mutex_restore(mutex, worker, depth)) return -1;
    if (turing_sync_waiter_retire(m, worker, latch) != 1) return -1;
    turing_sync_latch_destroy(latch); return result;
}
int turing_dispatch_condition_notify(turing_dispatch_condition *m, long count) {
    uint64_t worker = turing_sync_worker_token();
    int64_t token = turing_sync_mutex_token(turing_sync_wait_queue_mutex(m), worker);
    if (!token) return -1;
    while (count-- > 0) {
        turing_sync_latch *latch = turing_sync_waiter_select(m, worker, token);
        if (!latch) break;
        if (turing_sync_latch_signal(latch, 0) != 1) return -1;
    }
    return 0;
}
int turing_dispatch_condition_notify_all(turing_dispatch_condition *m) { return turing_dispatch_condition_notify(m, LONG_MAX); }
int turing_dispatch_condition_destroy(turing_dispatch_condition *m) {
    if (!m) return -1;
    if (m->guard) {
        pool_lock(&m->guard->monitor);
        int busy = m->waiters != NULL;
        pool_unlock(&m->guard->monitor);
        if (busy) return -1;
        free(m); return 0;
    }
    pool_lock(&m->monitor);
    if (m->owner || m->waiters || m->acquiring) { pool_unlock(&m->monitor); return -1; }
    pool_unlock(&m->monitor); sync_dispose(&m->monitor, &m->available); free(m); return 0;
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
