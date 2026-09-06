// Persistent worker-pool ABI for deployment frames.
//
// Shell-class targets link ``turing_pool.c`` and gain the ``pool`` lowering
// of a Deploy/Join frame: lanes and chunks are claimed from a shared atomic
// cursor by workers that start once and park between frames ("already up
// and waiting for jobs").  The pool is never load-bearing for semantics --
// with zero workers the deploying thread drains every job itself through
// the identical claim loop, which IS the serial fallback, so both
// dispositions run the same schedule and a target without the pool simply
// runs the recorded linear order.
//
// Claiming is one atomic fetch-add: for frames whose jobs are all known at
// deploy time, work stealing degenerates to a shared cursor -- every job is
// claimed exactly once (linearizability of the increment), balance is
// chunk-granular, and the join is a completion count.  This matches the
// repository's threading policy (docs/BACKEND_PERFORMANCE_HANDOFF.md):
// static/guided partitioning for uniform work; richer stealing belongs at
// the module-DAG tier, not here.
//
// The join implemented here is BARRIER.  INDEXED is the same wait with
// per-job result slots owned by the caller's context.  REDUCE and PRODUCT
// joins need a dedicated consumer and are refused at the deploy call.
#ifndef TURING_POOL_H
#define TURING_POOL_H

#ifdef __cplusplus
extern "C" {
#endif

// One unit of claimed work.  ``lane`` indexes the independent strand,
// ``chunk`` the element-range slice inside it; a frame with
// ``chunks_per_lane == 1`` is plain lane parallelism, a frame with
// ``lane_count == 1`` is plain element splitting.
typedef void (*turing_lane_fn)(void* context, long lane, long chunk,
                               long chunks_per_lane);

// One claim over a half-open item range. This is the native counterpart of
// HostDeploymentPool.deploy_span and the ABI used by tiled BLAS lanes:
// ``chunk_size`` controls claim granularity while every item is visited once.
typedef void (*turing_span_fn)(void* context, long start, long stop);

// Start (or grow to) ``workers`` parked worker threads.  Idempotent;
// never shrinks.  Returns the resulting worker count, or -1 on failure.
int turing_pool_start(int workers);

// Number of running workers (0 before any start).
int turing_pool_workers(void);

// Run one deployment frame: ``lane_count * chunks_per_lane`` jobs, each
// invoked exactly once as fn(context, lane, chunk, chunks_per_lane).
// Blocks until every job has completed (BARRIER join).  The calling
// thread participates in draining.  One frame may be in flight at a
// time; a nested deploy from inside a lane returns -2 instead of
// deadlocking.  Returns 0 on success, -1 on invalid arguments.
int turing_pool_deploy(turing_lane_fn fn, void* context, long lane_count,
                       long chunks_per_lane);

// Split [0, item_count) into ceil(item_count/chunk_size) independent claims.
// Each callback receives one nonempty half-open span. The call participates
// in and joins the same persistent pool as turing_pool_deploy.
int turing_pool_deploy_span(turing_span_fn fn, void* context, long item_count,
                            long chunk_size);

// Stop and join every worker.  Safe to call with none running; the pool
// may be started again afterwards.
void turing_pool_stop(void);

// Advisory critical section for ORDER-INSENSITIVE shared effects inside a
// deployed lane -- the sanctioned case is an append of a lane-invariant
// value, where every iteration pushes an identical element so only the
// count is observable and atomicity alone preserves serial semantics.
// This lock is deliberately separate from the pool's own scheduling lock,
// so holding it can never interact with claim/park logic.  It must not be
// used to launder order-DEPENDENT effects; those need an indexed join.
void turing_pool_effect_lock(void);
void turing_pool_effect_unlock(void);

// Dispatcher-owned communicating task, independent of the numerical frame
// pool. The entry is compiled native code and context is its captured frame.
// No serial fallback: start either creates a concurrent task or returns NULL.
typedef struct turing_dispatch_task turing_dispatch_task;
typedef void (*turing_task_fn)(void* context);
turing_dispatch_task* turing_task_start(turing_task_fn fn, void* context);
// Wait for completion: timeout_ms < 0 means indefinitely. Returns 1 when
// complete, 0 on timeout, -1 on invalid/OS error, -2 for a self-wait.
int turing_task_wait(turing_dispatch_task* task, long timeout_ms);
// Join and free the handle. Caller must own its lifetime exclusively; captured
// storage must remain alive until this succeeds. Returns 0, or a negative error.
int turing_task_destroy(turing_dispatch_task* task);

// Dispatcher condition with an owned recursive lock (Python Condition's
// default). Waiting releases ALL recursive acquisitions and restores their
// depth before returning. notify requires ownership and does not release it.
// This is communicating-task synchronization, never a numerical pool barrier.
typedef struct turing_dispatch_condition turing_dispatch_condition;
turing_dispatch_condition* turing_dispatch_condition_create(void);
int turing_dispatch_condition_acquire(turing_dispatch_condition* condition);
int turing_dispatch_condition_release(turing_dispatch_condition* condition);
// timeout_ms < 0 means indefinite; 0 polls. Returns 1 when notified, 0 on
// timeout, -1 on invalid ownership/OS error. The frontend maps Python's
// negative timeout to zero and None to the indefinite sentinel.
int turing_dispatch_condition_wait(turing_dispatch_condition* condition, long timeout_ms);
int turing_dispatch_condition_notify(turing_dispatch_condition* condition, long count);
int turing_dispatch_condition_notify_all(turing_dispatch_condition* condition);
// Exclusive lifetime ownership required; refuses an acquired/waited-on lock.
int turing_dispatch_condition_destroy(turing_dispatch_condition* condition);

// Manual-reset event. set wakes every current waiter; clear affects future
// waits and does not revoke notifications already delivered. Predicate/wait
// results are 0 or 1, with negative values reserved for runtime errors.
typedef struct turing_dispatch_event turing_dispatch_event;
turing_dispatch_event* turing_dispatch_event_create(void);
int turing_dispatch_event_set(turing_dispatch_event* event);
int turing_dispatch_event_clear(turing_dispatch_event* event);
int turing_dispatch_event_is_set(turing_dispatch_event* event);
int turing_dispatch_event_wait(turing_dispatch_event* event, long timeout_ms);
// As with Condition, exclusive lifetime ownership is required.
int turing_dispatch_event_destroy(turing_dispatch_event* event);

#ifdef __cplusplus
}
#endif

#endif  // TURING_POOL_H
