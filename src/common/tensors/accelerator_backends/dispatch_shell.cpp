// Dispatch decision shell: modules x backends x reduction-stages.
//
// Compiled as its own C++ translation unit and linked beside the cffi
// glue.  It cannot live inside the generated module: cffi's ABI-mode
// wrapper performs implicit void* to T* assignment, which is valid C and
// rejected by C++.

#include <stdint.h>
#include <vector>
#include <limits>

#if defined(_WIN32)
#include <windows.h>
static uint64_t turing_dispatch_now_ns(void) {
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (uint64_t)(
        ((double)counter.QuadPart * 1000000000.0)
        / (double)frequency.QuadPart
    );
}
#else
#include <time.h>
static uint64_t turing_dispatch_now_ns(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (uint64_t)value.tv_sec * UINT64_C(1000000000)
        + (uint64_t)value.tv_nsec;
}
#endif

extern "C" {

typedef int (*turing_compute_closure)(
    void *context,
    unsigned long long *device_ns
);

typedef struct TuringLaunchProfile {
    unsigned long long shell_ns;
    unsigned long long device_ns;
    int status;
    int language;
} TuringLaunchProfile;

struct TuringDispatchPlan {
    int modules;
    int backends;
    int stages;
    int feature_dim;

    std::vector<float> features;        /* modules * feature_dim */
    std::vector<double> work;           /* modules */
    std::vector<double> seconds;        /* modules * backends * stages */
    std::vector<double> device;         /* modules * backends * stages */
    std::vector<long long> counts;      /* modules * backends * stages */
    std::vector<int> decision_backend;  /* modules */
    std::vector<int> decision_stage;    /* modules */
    std::vector<double> backend_weight; /* backends */
    std::vector<int> edge_src;
    std::vector<int> edge_dst;
};

static bool plan_cell(
    const TuringDispatchPlan *plan,
    int module,
    int backend,
    int stage,
    size_t *index
) {
    if (plan == 0) return false;
    if (module < 0 || module >= plan->modules) return false;
    if (backend < 0 || backend >= plan->backends) return false;
    if (stage < 0 || stage >= plan->stages) return false;
    *index = ((size_t)module * (size_t)plan->backends + (size_t)backend)
        * (size_t)plan->stages + (size_t)stage;
    return true;
}

TuringDispatchPlan *turing_plan_create(
    int modules, int backends, int stages, int feature_dim
) {
    if (modules < 1 || backends < 1 || stages < 1 || feature_dim < 0) {
        return 0;
    }
    TuringDispatchPlan *plan = new TuringDispatchPlan();
    plan->modules = modules;
    plan->backends = backends;
    plan->stages = stages;
    plan->feature_dim = feature_dim;
    plan->features.assign((size_t)modules * (size_t)feature_dim, 0.0f);
    plan->work.assign((size_t)modules, 0.0);
    size_t cells = (size_t)modules * (size_t)backends * (size_t)stages;
    plan->seconds.assign(cells, 0.0);
    plan->device.assign(cells, 0.0);
    plan->counts.assign(cells, 0);
    plan->decision_backend.assign((size_t)modules, -1);
    plan->decision_stage.assign((size_t)modules, -1);
    plan->backend_weight.assign((size_t)backends, 1.0);
    return plan;
}

void turing_plan_destroy(TuringDispatchPlan *plan) { delete plan; }

int turing_plan_modules(const TuringDispatchPlan *p) {
    return p ? p->modules : 0;
}
int turing_plan_backends(const TuringDispatchPlan *p) {
    return p ? p->backends : 0;
}
int turing_plan_stages(const TuringDispatchPlan *p) {
    return p ? p->stages : 0;
}
int turing_plan_feature_dim(const TuringDispatchPlan *p) {
    return p ? p->feature_dim : 0;
}
int turing_plan_edge_count(const TuringDispatchPlan *p) {
    return p ? (int)p->edge_src.size() : 0;
}

int turing_plan_set_features(
    TuringDispatchPlan *plan, int module, const float *values
) {
    if (plan == 0 || values == 0) return 0;
    if (module < 0 || module >= plan->modules) return 0;
    for (int i = 0; i < plan->feature_dim; ++i) {
        plan->features[(size_t)module * plan->feature_dim + i] = values[i];
    }
    return 1;
}

int turing_plan_copy_features(const TuringDispatchPlan *plan, float *out) {
    if (plan == 0 || out == 0) return 0;
    for (size_t i = 0; i < plan->features.size(); ++i) out[i] = plan->features[i];
    return 1;
}

int turing_plan_set_work(TuringDispatchPlan *plan, int module, double work) {
    if (plan == 0 || module < 0 || module >= plan->modules) return 0;
    plan->work[(size_t)module] = work;
    return 1;
}

int turing_plan_copy_work(const TuringDispatchPlan *plan, double *out) {
    if (plan == 0 || out == 0) return 0;
    for (size_t i = 0; i < plan->work.size(); ++i) out[i] = plan->work[i];
    return 1;
}

int turing_plan_set_edges(
    TuringDispatchPlan *plan, const int *src, const int *dst, int count
) {
    if (plan == 0 || count < 0) return 0;
    if (count > 0 && (src == 0 || dst == 0)) return 0;
    plan->edge_src.clear();
    plan->edge_dst.clear();
    for (int i = 0; i < count; ++i) {
        if (src[i] < 0 || src[i] >= plan->modules) return 0;
        if (dst[i] < 0 || dst[i] >= plan->modules) return 0;
        plan->edge_src.push_back(src[i]);
        plan->edge_dst.push_back(dst[i]);
    }
    return 1;
}

int turing_plan_copy_edge_index(const TuringDispatchPlan *plan, int *out) {
    if (plan == 0 || out == 0) return 0;
    size_t count = plan->edge_src.size();
    for (size_t i = 0; i < count; ++i) {
        out[i] = plan->edge_src[i];
        out[count + i] = plan->edge_dst[i];
    }
    return 1;
}

int turing_plan_observe(
    TuringDispatchPlan *plan,
    int module, int backend, int stage,
    double seconds, double device_seconds
) {
    size_t index;
    if (!plan_cell(plan, module, backend, stage, &index)) return 0;
    plan->seconds[index] += seconds;
    plan->device[index] += device_seconds;
    plan->counts[index] += 1;
    return 1;
}

/* Mean seconds per observed cell; NaN where nothing has been measured, so an
   unexplored option is never mistaken for a free one. */
int turing_plan_copy_observations(const TuringDispatchPlan *plan, double *out) {
    if (plan == 0 || out == 0) return 0;
    for (size_t i = 0; i < plan->seconds.size(); ++i) {
        out[i] = plan->counts[i] > 0
            ? plan->seconds[i] / (double)plan->counts[i]
            : std::numeric_limits<double>::quiet_NaN();
    }
    return 1;
}

int turing_plan_copy_device(const TuringDispatchPlan *plan, double *out) {
    if (plan == 0 || out == 0) return 0;
    for (size_t i = 0; i < plan->device.size(); ++i) {
        out[i] = plan->counts[i] > 0
            ? plan->device[i] / (double)plan->counts[i]
            : std::numeric_limits<double>::quiet_NaN();
    }
    return 1;
}

int turing_plan_copy_counts(const TuringDispatchPlan *plan, long long *out) {
    if (plan == 0 || out == 0) return 0;
    for (size_t i = 0; i < plan->counts.size(); ++i) out[i] = plan->counts[i];
    return 1;
}

int turing_plan_set_decision(
    TuringDispatchPlan *plan, int module, int backend, int stage
) {
    size_t index;
    if (!plan_cell(plan, module, backend, stage, &index)) return 0;
    plan->decision_backend[(size_t)module] = backend;
    plan->decision_stage[(size_t)module] = stage;
    return 1;
}

int turing_plan_backend_for(const TuringDispatchPlan *plan, int module) {
    if (plan == 0 || module < 0 || module >= plan->modules) return -1;
    return plan->decision_backend[(size_t)module];
}

int turing_plan_stage_for(const TuringDispatchPlan *plan, int module) {
    if (plan == 0 || module < 0 || module >= plan->modules) return -1;
    return plan->decision_stage[(size_t)module];
}

int turing_plan_copy_decisions(const TuringDispatchPlan *plan, int *out) {
    if (plan == 0 || out == 0) return 0;
    for (int m = 0; m < plan->modules; ++m) {
        out[(size_t)m * 2 + 0] = plan->decision_backend[(size_t)m];
        out[(size_t)m * 2 + 1] = plan->decision_stage[(size_t)m];
    }
    return 1;
}

int turing_plan_set_backend_weights(
    TuringDispatchPlan *plan, const double *weights
) {
    if (plan == 0 || weights == 0) return 0;
    for (int b = 0; b < plan->backends; ++b) {
        plan->backend_weight[(size_t)b] = weights[b];
    }
    return 1;
}

/* Weighted cost of one option.  Measured mean time scaled by the backend's
   weight, then divided by apparent work so modules of different size are
   comparable.  NaN when unmeasured. */
double turing_plan_score(
    const TuringDispatchPlan *plan, int module, int backend, int stage
) {
    size_t index;
    if (!plan_cell(plan, module, backend, stage, &index)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (plan->counts[index] <= 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double mean = plan->seconds[index] / (double)plan->counts[index];
    double weighted = mean * plan->backend_weight[(size_t)backend];
    double work = plan->work[(size_t)module];
    return work > 0.0 ? weighted / work : weighted;
}

int turing_plan_select_best(TuringDispatchPlan *plan, int module) {
    if (plan == 0 || module < 0 || module >= plan->modules) return -1;
    double best = std::numeric_limits<double>::infinity();
    int best_backend = -1;
    int best_stage = -1;
    for (int b = 0; b < plan->backends; ++b) {
        for (int s = 0; s < plan->stages; ++s) {
            double score = turing_plan_score(plan, module, b, s);
            if (score == score && score < best) {  /* score == score: not NaN */
                best = score;
                best_backend = b;
                best_stage = s;
            }
        }
    }
    if (best_backend >= 0) {
        plan->decision_backend[(size_t)module] = best_backend;
        plan->decision_stage[(size_t)module] = best_stage;
    }
    return best_backend;
}

int turing_plan_select_all(TuringDispatchPlan *plan) {
    if (plan == 0) return 0;
    int decided = 0;
    for (int m = 0; m < plan->modules; ++m) {
        if (turing_plan_select_best(plan, m) >= 0) decided += 1;
    }
    return decided;
}

/* Launch one module through its current decision and record the result into
   the cube, so the decision surface improves simply by running. */
int turing_plan_launch(
    TuringDispatchPlan *plan,
    int module,
    turing_compute_closure compute,
    void *context,
    TuringLaunchProfile *profile
) {
    uint64_t started;
    uint64_t finished;
    unsigned long long device_ns = 0;
    int status;
    int backend;
    int stage;

    if (plan == 0 || compute == 0 || profile == 0) return 0;
    if (module < 0 || module >= plan->modules) return 0;
    backend = plan->decision_backend[(size_t)module];
    stage = plan->decision_stage[(size_t)module];

    started = turing_dispatch_now_ns();
    status = compute(context, &device_ns);
    finished = turing_dispatch_now_ns();

    profile->shell_ns = finished - started;
    profile->device_ns = device_ns;
    profile->status = status;
    profile->language = backend;

    if (backend >= 0 && stage >= 0) {
        turing_plan_observe(
            plan, module, backend, stage,
            (double)(finished - started) * 1.0e-9,
            (double)device_ns * 1.0e-9
        );
    }
    return status;
}

}  /* extern "C" */
