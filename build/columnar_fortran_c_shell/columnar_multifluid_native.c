
#include <stdint.h>

#if defined(_WIN32)
#include <windows.h>
static uint64_t turing_monotonic_ns(void) {
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
static uint64_t turing_monotonic_ns(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (
        (uint64_t)value.tv_sec * UINT64_C(1000000000)
        + (uint64_t)value.tv_nsec
    );
}
#endif

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

/* Accumulated across launches so a caller can characterise a dispatch without
   round-tripping every individual call into Python, which would cost more than
   the launch being measured. */
typedef struct TuringLaunchStats {
    unsigned long long calls;
    unsigned long long failures;
    unsigned long long shell_ns_total;
    unsigned long long shell_ns_min;
    unsigned long long shell_ns_max;
    unsigned long long device_ns_total;
    unsigned long long overhead_ns;
} TuringLaunchStats;

typedef void (*turing_launch_logger)(
    void *user,
    const TuringLaunchProfile *profile
);

void turing_launch_stats_reset(TuringLaunchStats *stats) {
    if (stats == 0) {
        return;
    }
    stats->calls = 0;
    stats->failures = 0;
    stats->shell_ns_total = 0;
    stats->shell_ns_min = ~(unsigned long long)0;
    stats->shell_ns_max = 0;
    stats->device_ns_total = 0;
    stats->overhead_ns = 0;
}

/* A closure that does nothing.  Timing the shell around it measures what the
   launch boundary itself costs, so that overhead can be reported separately
   from -- or subtracted from -- real dispatch timings. */
int turing_null_closure(void *context, unsigned long long *device_ns) {
    (void)context;
    if (device_ns != 0) {
        *device_ns = 0;
    }
    return 1;
}

int turing_profiled_launch_ex(
    turing_compute_closure compute,
    void *context,
    TuringLaunchProfile *profile,
    TuringLaunchStats *stats,
    turing_launch_logger logger,
    void *logger_user,
    int language
) {
    uint64_t started;
    uint64_t finished;
    uint64_t device_ns = 0;
    uint64_t shell_ns;
    int status;

    if (compute == 0 || profile == 0) {
        return 0;
    }
    started = turing_monotonic_ns();
    status = compute(context, &device_ns);
    finished = turing_monotonic_ns();
    shell_ns = finished - started;

    profile->shell_ns = shell_ns;
    profile->device_ns = device_ns;
    profile->status = status;
    profile->language = language;

    if (stats != 0) {
        if (stats->calls == 0 && stats->shell_ns_min == 0) {
            stats->shell_ns_min = ~(unsigned long long)0;
        }
        stats->calls += 1;
        if (!status) {
            stats->failures += 1;
        }
        stats->shell_ns_total += shell_ns;
        stats->device_ns_total += device_ns;
        if (shell_ns < stats->shell_ns_min) {
            stats->shell_ns_min = shell_ns;
        }
        if (shell_ns > stats->shell_ns_max) {
            stats->shell_ns_max = shell_ns;
        }
    }
    if (logger != 0) {
        logger(logger_user, profile);
    }
    return status;
}

int turing_profiled_launch(
    turing_compute_closure compute,
    void *context,
    TuringLaunchProfile *profile
) {
    return turing_profiled_launch_ex(
        compute, context, profile, 0, 0, 0, 0
    );
}

/* Cost of the launch boundary itself, in picoseconds.

   Timing one empty call and taking the minimum does not work: the platform
   clock resolves to roughly 100ns on Windows, an empty launch is far below
   that, and every sample floors to zero.  Instead time the whole batch and
   divide, which resolves well below one clock tick.  Picoseconds are returned
   because the answer is normally a small number of nanoseconds and integer
   nanoseconds would quantise it away again. */
unsigned long long turing_measure_launch_overhead_ps(int repeats) {
    uint64_t started;
    uint64_t finished;
    uint64_t device_ns = 0;
    int index;
    volatile int sink = 0;

    if (repeats < 1) {
        repeats = 1;
    }
    /* Warm instruction cache and branch predictors first. */
    for (index = 0; index < 64; ++index) {
        sink += turing_null_closure(0, &device_ns);
    }
    started = turing_monotonic_ns();
    for (index = 0; index < repeats; ++index) {
        sink += turing_null_closure(0, &device_ns);
    }
    finished = turing_monotonic_ns();
    (void)sink;
    return ((finished - started) * UINT64_C(1000)) / (uint64_t)repeats;
}

unsigned long long turing_measure_launch_overhead(int repeats) {
    return turing_measure_launch_overhead_ps(repeats) / UINT64_C(1000);
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#if !defined(_WIN32)
#error "The dependency-free native display adapter currently requires Win32"
#else
static HWND turing_display_window = NULL;
static int turing_display_running = 1;
static uint32_t *turing_display_pixels = NULL;
static int turing_display_width = 0;
static int turing_display_height = 0;

static LRESULT CALLBACK turing_display_proc(
    HWND window, UINT message, WPARAM wparam, LPARAM lparam
) {
    (void)wparam;
    (void)lparam;
    if (message == WM_CLOSE) {
        DestroyWindow(window);
        return 0;
    }
    if (message == WM_DESTROY) {
        turing_display_running = 0;
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcA(window, message, wparam, lparam);
}

static void turing_display_set_utf8_title(HWND window, const char *title) {
    int length = MultiByteToWideChar(CP_UTF8, 0, title, -1, NULL, 0);
    wchar_t *wide;
    if (length < 1) return;
    wide = (wchar_t *)calloc((size_t)length, sizeof(wchar_t));
    if (wide == NULL) return;
    if (MultiByteToWideChar(CP_UTF8, 0, title, -1, wide, length)) {
        SetWindowTextW(window, wide);
    }
    free(wide);
}

static int turing_display_open(int width, int height, const char *title) {
    WNDCLASSA window_class = {0};
    RECT rectangle = {0, 0, width, height};
    HINSTANCE instance = GetModuleHandleA(NULL);
    window_class.lpfnWndProc = turing_display_proc;
    window_class.hInstance = instance;
    window_class.lpszClassName = "TuringNativeDisplay";
    window_class.hCursor = LoadCursor(NULL, IDC_ARROW);
    if (!RegisterClassA(&window_class) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
        return 0;
    }
    AdjustWindowRect(&rectangle, WS_OVERLAPPEDWINDOW, FALSE);
    turing_display_window = CreateWindowExA(
        0, window_class.lpszClassName, "", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT,
        rectangle.right - rectangle.left, rectangle.bottom - rectangle.top,
        NULL, NULL, instance, NULL
    );
    if (turing_display_window == NULL) return 0;
    turing_display_set_utf8_title(turing_display_window, title);
    turing_display_pixels = (uint32_t *)calloc(
        (size_t)width * (size_t)height, sizeof(uint32_t)
    );
    if (turing_display_pixels == NULL) return 0;
    turing_display_width = width;
    turing_display_height = height;
    return 1;
}

static void turing_display_messages(void) {
    MSG message;
    while (PeekMessageA(&message, NULL, 0, 0, PM_REMOVE)) {
        if (message.message == WM_QUIT) turing_display_running = 0;
        TranslateMessage(&message);
        DispatchMessageA(&message);
    }
}

static unsigned int turing_display_channel(double value) {
    if (value <= 0.0) return 0;
    if (value >= 255.0) return 255;
    return (unsigned int)(value + 0.5);
}

static void turing_display_present(
    const double *red, const double *green, const double *blue
) {
    BITMAPINFO information = {0};
    RECT client;
    HDC device;
    size_t index;
    size_t count = (size_t)turing_display_width * (size_t)turing_display_height;
    for (index = 0; index < count; ++index) {
        unsigned int r = turing_display_channel(red[index]);
        unsigned int g = turing_display_channel(green[index]);
        unsigned int b = turing_display_channel(blue[index]);
        turing_display_pixels[index] = b | (g << 8) | (r << 16);
    }
    information.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    information.bmiHeader.biWidth = turing_display_width;
    information.bmiHeader.biHeight = -turing_display_height;
    information.bmiHeader.biPlanes = 1;
    information.bmiHeader.biBitCount = 32;
    information.bmiHeader.biCompression = BI_RGB;
    GetClientRect(turing_display_window, &client);
    device = GetDC(turing_display_window);
    StretchDIBits(
        device, 0, 0, client.right, client.bottom,
        0, 0, turing_display_width, turing_display_height,
        turing_display_pixels, &information, DIB_RGB_COLORS, SRCCOPY
    );
    ReleaseDC(turing_display_window, device);
}

static void turing_display_close(void) {
    free(turing_display_pixels);
    turing_display_pixels = NULL;
    if (turing_display_window != NULL && IsWindow(turing_display_window)) {
        DestroyWindow(turing_display_window);
    }
    turing_display_window = NULL;
}
#endif

extern void columnar_multifluid_rgb_step(int32_t, int32_t, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *);

static int turing_fortran_compute(void *context, unsigned long long *device_ns) {
    void **slots = (void **)context;
    *device_ns = 0;
    columnar_multifluid_rgb_step(1, 45056, (double *)slots[0], (double *)slots[1], (double *)slots[2], (double *)slots[3], (double *)slots[4], (double *)slots[5], (double *)slots[6], (double *)slots[7], (double *)slots[8], (double *)slots[9], (double *)slots[10], (double *)slots[11], (double *)slots[12], (double *)slots[13], (double *)slots[14], (double *)slots[15], (double *)slots[16], (double *)slots[17], (double *)slots[18], (double *)slots[19], (double *)slots[20], (double *)slots[21], (double *)slots[22], (double *)slots[23], (double *)slots[24], (double *)slots[25], (double *)slots[26], (double *)slots[27], (double *)slots[28], (double *)slots[29], (double *)slots[30], (double *)slots[31], (double *)slots[32], (double *)slots[33], (double *)slots[34], (double *)slots[35], (double *)slots[36], (double *)slots[37], (double *)slots[38], (double *)slots[39], (double *)slots[40], (double *)slots[41], (double *)slots[42], (double *)slots[43], (double *)slots[44], (double *)slots[45], (double *)slots[46], (double *)slots[47], (double *)slots[48], (double *)slots[49], (double *)slots[50], (double *)slots[51], (double *)slots[52], (double *)slots[53], (double *)slots[54], (double *)slots[55], (double *)slots[56], (double *)slots[57], (double *)slots[58], (double *)slots[59], (double *)slots[60], (double *)slots[61], (double *)slots[62], (double *)slots[63], (double *)slots[64]);
    return 1;
}

int main(int argc, char **argv) {
    int frames = argc > 1 ? atoi(argv[1]) : 0;
    void *slots[65] = {0};
    TuringLaunchProfile profile = {0};
    TuringLaunchStats stats = {0};
    int frame;
    FILE *state = fopen("initial-state.bin", "rb");
    if (frames < 0) return 2;
    if (!state) { perror("initial state"); return 2; }
    slots[0] = calloc(45056, sizeof(double));
    if (!slots[0]) return 3;
    slots[1] = calloc(45056, sizeof(double));
    if (!slots[1]) return 3;
    slots[2] = calloc(45056, sizeof(double));
    if (!slots[2]) return 3;
    slots[3] = calloc(45056, sizeof(double));
    if (!slots[3]) return 3;
    slots[4] = calloc(45056, sizeof(double));
    if (!slots[4]) return 3;
    slots[5] = calloc(45056, sizeof(double));
    if (!slots[5]) return 3;
    slots[6] = calloc(45056, sizeof(double));
    if (!slots[6]) return 3;
    slots[7] = calloc(45056, sizeof(double));
    if (!slots[7]) return 3;
    slots[8] = calloc(45056, sizeof(double));
    if (!slots[8]) return 3;
    slots[9] = calloc(45056, sizeof(double));
    if (!slots[9]) return 3;
    slots[10] = calloc(45056, sizeof(double));
    if (!slots[10]) return 3;
    slots[11] = calloc(45056, sizeof(double));
    if (!slots[11]) return 3;
    slots[12] = calloc(45056, sizeof(double));
    if (!slots[12]) return 3;
    slots[13] = calloc(45056, sizeof(double));
    if (!slots[13]) return 3;
    slots[14] = calloc(45056, sizeof(double));
    if (!slots[14]) return 3;
    slots[15] = calloc(45056, sizeof(double));
    if (!slots[15]) return 3;
    slots[16] = calloc(45056, sizeof(double));
    if (!slots[16]) return 3;
    slots[17] = calloc(45056, sizeof(double));
    if (!slots[17]) return 3;
    slots[18] = calloc(45056, sizeof(double));
    if (!slots[18]) return 3;
    slots[19] = calloc(45056, sizeof(double));
    if (!slots[19]) return 3;
    slots[20] = calloc(45056, sizeof(double));
    if (!slots[20]) return 3;
    slots[21] = calloc(45056, sizeof(double));
    if (!slots[21]) return 3;
    slots[22] = calloc(45056, sizeof(double));
    if (!slots[22]) return 3;
    slots[23] = calloc(45056, sizeof(double));
    if (!slots[23]) return 3;
    slots[24] = calloc(45056, sizeof(double));
    if (!slots[24]) return 3;
    slots[25] = calloc(45056, sizeof(double));
    if (!slots[25]) return 3;
    slots[26] = calloc(45056, sizeof(double));
    if (!slots[26]) return 3;
    slots[27] = calloc(45056, sizeof(double));
    if (!slots[27]) return 3;
    slots[28] = calloc(45056, sizeof(double));
    if (!slots[28]) return 3;
    slots[29] = calloc(45056, sizeof(double));
    if (!slots[29]) return 3;
    slots[30] = calloc(45056, sizeof(double));
    if (!slots[30]) return 3;
    slots[31] = calloc(45056, sizeof(double));
    if (!slots[31]) return 3;
    slots[32] = calloc(45056, sizeof(double));
    if (!slots[32]) return 3;
    slots[33] = calloc(45056, sizeof(double));
    if (!slots[33]) return 3;
    slots[34] = calloc(45056, sizeof(double));
    if (!slots[34]) return 3;
    slots[35] = calloc(45056, sizeof(double));
    if (!slots[35]) return 3;
    slots[36] = calloc(45056, sizeof(double));
    if (!slots[36]) return 3;
    slots[37] = calloc(45056, sizeof(double));
    if (!slots[37]) return 3;
    slots[38] = calloc(45056, sizeof(double));
    if (!slots[38]) return 3;
    slots[39] = calloc(45056, sizeof(double));
    if (!slots[39]) return 3;
    slots[40] = calloc(45056, sizeof(double));
    if (!slots[40]) return 3;
    slots[41] = calloc(45056, sizeof(double));
    if (!slots[41]) return 3;
    slots[42] = calloc(45056, sizeof(double));
    if (!slots[42]) return 3;
    slots[43] = calloc(45056, sizeof(double));
    if (!slots[43]) return 3;
    slots[44] = calloc(45056, sizeof(double));
    if (!slots[44]) return 3;
    slots[45] = calloc(45056, sizeof(double));
    if (!slots[45]) return 3;
    slots[46] = calloc(45056, sizeof(double));
    if (!slots[46]) return 3;
    slots[47] = calloc(45056, sizeof(double));
    if (!slots[47]) return 3;
    slots[48] = calloc(45056, sizeof(double));
    if (!slots[48]) return 3;
    slots[49] = calloc(45056, sizeof(double));
    if (!slots[49]) return 3;
    slots[50] = calloc(45056, sizeof(double));
    if (!slots[50]) return 3;
    slots[51] = calloc(45056, sizeof(double));
    if (!slots[51]) return 3;
    slots[52] = calloc(45056, sizeof(double));
    if (!slots[52]) return 3;
    slots[53] = calloc(45056, sizeof(double));
    if (!slots[53]) return 3;
    slots[54] = calloc(45056, sizeof(double));
    if (!slots[54]) return 3;
    slots[55] = calloc(45056, sizeof(double));
    if (!slots[55]) return 3;
    slots[56] = calloc(45056, sizeof(double));
    if (!slots[56]) return 3;
    slots[57] = calloc(45056, sizeof(double));
    if (!slots[57]) return 3;
    slots[58] = calloc(45056, sizeof(double));
    if (!slots[58]) return 3;
    slots[59] = calloc(45056, sizeof(double));
    if (!slots[59]) return 3;
    slots[60] = calloc(45056, sizeof(double));
    if (!slots[60]) return 3;
    slots[61] = calloc(45056, sizeof(double));
    if (!slots[61]) return 3;
    slots[62] = calloc(45056, sizeof(double));
    if (!slots[62]) return 3;
    slots[63] = calloc(45056, sizeof(double));
    if (!slots[63]) return 3;
    slots[64] = calloc(45056, sizeof(double));
    if (!slots[64]) return 3;
    if (fread(slots[0], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at managed_time\n");
        return 4;
    }
    if (fread(slots[1], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at dt\n");
        return 4;
    }
    if (fread(slots[2], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at audio_low\n");
        return 4;
    }
    if (fread(slots[3], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at audio_high\n");
        return 4;
    }
    if (fread(slots[4], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at audio_mid\n");
        return 4;
    }
    if (fread(slots[5], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at column_x\n");
        return 4;
    }
    if (fread(slots[6], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at column_y\n");
        return 4;
    }
    if (fread(slots[7], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at ink_red\n");
        return 4;
    }
    if (fread(slots[8], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at ink_yellow\n");
        return 4;
    }
    if (fread(slots[9], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at ink_green\n");
        return 4;
    }
    if (fread(slots[10], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at ink_cyan\n");
        return 4;
    }
    if (fread(slots[11], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at ink_blue\n");
        return 4;
    }
    if (fread(slots[12], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at ink_magenta\n");
        return 4;
    }
    if (fread(slots[13], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at entity_cargo\n");
        return 4;
    }
    if (fread(slots[14], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at entity_b_cargo\n");
        return 4;
    }
    if (fread(slots[15], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at entity_c_cargo\n");
        return 4;
    }
    if (fread(slots[16], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at food_store\n");
        return 4;
    }
    if (fread(slots[17], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at entity_x\n");
        return 4;
    }
    if (fread(slots[18], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at entity_y\n");
        return 4;
    }
    if (fread(slots[19], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at entity_b_x\n");
        return 4;
    }
    if (fread(slots[20], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at entity_b_y\n");
        return 4;
    }
    if (fread(slots[21], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at entity_c_x\n");
        return 4;
    }
    if (fread(slots[22], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at entity_c_y\n");
        return 4;
    }
    if (fread(slots[23], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at entity_velocity_x\n");
        return 4;
    }
    if (fread(slots[24], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at entity_velocity_y\n");
        return 4;
    }
    if (fread(slots[25], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at entity_b_velocity_x\n");
        return 4;
    }
    if (fread(slots[26], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at entity_b_velocity_y\n");
        return 4;
    }
    if (fread(slots[27], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at entity_c_velocity_x\n");
        return 4;
    }
    if (fread(slots[28], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at entity_c_velocity_y\n");
        return 4;
    }
    if (fread(slots[29], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at audio_level\n");
        return 4;
    }
    if (fread(slots[30], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at nest_food\n");
        return 4;
    }
    if (fread(slots[31], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at filter_reservoir\n");
        return 4;
    }
    if (fread(slots[32], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at displacement\n");
        return 4;
    }
    if (fread(slots[33], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at displacement_velocity\n");
        return 4;
    }
    if (fread(slots[34], sizeof(double), 45056, state) != 45056) {
        fprintf(stderr, "short initial state at rest_surface\n");
        return 4;
    }
    fclose(state);
    if (!turing_display_open(256, 176, "Managed Columnar Multifluid World \u2014 native Fortran")) return 7;
    turing_launch_stats_reset(&stats);
    for (frame = 0; turing_display_running && (frames == 0 || frame < frames); ++frame) {
        turing_display_messages();
        if (!turing_display_running) break;
        if (turing_profiled_launch_ex(turing_fortran_compute, slots,
                &profile, &stats, NULL, NULL, 3) != 1) return 5;
        memcpy(slots[32], slots[38], 45056 * sizeof(double));
        memcpy(slots[33], slots[39], 45056 * sizeof(double));
        memcpy(slots[17], slots[40], 45056 * sizeof(double));
        memcpy(slots[18], slots[41], 45056 * sizeof(double));
        memcpy(slots[23], slots[42], 45056 * sizeof(double));
        memcpy(slots[24], slots[43], 45056 * sizeof(double));
        memcpy(slots[19], slots[44], 45056 * sizeof(double));
        memcpy(slots[20], slots[45], 45056 * sizeof(double));
        memcpy(slots[25], slots[46], 45056 * sizeof(double));
        memcpy(slots[26], slots[47], 45056 * sizeof(double));
        memcpy(slots[21], slots[48], 45056 * sizeof(double));
        memcpy(slots[22], slots[49], 45056 * sizeof(double));
        memcpy(slots[27], slots[50], 45056 * sizeof(double));
        memcpy(slots[28], slots[51], 45056 * sizeof(double));
        memcpy(slots[13], slots[52], 45056 * sizeof(double));
        memcpy(slots[14], slots[53], 45056 * sizeof(double));
        memcpy(slots[15], slots[54], 45056 * sizeof(double));
        memcpy(slots[16], slots[55], 45056 * sizeof(double));
        memcpy(slots[30], slots[56], 45056 * sizeof(double));
        memcpy(slots[31], slots[57], 45056 * sizeof(double));
        memcpy(slots[0], slots[58], 45056 * sizeof(double));
        memcpy(slots[7], slots[59], 45056 * sizeof(double));
        memcpy(slots[8], slots[60], 45056 * sizeof(double));
        memcpy(slots[9], slots[61], 45056 * sizeof(double));
        memcpy(slots[10], slots[62], 45056 * sizeof(double));
        memcpy(slots[11], slots[63], 45056 * sizeof(double));
        memcpy(slots[12], slots[64], 45056 * sizeof(double));
        turing_display_present(
            (const double *)slots[35],
            (const double *)slots[36],
            (const double *)slots[37]);
        turing_display_messages();
    }
    turing_display_close();
    printf("{\"status\":%d,\"frames\":%d,\"shell_ns_total\":%llu,\"outputs\":{",
           profile.status, frame, stats.shell_ns_total);
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[35])[i];
      printf("\"red\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[35])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[36])[i];
      printf(",\"green\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[36])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[37])[i];
      printf(",\"blue\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[37])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[38])[i];
      printf(",\"next_displacement\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[38])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[39])[i];
      printf(",\"next_velocity\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[39])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[40])[i];
      printf(",\"next_entity_x\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[40])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[41])[i];
      printf(",\"next_entity_y\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[41])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[42])[i];
      printf(",\"next_entity_velocity_x\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[42])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[43])[i];
      printf(",\"next_entity_velocity_y\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[43])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[44])[i];
      printf(",\"next_entity_b_x\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[44])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[45])[i];
      printf(",\"next_entity_b_y\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[45])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[46])[i];
      printf(",\"next_entity_b_velocity_x\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[46])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[47])[i];
      printf(",\"next_entity_b_velocity_y\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[47])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[48])[i];
      printf(",\"next_entity_c_x\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[48])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[49])[i];
      printf(",\"next_entity_c_y\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[49])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[50])[i];
      printf(",\"next_entity_c_velocity_x\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[50])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[51])[i];
      printf(",\"next_entity_c_velocity_y\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[51])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[52])[i];
      printf(",\"next_entity_cargo\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[52])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[53])[i];
      printf(",\"next_entity_b_cargo\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[53])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[54])[i];
      printf(",\"next_entity_c_cargo\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[54])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[55])[i];
      printf(",\"next_food_store\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[55])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[56])[i];
      printf(",\"next_nest_food\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[56])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[57])[i];
      printf(",\"next_filter_reservoir\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[57])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[58])[i];
      printf(",\"next_time\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[58])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[59])[i];
      printf(",\"next_ink_red\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[59])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[60])[i];
      printf(",\"next_ink_yellow\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[60])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[61])[i];
      printf(",\"next_ink_green\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[61])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[62])[i];
      printf(",\"next_ink_cyan\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[62])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[63])[i];
      printf(",\"next_ink_blue\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[63])[0], sum); }
    { double sum = 0.0; size_t i;
      for (i = 0; i < 45056; ++i) sum += ((double *)slots[64])[i];
      printf(",\"next_ink_magenta\":{\"first\":%.17g,\"sum\":%.17g}",
             (double)((double *)slots[64])[0], sum); }
    printf("}}\n");
    { FILE *outputs_file = fopen("final-outputs.bin", "wb");
      if (!outputs_file) { perror("final outputs"); return 6; }
    fwrite(slots[35], sizeof(double), 45056, outputs_file);
    fwrite(slots[36], sizeof(double), 45056, outputs_file);
    fwrite(slots[37], sizeof(double), 45056, outputs_file);
    fwrite(slots[38], sizeof(double), 45056, outputs_file);
    fwrite(slots[39], sizeof(double), 45056, outputs_file);
    fwrite(slots[40], sizeof(double), 45056, outputs_file);
    fwrite(slots[41], sizeof(double), 45056, outputs_file);
    fwrite(slots[42], sizeof(double), 45056, outputs_file);
    fwrite(slots[43], sizeof(double), 45056, outputs_file);
    fwrite(slots[44], sizeof(double), 45056, outputs_file);
    fwrite(slots[45], sizeof(double), 45056, outputs_file);
    fwrite(slots[46], sizeof(double), 45056, outputs_file);
    fwrite(slots[47], sizeof(double), 45056, outputs_file);
    fwrite(slots[48], sizeof(double), 45056, outputs_file);
    fwrite(slots[49], sizeof(double), 45056, outputs_file);
    fwrite(slots[50], sizeof(double), 45056, outputs_file);
    fwrite(slots[51], sizeof(double), 45056, outputs_file);
    fwrite(slots[52], sizeof(double), 45056, outputs_file);
    fwrite(slots[53], sizeof(double), 45056, outputs_file);
    fwrite(slots[54], sizeof(double), 45056, outputs_file);
    fwrite(slots[55], sizeof(double), 45056, outputs_file);
    fwrite(slots[56], sizeof(double), 45056, outputs_file);
    fwrite(slots[57], sizeof(double), 45056, outputs_file);
    fwrite(slots[58], sizeof(double), 45056, outputs_file);
    fwrite(slots[59], sizeof(double), 45056, outputs_file);
    fwrite(slots[60], sizeof(double), 45056, outputs_file);
    fwrite(slots[61], sizeof(double), 45056, outputs_file);
    fwrite(slots[62], sizeof(double), 45056, outputs_file);
    fwrite(slots[63], sizeof(double), 45056, outputs_file);
    fwrite(slots[64], sizeof(double), 45056, outputs_file);
      fclose(outputs_file); }
    for (frame = 0; frame < 65; ++frame) free(slots[frame]);
    return 0;
}
