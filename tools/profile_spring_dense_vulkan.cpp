// Stream full-size sheets through the Turing spring law's existing f64 SPIR-V.
// The workload is generated per batch; it never claims that all fabric state
// resides in VRAM, system RAM, or this machine's small free disk space.
#include "kernel_vulkan_runner.h"
#include "mem_backend.h"
#include "vulkan_state_bank.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {
using Handle = std::unique_ptr<void, decltype(&gp_mem_backend_release)>;
constexpr std::uint64_t kBindings = 26;
constexpr std::uint64_t kFeeds = 24;
constexpr std::uint64_t kPortableGroupLimit = 65535;

double feed_value(int feed, std::uint64_t global_lane,
                  std::uint64_t width, std::uint64_t height) {
    const auto nodes = width * height;
    const auto fabric = global_lane / (nodes * 3);
    const auto node = (global_lane / 3) % nodes;
    const auto axis = global_lane % 3;
    const auto x = node % width, y = node / width;
    const double px = static_cast<double>(x) * .01;
    const double py = static_cast<double>(height - y) * .01;
    const double position = axis == 0 ? px : axis == 1 ? py : 0.0;
    const double velocity = axis == 0 ? static_cast<double>(fabric % 31) * .0001 : 0.0;
    switch (feed) {
    case 0: return position; // anchor
    case 1: case 2: return 0.0; // contact
    case 3: return .05; // drag
    case 4: return .001; // dt
    case 5: return x > 0 ? 1.0 : 0.0;
    case 6: return x + 1 < width ? 1.0 : 0.0;
    case 7: return y > 0 ? 1.0 : 0.0;
    case 8: return y + 1 < height ? 1.0 : 0.0;
    case 9: return axis == 1 ? -9.8 : 0.0; // acceleration
    case 10: return 1.0; // inverse mass
    case 11: return position - (axis == 0 && x > 0 ? .01 : 0.0);
    case 12: return position + (axis == 0 && x + 1 < width ? .01 : 0.0);
    case 13: return position + (axis == 1 && y > 0 ? .01 : 0.0);
    case 14: return position - (axis == 1 && y + 1 < height ? .01 : 0.0);
    case 15: return 1.0; // occupied
    case 16: return y == 0 ? 1.0 : 0.0; // attached top
    case 17: return position;
    case 18: return axis == 0 ? -.01 : 0.0;
    case 19: return axis == 0 ? .01 : 0.0;
    case 20: return axis == 1 ? .01 : 0.0;
    case 21: return axis == 1 ? -.01 : 0.0;
    case 22: return 10.0; // stiffness
    case 23: return velocity;
    default: return 0.0;
    }
}
}

int main(int argc, char** argv) {
    if (argc < 6 || argc > 8) {
        std::cerr << "usage: profile_spring_dense_vulkan SPV fabrics width height requested_batch [local_size] [device_index]\n";
        return 2;
    }
    const std::uint64_t fabrics = std::stoull(argv[2]);
    const std::uint64_t width = std::stoull(argv[3]);
    const std::uint64_t height = std::stoull(argv[4]);
    const std::uint64_t requested_batch = std::stoull(argv[5]);
    const std::uint64_t local_size = argc >= 7 ? std::stoull(argv[6]) : 64;
    const auto device_index = argc == 8 ? static_cast<std::uint32_t>(std::stoul(argv[7]))
                                        : UINT32_MAX;
    if (!fabrics || !width || !height || !requested_batch ||
        !local_size || local_size > 1024 ||
        width > UINT32_MAX / height || width * height > UINT32_MAX / 3) return 2;
    std::ifstream file(argv[1], std::ios::binary);
    std::vector<char> bytes{std::istreambuf_iterator<char>(file), {}};
    if (bytes.empty() || bytes.size() % 4) return 2;
    std::vector<std::uint32_t> shader(bytes.size() / 4);
    std::memcpy(shader.data(), bytes.data(), bytes.size());

    gp_mem_backend_vulkan_params_t params{device_index, GP_VULKAN_ALLOC_DEVICE_LOCAL};
    nodus::kernels::VulkanDeviceInfo device;
    std::string error;
    Handle context(nodus::kernels::create_vulkan_backend(&params, &device, &error),
                   &gp_mem_backend_release);
    if (!context) { std::cerr << error << '\n'; return 3; }
    const auto* vtable = gp_mem_backend_get_vtable(context.get());
    nodus::kernels::VulkanBackendStatus status;
    if (!vtable || !vtable->alloc ||
        !nodus::kernels::query_vulkan_backend_status(context.get(), &status, &error)) return 3;

    const std::uint64_t lanes_per_fabric = width * height * 3;
    const std::uint64_t max_group_lanes = kPortableGroupLimit * local_size;
    // The device-local budget is shared with other server work. Use at most
    // half of its reported free headroom for this one streamed batch.
    std::uint64_t available = 0;
    if (status.budget_available)
        for (const auto& heap : status.heaps)
            if (heap.device_local && heap.budget_bytes > heap.usage_bytes)
                available = std::max(available, heap.budget_bytes - heap.usage_bytes);
    else
        for (const auto& heap : status.heaps)
            if (heap.device_local) available = std::max(available, heap.size_bytes / 2);
    if (!available) return 4;
    const std::uint64_t bytes_per_fabric = lanes_per_fabric * sizeof(double) * kBindings;
    const std::uint64_t half_budget = available / 2;
    const std::uint64_t max_budget_lanes = half_budget > 1024 * 1024
        ? (half_budget - 1024 * 1024) / (sizeof(double) * kBindings) : 0;
    // A low-VRAM device may process a proper subset of one fabric. Its
    // neighbor inputs are gathered from the previous state before dispatch.
    const std::uint64_t lanes = std::min({fabrics * lanes_per_fabric,
        requested_batch * lanes_per_fabric, max_group_lanes,
        max_budget_lanes}) / 3 * 3;
    if (!lanes) { std::cerr << "device budget cannot hold one XYZ lane\n"; return 4; }
    const std::uint64_t column_bytes = lanes * sizeof(double);
    const auto bank_budget = std::min<std::uint64_t>(
        available / 2, 26 * column_bytes + 1024 * 1024);
    nodus::gpu::VulkanStateBank bank(context.get(), bank_budget);
    std::vector<double> feed(lanes);
    std::uint64_t complete_lanes = 0, dispatches = 0;
    std::uint64_t round = 0;
    const auto started = std::chrono::steady_clock::now();
    const std::uint64_t total_lanes = fabrics * lanes_per_fabric;
    while (complete_lanes < total_lanes) {
        const auto active_lanes = std::min(lanes, total_lanes - complete_lanes);
        std::vector<nodus::gpu::VulkanStateBank::Lease> leases;
        std::vector<nodus::kernels::VulkanBackendBinding> bindings;
        leases.reserve(27);
        bindings.reserve(27);
        for (std::uint64_t i = 0; i < 27; ++i) {
            const auto address = (++round) * 32 + i + 1;
            const auto size = i == 26 ? 4 : column_bytes;
            if (!bank.add(address, size)) return 5;
            leases.emplace_back(bank.lease(address));
            if (!leases.back()) return 5;
            bindings.push_back({leases.back().handle(), 0, size});
        }
        for (int i = 0; i < static_cast<int>(kFeeds); ++i) {
            for (std::uint64_t lane = 0; lane < active_lanes; ++lane)
                feed[lane] = feed_value(i, complete_lanes + lane, width, height);
            if (!vtable->copy_to_backend(leases[i].handle(), 0, feed.data(),
                                         active_lanes * sizeof(double))) return 6;
        }
        const auto count = static_cast<std::uint32_t>(active_lanes);
        if (!vtable->copy_to_backend(leases[26].handle(), 0, &count, 4) ||
            !nodus::kernels::dispatch_spirv_backends(
                shader, "main", bindings,
                static_cast<std::uint32_t>((active_lanes + local_size - 1) / local_size),
                &error)) { std::cerr << error << '\n'; return 7; }
        double result = 0;
        if (!vtable->copy_from_backend(leases[24].handle(),
                (active_lanes - 1) * sizeof(double), &result, sizeof(result)) ||
            !std::isfinite(result)) return 8;
        complete_lanes += active_lanes;
        ++dispatches;
    }
    const auto seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    const auto stats = bank.stats();
    std::cout << "device=" << device.name << " grid=" << width << 'x' << height
              << " fabrics=" << fabrics << " dispatches=" << dispatches
              << " fabric_equivalents_per_dispatch="
              << static_cast<double>(lanes) / lanes_per_fabric
              << " local_size=" << local_size
              << " lanes_per_dispatch=" << lanes << " seconds=" << seconds
              << " fabrics_per_second=" << fabrics / seconds
              << " bytes_per_fabric=" << bytes_per_fabric
              << " free_device_budget_bytes=" << available
              << " resident_bytes=" << stats.resident_bytes
              << " bank_budget_bytes=" << stats.budget_bytes
              << " evicted=" << stats.evicted
              << " reused=" << stats.reused << '\n';
    return 0;
}
