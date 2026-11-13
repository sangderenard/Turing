// Exercise a Turing shader product through Nodus's existing Vulkan runner.
// Usage: nodus_spring_shader_probe module.spv gx gy gz buffer0 ... bufferN
#include "kernel_vulkan_runner.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 7) return 2;
  std::ifstream source(argv[1], std::ios::binary);
  if (!source) return 3;
  std::vector<char> bytes((std::istreambuf_iterator<char>(source)), {});
  if (bytes.empty() || bytes.size() % sizeof(std::uint32_t)) return 4;
  std::vector<std::uint32_t> spirv(bytes.size() / sizeof(std::uint32_t));
  std::memcpy(spirv.data(), bytes.data(), bytes.size());
  const auto groups = static_cast<std::uint32_t>(std::stoul(argv[2]));
  if (std::stoul(argv[3]) != 1 || std::stoul(argv[4]) != 1) return 10;
  std::vector<std::vector<char>> payloads;
  std::vector<nodus::kernels::VulkanBufferBinding> bindings;
  payloads.reserve(argc - 5);
  bindings.reserve(argc - 5);
  for (int i = 5; i < argc; ++i) {
    std::ifstream input(argv[i], std::ios::binary);
    if (!input) return 5;
    payloads.emplace_back(std::istreambuf_iterator<char>(input),
                          std::istreambuf_iterator<char>());
    auto& payload = payloads.back();
    if (payload.empty()) return 6;
    bindings.push_back({payload.data(), payload.size(), true});
  }
  std::string error;
  if (!nodus::kernels::dispatch_spirv(spirv, "main", bindings, groups, &error)) {
    std::cerr << error << '\n';
    return 7;
  }
  for (int i = 5; i < argc; ++i) {
    std::ofstream output(argv[i], std::ios::binary | std::ios::trunc);
    if (!output) return 8;
    const auto& payload = payloads[i - 5];
    output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    if (!output) return 9;
  }
  return 0;
}
