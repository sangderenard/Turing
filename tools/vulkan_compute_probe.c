/* Minimal Vulkan compute probe for Turing's deployable-shader products.
 *
 * Runs one GLCompute SPIR-V module whose resources are storage buffers in
 * descriptor set 0 at contiguous bindings, which is the descriptor ABI
 * src/compiler/ssa_spirv_backend.py:emit_deployable_compute_shader declares
 * and the product's shader_deployment record publishes. It exists so tests
 * can compare a deployable product against Python on any Vulkan driver
 * (a CPU driver such as lavapipe is enough). It is a verification
 * instrument, not a runtime: no consumer types or handles appear here.
 *
 * usage: vulkan_compute_probe module.spv gx gy gz buffer0 buffer1 ...
 * Each buffer file is loaded into the storage buffer at binding i before the
 * dispatch and rewritten with that buffer's contents after it.
 */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed %d\n", #x, r_); return 2; } } while (0)
static void *slurp(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); *n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    void *b = malloc(*n ? *n : 1); if (fread(b, 1, *n, f) != *n) { fclose(f); return NULL; }
    fclose(f); return b;
}
int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage\n"); return 1; }
    size_t code_n; void *code = slurp(argv[1], &code_n); if (!code) return 1;
    uint32_t g[3] = {(uint32_t)atoi(argv[2]), (uint32_t)atoi(argv[3]), (uint32_t)atoi(argv[4])};
    int nb = argc - 5;
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "turing", 1, "turing", 1, VK_API_VERSION_1_1};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
    VkInstance inst; CK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t pc = 1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(inst, &pc, &pd); if (!pc) return 3;
    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, NULL);
    VkQueueFamilyProperties qp[16]; if (qn > 16) qn = 16; vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qp);
    uint32_t qf = 0; while (qf < qn && !(qp[qf].queueFlags & VK_QUEUE_COMPUTE_BIT)) qf++;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, qf, 1, &prio};
    VkPhysicalDeviceFeatures feat; vkGetPhysicalDeviceFeatures(pd, &feat);
    VkPhysicalDeviceFeatures want = {0}; want.shaderFloat64 = feat.shaderFloat64; want.shaderInt64 = feat.shaderInt64;
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &qci, 0, NULL, 0, NULL, &want};
    VkDevice dev; CK(vkCreateDevice(pd, &dci, NULL, &dev));
    VkQueue q; vkGetDeviceQueue(dev, qf, 0, &q);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    VkBuffer *bufs = calloc(nb, sizeof *bufs); VkDeviceMemory *mems = calloc(nb, sizeof *mems); size_t *sizes = calloc(nb, sizeof *sizes);
    for (int i = 0; i < nb; ++i) {
        void *data = slurp(argv[5 + i], &sizes[i]); if (!data) return 4;
        VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, sizes[i] ? sizes[i] : 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE};
        CK(vkCreateBuffer(dev, &bci, NULL, &bufs[i]));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, bufs[i], &mr);
        uint32_t t = 0; for (; t < mp.memoryTypeCount; ++t) if ((mr.memoryTypeBits & (1u << t)) && (mp.memoryTypes[t].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) break;
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, t};
        CK(vkAllocateMemory(dev, &mai, NULL, &mems[i])); CK(vkBindBufferMemory(dev, bufs[i], mems[i], 0));
        void *m; CK(vkMapMemory(dev, mems[i], 0, VK_WHOLE_SIZE, 0, &m)); memcpy(m, data, sizes[i]); vkUnmapMemory(dev, mems[i]); free(data);
    }
    VkDescriptorSetLayoutBinding *lb = calloc(nb, sizeof *lb);
    for (int i = 0; i < nb; ++i) { lb[i].binding = i; lb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; lb[i].descriptorCount = 1; lb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo lci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, nb, lb};
    VkDescriptorSetLayout dsl; CK(vkCreateDescriptorSetLayout(dev, &lci, NULL, &dsl));
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl};
    VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, code_n, code};
    VkShaderModule sm; CK(vkCreateShaderModule(dev, &smci, NULL, &sm));
    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL}, pl};
    VkPipeline pipe; CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nb};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 1, &ps};
    VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 1, &dsl};
    VkDescriptorSet ds; CK(vkAllocateDescriptorSets(dev, &dsai, &ds));
    for (int i = 0; i < nb; ++i) {
        VkDescriptorBufferInfo bi = {bufs[i], 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi};
        vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);
    }
    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, qf};
    VkCommandPool cp; CK(vkCreateCommandPool(dev, &cpi, NULL, &cp));
    VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &cbai, &cb));
    VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CK(vkBeginCommandBuffer(cb, &cbbi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cb, g[0], g[1], g[2]);
    CK(vkEndCommandBuffer(cb));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb};
    CK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE)); CK(vkQueueWaitIdle(q));
    for (int i = 0; i < nb; ++i) {
        void *m; CK(vkMapMemory(dev, mems[i], 0, VK_WHOLE_SIZE, 0, &m));
        FILE *f = fopen(argv[5 + i], "wb"); fwrite(m, 1, sizes[i], f); fclose(f); vkUnmapMemory(dev, mems[i]);
    }
    return 0;
}
