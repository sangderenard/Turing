#include <android/input.h>
#include <android/log.h>
#include <android/native_window.h>
#include <android_native_app_glue.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "generated/mobile_shaders.h"
#include "generated/turing_program_abi.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "TuringMobile", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "TuringMobile", __VA_ARGS__)
#define VK_OK(call) do { VkResult _r = (call); if (_r != VK_SUCCESS) { LOGE("%s failed: %d", #call, int(_r)); return false; } } while (0)

namespace {

struct PushColor { float color[4]; };

class VulkanShell {
public:
    bool start(ANativeWindow* window) {
        window_ = window;
        if (!create_device()) return false;
        return create_swapchain();
    }

    void stop() {
        if (device_) vkDeviceWaitIdle(device_);
        destroy_swapchain();
        if (device_) vkDestroyDevice(device_, nullptr);
        if (surface_) vkDestroySurfaceKHR(instance_, surface_, nullptr);
        if (instance_) vkDestroyInstance(instance_, nullptr);
        *this = VulkanShell{};
    }

    bool draw(float phase, float touch_x, float touch_y) {
        if (!device_ || !swapchain_) return false;
        uint32_t image = 0;
        VkResult acquired = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
                                                   image_ready_, VK_NULL_HANDLE, &image);
        if (acquired == VK_ERROR_OUT_OF_DATE_KHR) return rebuild();
        if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR) return false;

        vkWaitForFences(device_, 1, &frame_fence_, VK_TRUE, UINT64_MAX);
        vkResetFences(device_, 1, &frame_fence_);
        vkResetCommandBuffer(command_, 0);

        TuringMobileColor computed = turing_mobile_eval(phase, touch_x, touch_y);
        PushColor push{{float(computed.red), float(computed.green), float(computed.blue), 1.0f}};
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_OK(vkBeginCommandBuffer(command_, &begin));
        VkClearValue clear{};
        VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        pass.renderPass = render_pass_;
        pass.framebuffer = framebuffers_[image];
        pass.renderArea.extent = extent_;
        pass.clearValueCount = 1;
        pass.pClearValues = &clear;
        vkCmdBeginRenderPass(command_, &pass, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(command_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        vkCmdPushConstants(command_, pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(push), &push);
        vkCmdDraw(command_, 3, 1, 0, 0);
        vkCmdEndRenderPass(command_);
        VK_OK(vkEndCommandBuffer(command_));

        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &image_ready_;
        submit.pWaitDstStageMask = &wait_stage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &render_done_;
        VK_OK(vkQueueSubmit(queue_, 1, &submit, frame_fence_));

        VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &render_done_;
        present.swapchainCount = 1;
        present.pSwapchains = &swapchain_;
        present.pImageIndices = &image;
        VkResult shown = vkQueuePresentKHR(queue_, &present);
        if (shown == VK_ERROR_OUT_OF_DATE_KHR || shown == VK_SUBOPTIMAL_KHR) return rebuild();
        return shown == VK_SUCCESS;
    }

private:
    bool create_device() {
        const char* instance_extensions[] = {"VK_KHR_surface", "VK_KHR_android_surface"};
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "Turing Mobile";
        app.apiVersion = VK_API_VERSION_1_0;
        VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        create.pApplicationInfo = &app;
        create.enabledExtensionCount = 2;
        create.ppEnabledExtensionNames = instance_extensions;
        VK_OK(vkCreateInstance(&create, nullptr, &instance_));

        VkAndroidSurfaceCreateInfoKHR surface_info{VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR};
        surface_info.window = window_;
        VK_OK(vkCreateAndroidSurfaceKHR(instance_, &surface_info, nullptr, &surface_));

        uint32_t physical_count = 0;
        VK_OK(vkEnumeratePhysicalDevices(instance_, &physical_count, nullptr));
        std::vector<VkPhysicalDevice> physicals(physical_count);
        VK_OK(vkEnumeratePhysicalDevices(instance_, &physical_count, physicals.data()));
        for (VkPhysicalDevice candidate : physicals) {
            uint32_t family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, nullptr);
            std::vector<VkQueueFamilyProperties> families(family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, families.data());
            for (uint32_t i = 0; i < family_count; ++i) {
                VkBool32 present = VK_FALSE;
                vkGetPhysicalDeviceSurfaceSupportKHR(candidate, i, surface_, &present);
                if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
                    physical_ = candidate;
                    queue_family_ = i;
                    break;
                }
            }
            if (physical_) break;
        }
        if (!physical_) { LOGE("No Vulkan graphics/present queue"); return false; }

        float priority = 1.0f;
        VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue.queueFamilyIndex = queue_family_;
        queue.queueCount = 1;
        queue.pQueuePriorities = &priority;
        const char* device_extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        device.queueCreateInfoCount = 1;
        device.pQueueCreateInfos = &queue;
        device.enabledExtensionCount = 1;
        device.ppEnabledExtensionNames = device_extensions;
        VK_OK(vkCreateDevice(physical_, &device, nullptr, &device_));
        vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool.queueFamilyIndex = queue_family_;
        VK_OK(vkCreateCommandPool(device_, &pool, nullptr, &command_pool_));
        VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        alloc.commandPool = command_pool_;
        alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = 1;
        VK_OK(vkAllocateCommandBuffers(device_, &alloc, &command_));
        VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_OK(vkCreateSemaphore(device_, &semaphore, nullptr, &image_ready_));
        VK_OK(vkCreateSemaphore(device_, &semaphore, nullptr, &render_done_));
        VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_OK(vkCreateFence(device_, &fence, nullptr, &frame_fence_));
        return true;
    }

    bool create_swapchain() {
        VkSurfaceCapabilitiesKHR capabilities{};
        VK_OK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_, surface_, &capabilities));
        uint32_t format_count = 0;
        VK_OK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &format_count, nullptr));
        std::vector<VkSurfaceFormatKHR> formats(format_count);
        VK_OK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &format_count, formats.data()));
        VkSurfaceFormatKHR selected = formats.front();
        for (const auto& candidate : formats) {
            if (candidate.format == VK_FORMAT_R8G8B8A8_UNORM ||
                candidate.format == VK_FORMAT_B8G8R8A8_UNORM) selected = candidate;
        }
        format_ = selected.format;
        extent_ = capabilities.currentExtent;
        if (extent_.width == UINT32_MAX) {
            extent_.width = uint32_t(ANativeWindow_getWidth(window_));
            extent_.height = uint32_t(ANativeWindow_getHeight(window_));
        }
        uint32_t image_count = std::min(capabilities.minImageCount + 1,
            capabilities.maxImageCount ? capabilities.maxImageCount : UINT32_MAX);
        VkSwapchainCreateInfoKHR swap{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        swap.surface = surface_;
        swap.minImageCount = image_count;
        swap.imageFormat = selected.format;
        swap.imageColorSpace = selected.colorSpace;
        swap.imageExtent = extent_;
        swap.imageArrayLayers = 1;
        swap.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        swap.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        swap.preTransform = capabilities.currentTransform;
        swap.compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
        if (!(capabilities.supportedCompositeAlpha & swap.compositeAlpha))
            swap.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        swap.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        swap.clipped = VK_TRUE;
        VK_OK(vkCreateSwapchainKHR(device_, &swap, nullptr, &swapchain_));

        VK_OK(vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, nullptr));
        images_.resize(image_count);
        VK_OK(vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, images_.data()));
        views_.resize(image_count);
        for (size_t i = 0; i < images_.size(); ++i) {
            VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            view.image = images_[i];
            view.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view.format = format_;
            view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            view.subresourceRange.levelCount = 1;
            view.subresourceRange.layerCount = 1;
            VK_OK(vkCreateImageView(device_, &view, nullptr, &views_[i]));
        }

        VkAttachmentDescription attachment{};
        attachment.format = format_;
        attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentReference color{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color;
        VkRenderPassCreateInfo render{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        render.attachmentCount = 1;
        render.pAttachments = &attachment;
        render.subpassCount = 1;
        render.pSubpasses = &subpass;
        VK_OK(vkCreateRenderPass(device_, &render, nullptr, &render_pass_));

        auto shader = [this](const uint32_t* code, size_t bytes) {
            VkShaderModule module{};
            VkShaderModuleCreateInfo create{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            create.codeSize = bytes;
            create.pCode = code;
            vkCreateShaderModule(device_, &create, nullptr, &module);
            return module;
        };
        VkShaderModule vertex = shader(kFullscreenVert, kFullscreenVertSize);
        VkShaderModule fragment = shader(kMainOutputFrag, kMainOutputFragSize);
        if (!vertex || !fragment) return false;
        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        push.size = sizeof(PushColor);
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.pushConstantRangeCount = 1;
        layout.pPushConstantRanges = &push;
        VK_OK(vkCreatePipelineLayout(device_, &layout, nullptr, &pipeline_layout_));
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_VERTEX_BIT, vertex, "main", nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_FRAGMENT_BIT, fragment, "main", nullptr};
        VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkViewport viewport{0, 0, float(extent_.width), float(extent_.height), 0, 1};
        VkRect2D scissor{{0, 0}, extent_};
        VkPipelineViewportStateCreateInfo viewport_state{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport_state.viewportCount = 1; viewport_state.pViewports = &viewport;
        viewport_state.scissorCount = 1; viewport_state.pScissors = &scissor;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo samples{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        samples.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState blend_attachment{};
        blend_attachment.colorWriteMask = 0xf;
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = 1; blend.pAttachments = &blend_attachment;
        VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipeline.stageCount = 2; pipeline.pStages = stages;
        pipeline.pVertexInputState = &vertex_input;
        pipeline.pInputAssemblyState = &assembly;
        pipeline.pViewportState = &viewport_state;
        pipeline.pRasterizationState = &raster;
        pipeline.pMultisampleState = &samples;
        pipeline.pColorBlendState = &blend;
        pipeline.layout = pipeline_layout_;
        pipeline.renderPass = render_pass_;
        VK_OK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeline, nullptr, &pipeline_));
        vkDestroyShaderModule(device_, vertex, nullptr);
        vkDestroyShaderModule(device_, fragment, nullptr);

        framebuffers_.resize(views_.size());
        for (size_t i = 0; i < views_.size(); ++i) {
            VkFramebufferCreateInfo framebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            framebuffer.renderPass = render_pass_;
            framebuffer.attachmentCount = 1;
            framebuffer.pAttachments = &views_[i];
            framebuffer.width = extent_.width;
            framebuffer.height = extent_.height;
            framebuffer.layers = 1;
            VK_OK(vkCreateFramebuffer(device_, &framebuffer, nullptr, &framebuffers_[i]));
        }
        return true;
    }

    void destroy_swapchain() {
        for (auto framebuffer : framebuffers_) vkDestroyFramebuffer(device_, framebuffer, nullptr);
        framebuffers_.clear();
        if (pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr);
        if (pipeline_layout_) vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        if (render_pass_) vkDestroyRenderPass(device_, render_pass_, nullptr);
        for (auto view : views_) vkDestroyImageView(device_, view, nullptr);
        views_.clear(); images_.clear();
        if (swapchain_) vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        pipeline_ = VK_NULL_HANDLE; pipeline_layout_ = VK_NULL_HANDLE;
        render_pass_ = VK_NULL_HANDLE; swapchain_ = VK_NULL_HANDLE;
    }

    bool rebuild() {
        vkDeviceWaitIdle(device_);
        destroy_swapchain();
        return create_swapchain();
    }

    ANativeWindow* window_{};
    VkInstance instance_{};
    VkSurfaceKHR surface_{};
    VkPhysicalDevice physical_{};
    VkDevice device_{};
    VkQueue queue_{};
    uint32_t queue_family_{};
    VkSwapchainKHR swapchain_{};
    VkFormat format_{};
    VkExtent2D extent_{};
    std::vector<VkImage> images_;
    std::vector<VkImageView> views_;
    VkRenderPass render_pass_{};
    VkPipelineLayout pipeline_layout_{};
    VkPipeline pipeline_{};
    std::vector<VkFramebuffer> framebuffers_;
    VkCommandPool command_pool_{};
    VkCommandBuffer command_{};
    VkSemaphore image_ready_{};
    VkSemaphore render_done_{};
    VkFence frame_fence_{};
};

struct AppState {
    VulkanShell shell;
    bool ready = false;
    float touch_x = 0.5f;
    float touch_y = 0.5f;
    std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
};

void handle_command(android_app* app, int32_t command) {
    auto* state = static_cast<AppState*>(app->userData);
    if (command == APP_CMD_INIT_WINDOW && app->window) {
        state->ready = state->shell.start(app->window);
        LOGI("Vulkan shell %s", state->ready ? "ready" : "failed");
    } else if (command == APP_CMD_TERM_WINDOW) {
        state->shell.stop();
        state->ready = false;
    }
}

int32_t handle_input(android_app* app, AInputEvent* event) {
    if (AInputEvent_getType(event) != AINPUT_EVENT_TYPE_MOTION) return 0;
    auto* state = static_cast<AppState*>(app->userData);
    state->touch_x = std::clamp(AMotionEvent_getX(event, 0) /
                                float(std::max(1, ANativeWindow_getWidth(app->window))), 0.0f, 1.0f);
    state->touch_y = std::clamp(AMotionEvent_getY(event, 0) /
                                float(std::max(1, ANativeWindow_getHeight(app->window))), 0.0f, 1.0f);
    return 1;
}

}  // namespace

void android_main(android_app* app) {
    AppState state;
    app->userData = &state;
    app->onAppCmd = handle_command;
    app->onInputEvent = handle_input;
    while (!app->destroyRequested) {
        int events = 0;
        android_poll_source* source = nullptr;
        while (ALooper_pollOnce(state.ready ? 0 : -1, nullptr, &events,
                                reinterpret_cast<void**>(&source)) >= 0) {
            if (source) source->process(app, source);
            if (app->destroyRequested) break;
        }
        if (state.ready) {
            float seconds = std::chrono::duration<float>(
                std::chrono::steady_clock::now() - state.started).count();
            float phase = 0.5f + 0.5f * std::sin(seconds * 0.7f);
            if (!state.shell.draw(phase, state.touch_x, state.touch_y)) LOGE("draw failed");
        }
    }
    state.shell.stop();
}
