// SPDX-License-Identifier: MIT
#include "../internal.hpp"
#include "vpp.hpp"
#include <vulkan/vulkan.h>
#include "tonemap_spirv.h"
#include "overlay_spirv.h"
#include "transform_spirv.h"
#include "vertical_spirv.h"
#include "../cdsp/filter.h"
#include <array>
#include <map>
#include <cmath>
#include <cerrno>
#include <linux/dma-buf.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

namespace irisva {
namespace {
void vk_check(VkResult result, const char *operation) {
    if (result != VK_SUCCESS) {
        trace("Vulkan %s failed: %d", operation, result);
        throw Error(VA_STATUS_ERROR_OPERATION_FAILED, operation);
    }
}
static_assert(sizeof(OverlayParams) == 120, "overlay push constant ABI");
struct Parameters {
    uint32_t width, height, src_stride, src_y, src_uv, dst_stride, dst_y, dst_uv;
    float source_pq, destination, knee;
};
struct TransformParameters {
    int32_t src_width, src_height, scaled_width, scaled_height, width, height;
    int32_t src_stride, src_y, src_uv, dst_stride, dst_y, dst_uv;
    int32_t src_wide, dst_wide, rotation, enlarging;
    int32_t filter_x, filter_y, filter_uv_x, filter_uv_y;
};
static_assert(sizeof(TransformParameters) <= sizeof(OverlayParams), "transform push constant ABI");
static_assert(sizeof(TransformParameters) == 80 && FILTER_MAX_TAPS == 22,
              "transform shader parameter and filter layout");
double encode_pq(double nits) {
    double power = std::pow(nits / 10000.0, 0.1593017578125);
    return std::pow((0.8359375 + 18.8515625 * power) / (1.0 + 18.6875 * power), 78.84375);
}
void validate_layout(const Memory &memory, const Surface &surface) {
    unsigned row_bytes = surface.width * (surface.fourcc == VA_FOURCC_P010 ? 2 : 1);
    check(memory.fd >= 0 && memory.fourcc == surface.fourcc && memory.width >= surface.width &&
              memory.height >= surface.height && memory.stride >= row_bytes &&
              memory.storage_height >= surface.height && !(memory.stride & 1) &&
              !(memory.data_offset & 1) && !(memory.chroma() & 1) && memory.size <= UINT32_MAX &&
              uint64_t(memory.chroma()) + uint64_t(memory.stride) * (surface.height / 2 - 1) +
                      row_bytes <=
                  memory.size,
          "unsupported Vulkan DMA-BUF layout", VA_STATUS_ERROR_INVALID_SURFACE);
}
void publish_dma_buf(const Memory &memory) {
    // Extra exporter synchronization fixes GPU-to-CDSP visibility on the
    // tested SC8280XP stack. These CPU-access hooks neither map nor copy
    // pixels; this is a workaround, not a generic device-cache flush.
    for (uint64_t phase : {DMA_BUF_SYNC_START, DMA_BUF_SYNC_END}) {
        dma_buf_sync sync{phase | DMA_BUF_SYNC_RW};
        int result;
        do {
            result = ioctl(memory.fd, DMA_BUF_IOCTL_SYNC, &sync);
        } while (result < 0 && errno == EINTR);
        check(result == 0, "publish Vulkan DMA-BUF writes");
    }
}
} // namespace

struct VulkanVpp::Impl : std::enable_shared_from_this<Impl> {
    // Imported buffers retain the device, even after the VPP context is destroyed.
    // Memory releases its registration before closing the original DMA-BUF fd.
    struct Import {
        std::shared_ptr<Impl> owner;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        ~Import() {
            vkDestroyBuffer(owner->device, buffer, nullptr);
            vkFreeMemory(owner->device, memory, nullptr);
        }
    };
    std::mutex mutex;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t family = 0;
    uint32_t max_buffer_range = 0;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkShaderModule overlay_shader = VK_NULL_HANDLE;
    VkPipeline overlay_pipeline = VK_NULL_HANDLE;
    VkShaderModule transform_shader = VK_NULL_HANDLE;
    VkShaderModule vertical_shader = VK_NULL_HANDLE;
    std::map<std::array<uint32_t, 6>, std::array<VkPipeline, 2>> transform_pipelines;
    VkBuffer scratch_buffer = VK_NULL_HANDLE;
    VkDeviceMemory scratch_memory = VK_NULL_HANDLE;
    VkDeviceSize scratch_size = 0;
    VkBuffer filter_buffer = VK_NULL_HANDLE;
    VkDeviceMemory filter_memory = VK_NULL_HANDLE;
    std::array<int, 4> filter_geometry{};
    std::array<int, 4> filter_offsets{};
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    PFN_vkGetMemoryFdPropertiesKHR get_fd_properties = nullptr;

    ~Impl() {
        if (device) {
            vkDestroyBuffer(device, scratch_buffer, nullptr);
            vkFreeMemory(device, scratch_memory, nullptr);
            for (const auto &entry : transform_pipelines)
                for (auto pipeline : entry.second)
                    vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyShaderModule(device, vertical_shader, nullptr);
            vkDestroyBuffer(device, filter_buffer, nullptr);
            vkFreeMemory(device, filter_memory, nullptr);
            vkDestroyShaderModule(device, transform_shader, nullptr);
            vkDestroyFence(device, fence, nullptr);
            vkDestroyCommandPool(device, command_pool, nullptr);
            vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
            vkDestroyPipeline(device, overlay_pipeline, nullptr);
            vkDestroyShaderModule(device, overlay_shader, nullptr);
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyShaderModule(device, shader, nullptr);
            vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        vkDestroyInstance(instance, nullptr);
    }
    void initialize(dev_t render_device) {
        VkApplicationInfo application{};
        application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        application.pApplicationName = "qcom-va-driver";
        application.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo instance_info{};
        instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instance_info.pApplicationInfo = &application;
        vk_check(vkCreateInstance(&instance_info, nullptr, &instance), "create instance");
        uint32_t count = 0;
        vk_check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "enumerate devices");
        std::vector<VkPhysicalDevice> devices(count);
        vk_check(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "enumerate devices");
        const char *extensions[] = {
            VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
            VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME, VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME};
        for (auto candidate : devices) {
            uint32_t extension_count = 0;
            vk_check(
                vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extension_count, nullptr),
                "enumerate extensions");
            std::vector<VkExtensionProperties> available(extension_count);
            vk_check(vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extension_count,
                                                          available.data()),
                     "enumerate extensions");
            bool supported = true;
            for (auto name : extensions)
                supported &= std::any_of(available.begin(), available.end(), [&](const auto &e) {
                    return std::strcmp(name, e.extensionName) == 0;
                });
            if (!supported)
                continue;
            VkPhysicalDeviceDrmPropertiesEXT drm{};
            drm.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT;
            VkPhysicalDeviceProperties2 properties{};
            properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            properties.pNext = &drm;
            vkGetPhysicalDeviceProperties2(candidate, &properties);
            bool matches_render = drm.hasRender && drm.renderMajor == major(render_device) &&
                                  drm.renderMinor == minor(render_device);
            bool matches_primary = drm.hasPrimary && drm.primaryMajor == major(render_device) &&
                                   drm.primaryMinor == minor(render_device);
            if ((!matches_render && !matches_primary) ||
                properties.properties.apiVersion < VK_API_VERSION_1_1)
                continue;
            VkPhysicalDevice16BitStorageFeatures storage{};
            storage.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
            VkPhysicalDeviceFeatures2 features{};
            features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features.pNext = &storage;
            vkGetPhysicalDeviceFeatures2(candidate, &features);
            if (!storage.storageBuffer16BitAccess)
                continue;
            VkPhysicalDeviceExternalBufferInfo external{};
            external.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
            external.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            external.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
            VkExternalBufferProperties external_properties{};
            external_properties.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
            vkGetPhysicalDeviceExternalBufferProperties(candidate, &external, &external_properties);
            if (!(external_properties.externalMemoryProperties.externalMemoryFeatures &
                  VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT))
                continue;
            uint32_t family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, nullptr);
            std::vector<VkQueueFamilyProperties> families(family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, families.data());
            for (uint32_t i = 0; i < family_count; ++i) {
                if (!(families[i].queueFlags & VK_QUEUE_COMPUTE_BIT))
                    continue;
                physical = candidate;
                family = i;
                max_buffer_range = properties.properties.limits.maxStorageBufferRange;
                trace("Vulkan VPP: %s", properties.properties.deviceName);
                break;
            }
            if (physical)
                break;
        }
        check(physical, "Vulkan DMA-BUF compute unavailable", VA_STATUS_ERROR_UNSUPPORTED_FILTER);
        float priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;
        VkPhysicalDevice16BitStorageFeatures storage{};
        storage.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
        storage.storageBuffer16BitAccess = VK_TRUE;
        VkDeviceCreateInfo device_info{};
        device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info.pNext = &storage;
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        device_info.enabledExtensionCount = std::size(extensions);
        device_info.ppEnabledExtensionNames = extensions;
        vk_check(vkCreateDevice(physical, &device_info, nullptr, &device), "create device");
        vkGetDeviceQueue(device, family, 0, &queue);
        get_fd_properties = reinterpret_cast<PFN_vkGetMemoryFdPropertiesKHR>(
            vkGetDeviceProcAddr(device, "vkGetMemoryFdPropertiesKHR"));
        check(get_fd_properties, "Vulkan external memory entrypoint unavailable");
        VkDescriptorSetLayoutBinding bindings[5]{};
        for (unsigned i = 0; i < 5; ++i)
            bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
                           nullptr};
        VkDescriptorSetLayoutCreateInfo set_info{};
        set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        set_info.bindingCount = 5;
        set_info.pBindings = bindings;
        vk_check(vkCreateDescriptorSetLayout(device, &set_info, nullptr, &set_layout),
                 "create descriptor layout");
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(OverlayParams)};
        VkPipelineLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts = &set_layout;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push;
        vk_check(vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout),
                 "create pipeline layout");
        VkShaderModuleCreateInfo shader_info{};
        shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shader_info.codeSize = sizeof(tonemap_spirv);
        shader_info.pCode = tonemap_spirv;
        vk_check(vkCreateShaderModule(device, &shader_info, nullptr, &shader), "create shader");
        VkComputePipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = shader;
        pipeline_info.stage.pName = "main";
        pipeline_info.layout = pipeline_layout;
        vk_check(
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline),
            "create pipeline");
        shader_info.codeSize = sizeof(overlay_spirv);
        shader_info.pCode = overlay_spirv;
        vk_check(vkCreateShaderModule(device, &shader_info, nullptr, &overlay_shader),
                 "create overlay shader");
        pipeline_info.stage.module = overlay_shader;
        vk_check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
                                          &overlay_pipeline),
                 "create overlay pipeline");
        shader_info.codeSize = sizeof(transform_spirv);
        shader_info.pCode = transform_spirv;
        vk_check(vkCreateShaderModule(device, &shader_info, nullptr, &transform_shader),
                 "create transform shader");
        shader_info.codeSize = sizeof(vertical_spirv);
        shader_info.pCode = vertical_spirv;
        vk_check(vkCreateShaderModule(device, &shader_info, nullptr, &vertical_shader),
                 "create vertical shader");
        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5};
        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        vk_check(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool),
                 "create descriptor pool");
        VkDescriptorSetAllocateInfo set_allocate{};
        set_allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        set_allocate.descriptorPool = descriptor_pool;
        set_allocate.descriptorSetCount = 1;
        set_allocate.pSetLayouts = &set_layout;
        vk_check(vkAllocateDescriptorSets(device, &set_allocate, &descriptor_set),
                 "allocate descriptors");
        VkCommandPoolCreateInfo command_info{};
        command_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        command_info.queueFamilyIndex = family;
        vk_check(vkCreateCommandPool(device, &command_info, nullptr, &command_pool),
                 "create command pool");
        VkCommandBufferAllocateInfo command_allocate{};
        command_allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        command_allocate.commandPool = command_pool;
        command_allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_allocate.commandBufferCount = 1;
        vk_check(vkAllocateCommandBuffers(device, &command_allocate, &commands),
                 "allocate command buffer");
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vk_check(vkCreateFence(device, &fence_info, nullptr, &fence), "create fence");
    }
    std::shared_ptr<Import> import(Memory &memory) {
        if (memory.vulkan_registration) {
            auto imported = std::static_pointer_cast<Import>(memory.vulkan_registration);
            check(imported->owner.get() == this, "DMA-BUF belongs to another Vulkan device");
            return imported;
        }
        check(memory.size <= max_buffer_range, "Vulkan storage buffer exceeds device limit",
              VA_STATUS_ERROR_RESOLUTION_NOT_SUPPORTED);
        auto imported = std::make_shared<Import>();
        imported->owner = shared_from_this();
        VkExternalMemoryBufferCreateInfo external{};
        external.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.pNext = &external;
        buffer_info.size = memory.size;
        buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        vk_check(vkCreateBuffer(device, &buffer_info, nullptr, &imported->buffer),
                 "create external buffer");
        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(device, imported->buffer, &requirements);
        check(requirements.size <= memory.size, "DMA-BUF too small for Vulkan alignment",
              VA_STATUS_ERROR_INVALID_SURFACE);
        VkMemoryFdPropertiesKHR properties{};
        properties.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR;
        vk_check(get_fd_properties(device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
                                   memory.fd, &properties),
                 "query DMA-BUF memory");
        uint32_t types = properties.memoryTypeBits & requirements.memoryTypeBits;
        check(types, "no compatible Vulkan DMA-BUF memory type", VA_STATUS_ERROR_INVALID_SURFACE);
        VkMemoryDedicatedAllocateInfo dedicated{};
        dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        dedicated.buffer = imported->buffer;
        VkImportMemoryFdInfoKHR fd_info{};
        fd_info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
        fd_info.pNext = &dedicated;
        fd_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        fd_info.fd = dup(memory.fd);
        check(fd_info.fd >= 0, "duplicate Vulkan DMA-BUF fd", VA_STATUS_ERROR_ALLOCATION_FAILED);
        VkMemoryAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocation.pNext = &fd_info;
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = __builtin_ctz(types);
        VkResult result = vkAllocateMemory(device, &allocation, nullptr, &imported->memory);
        // Vulkan takes ownership of the duplicated fd only on successful import.
        if (result != VK_SUCCESS)
            close(fd_info.fd);
        vk_check(result, "import DMA-BUF");
        vk_check(vkBindBufferMemory(device, imported->buffer, imported->memory, 0), "bind DMA-BUF");
        memory.vulkan_registration = imported;
        return imported;
    }
    std::array<VkPipeline, 2> transform_kernels(const TransformParameters &p) {
        auto taps = [](int input, int output) -> uint32_t {
            return input < output ? 4 : input % output || input / output > 4 ? 0 : input / output;
        };
        unsigned mode = p.enlarging                                                          ? 2
                        : (p.src_width != p.scaled_width || p.src_height != p.scaled_height) ? 1
                                                                                             : 0;
        std::array<uint32_t, 6> key{uint32_t(p.src_wide),
                                    uint32_t(p.dst_wide),
                                    uint32_t(p.rotation),
                                    mode,
                                    taps(p.src_width, p.scaled_width),
                                    taps(p.src_height, p.scaled_height)};
        // Large area footprints benefit from a shared vertical pass too. Keep
        // their tap loops dynamic to avoid unrolling hundreds of buffer loads.
        if (mode == 1 && (p.src_width > p.scaled_width * 4 || p.src_height > p.scaled_height * 4))
            key[3] = mode = 3;
        auto &pipelines = transform_pipelines[key];
        VkSpecializationMapEntry entries[6];
        for (uint32_t i = 0; i < 6; ++i)
            entries[i] = {i, i * uint32_t(sizeof(uint32_t)), sizeof(uint32_t)};
        VkSpecializationInfo specialization{6, entries, sizeof(key), key.data()};
        for (unsigned i = 0; i < (mode >= 2 ? 2u : 1u); ++i) {
            if (pipelines[i])
                continue;
            VkComputePipelineCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            info.layout = pipeline_layout;
            info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            info.stage.module = i ? vertical_shader : transform_shader;
            info.stage.pName = "main";
            info.stage.pSpecializationInfo = &specialization;
            vk_check(
                vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipelines[i]),
                "create transform pipeline");
        }
        return pipelines;
    }
    void allocate_buffer(VkBuffer &buffer, VkDeviceMemory &memory, VkDeviceSize size,
                         VkMemoryPropertyFlags flags) {
        check(size <= max_buffer_range, "Vulkan internal buffer exceeds device limit",
              VA_STATUS_ERROR_RESOLUTION_NOT_SUPPORTED);
        vkDestroyBuffer(device, buffer, nullptr);
        vkFreeMemory(device, memory, nullptr);
        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = size;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        vk_check(vkCreateBuffer(device, &info, nullptr, &buffer), "create internal buffer");
        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(device, buffer, &requirements);
        VkPhysicalDeviceMemoryProperties properties;
        vkGetPhysicalDeviceMemoryProperties(physical, &properties);
        unsigned index = properties.memoryTypeCount;
        for (unsigned i = 0; i < properties.memoryTypeCount; ++i)
            if ((requirements.memoryTypeBits & (1u << i)) &&
                (properties.memoryTypes[i].propertyFlags & flags) == flags) {
                index = i;
                break;
            }
        check(index < properties.memoryTypeCount, "Vulkan buffer memory type unavailable");
        VkMemoryAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = index;
        vk_check(vkAllocateMemory(device, &allocation, nullptr, &memory),
                 "allocate internal buffer");
        vk_check(vkBindBufferMemory(device, buffer, memory, 0), "bind internal buffer");
    }
    void prepare_filters(int sw, int sh, int dw, int dh) {
        std::array<int, 4> geometry{sw, sh, dw, dh};
        if (geometry == filter_geometry)
            return;
        std::vector<uint32_t> words;
        for (int axis = 0; axis < 4; ++axis) {
            int channels = axis >= 2 ? 2 : 1;
            int input = (axis & 1 ? sh : sw) / channels;
            int output = (axis & 1 ? dh : dw) / channels;
            filter_offsets[axis] = words.size();
            for (int i = 0; i < output; ++i) {
                auto filter = make_filter(input, output, i);
                words.push_back(filter.tap_count);
                words.push_back(filter.shift);
                for (int tap = 0; tap < FILTER_MAX_TAPS; ++tap)
                    words.push_back(uint32_t(filter.indices[tap]) |
                                    (uint32_t(uint16_t(filter.weights[tap])) << 16));
            }
        }
        // Only coefficients are CPU mapped. Input/output pixels remain in their
        // original DMA-BUFs, and unchanged geometry reuses the coefficient buffer.
        filter_geometry = {};
        VkDeviceSize size = words.size() * sizeof(uint32_t);
        allocate_buffer(filter_buffer, filter_memory, size,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void *mapped;
        vk_check(vkMapMemory(device, filter_memory, 0, size, 0, &mapped), "map filters");
        std::memcpy(mapped, words.data(), size);
        vkUnmapMemory(device, filter_memory);
        filter_geometry = geometry;
    }
    void dispatch(VkPipeline selected_pipeline, Memory *src, Memory *dst, Memory *rgba,
                  const void *parameters, uint32_t parameter_size, unsigned groups_x,
                  unsigned groups_y, bool filtering = false, VkPipeline vertical = VK_NULL_HANDLE,
                  unsigned vertical_x = 0, unsigned vertical_y = 0) {
        Memory *memories[] = {src, dst, rgba ? rgba : src};
        VkDescriptorBufferInfo buffers[5]{};
        VkWriteDescriptorSet writes[5]{};
        for (unsigned i = 0; i < 5; ++i) {
            if (i < 3) {
                wait_surface_access(*memories[i], i == 1 ? POLLOUT : POLLIN);
                buffers[i] = {import(*memories[i])->buffer, 0, memories[i]->size};
            } else if (i == 3)
                buffers[i] = filtering ? VkDescriptorBufferInfo{filter_buffer, 0, VK_WHOLE_SIZE}
                                       : buffers[0];
            else
                buffers[i] = vertical_x ? VkDescriptorBufferInfo{scratch_buffer, 0, VK_WHOLE_SIZE}
                                        : buffers[0];
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptor_set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffers[i];
        }
        vkUpdateDescriptorSets(device, 5, writes, 0, nullptr);
        vk_check(vkResetCommandPool(device, command_pool, 0), "reset command pool");
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_check(vkBeginCommandBuffer(commands, &begin), "begin command buffer");
        // DMA-BUF producers/consumers are outside this Vulkan queue family. Import
        // alone does not acquire ownership or replace their implicit fences.
        VkBufferMemoryBarrier barriers[3]{};
        unsigned barrier_count = 0;
        for (unsigned i = 0; i < 3; ++i) {
            if (i && buffers[i].buffer == buffers[0].buffer) {
                if (i == 1)
                    barriers[0].dstAccessMask |= VK_ACCESS_SHADER_WRITE_BIT;
                continue;
            }
            unsigned j = barrier_count++;
            barriers[j].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barriers[j].dstAccessMask =
                i == 1 ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT;
            barriers[j].srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
            barriers[j].dstQueueFamilyIndex = family;
            barriers[j].buffer = buffers[i].buffer;
            barriers[j].size = VK_WHOLE_SIZE;
        }
        vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, barrier_count,
                             barriers, 0, nullptr);
        vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                                &descriptor_set, 0, nullptr);
        vkCmdPushConstants(commands, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           parameter_size, parameters);
        if (vertical_x) {
            vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, vertical);
            vkCmdDispatch(commands, vertical_x, vertical_y, 1);
            VkBufferMemoryBarrier scratch{};
            scratch.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            scratch.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            scratch.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            scratch.srcQueueFamilyIndex = scratch.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            scratch.buffer = scratch_buffer;
            scratch.size = VK_WHOLE_SIZE;
            vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &scratch,
                                 0, nullptr);
        }
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, selected_pipeline);
        vkCmdDispatch(commands, groups_x, groups_y, 1);
        for (unsigned i = 0; i < barrier_count; ++i) {
            auto &barrier = barriers[i];
            barrier.srcAccessMask = barrier.dstAccessMask;
            barrier.dstAccessMask = 0;
            barrier.srcQueueFamilyIndex = family;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
        }
        vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, barrier_count,
                             barriers, 0, nullptr);
        vk_check(vkEndCommandBuffer(commands), "end command buffer");
        vk_check(vkResetFences(device, 1, &fence), "reset fence");
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commands;
        vk_check(vkQueueSubmit(queue, 1, &submit, fence), "submit VPP");
        // Complete before returning buffers to CDSP, V4L2, or the VA caller. No CPU
        // mapping or pixel copy is involved; imports persist for the Memory lifetime.
        vk_check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "wait for VPP");
        publish_dma_buf(*memories[1]);
    }
};

VulkanVpp::VulkanVpp(int render_fd) {
    struct stat info{};
    check(fstat(render_fd, &info) == 0 && S_ISCHR(info.st_mode), "invalid Vulkan render device");
    static std::mutex cache_mutex;
    static std::map<dev_t, std::weak_ptr<Impl>> cache;
    std::lock_guard<std::mutex> guard(cache_mutex);
    impl_ = cache[info.st_rdev].lock();
    if (!impl_) {
        impl_ = std::make_shared<Impl>();
        impl_->initialize(info.st_rdev);
        cache[info.st_rdev] = impl_;
    }
}
void VulkanVpp::process(const VppPicture &picture, Surface &target) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    auto &src = *picture.source->memory;
    auto &dst = *target.memory;
    validate_layout(src, *picture.source);
    validate_layout(dst, target);
    auto crop = picture.region;
    double source_pq = encode_pq(picture.source_peak);
    trace("Vulkan tone mapping %ux%u peak=%g src=%u/%u/%u dst=%u/%u/%u", crop.width, crop.height,
          picture.source_peak, src.stride, src.data_offset, src.storage_height, dst.stride,
          dst.data_offset, dst.storage_height);
    float destination = encode_pq(100.0) / source_pq;
    Parameters params{crop.width,
                      crop.height,
                      src.stride,
                      src.data_offset + unsigned(crop.y) * src.stride + unsigned(crop.x) * 2,
                      unsigned(src.chroma()) + unsigned(crop.y / 2) * src.stride +
                          unsigned(crop.x) * 2,
                      dst.stride,
                      dst.data_offset,
                      unsigned(dst.chroma()),
                      float(source_pq),
                      destination,
                      1.5f * destination - 0.5f};
    impl_->dispatch(impl_->pipeline, &src, &dst, nullptr, &params, sizeof(params),
                    (crop.width + 255) / 256, crop.height / 2);
}
void VulkanVpp::overlay(Surface &base, Surface &rgba, Surface &target,
                        const OverlayParams &params) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    validate_layout(*base.memory, base);
    validate_layout(*target.memory, target);
    impl_->dispatch(impl_->overlay_pipeline, base.memory.get(), target.memory.get(),
                    rgba.memory.get(), &params, sizeof(params), (target.width + 127) / 128,
                    target.height / 2);
}
void VulkanVpp::transform(const VppPicture &picture, Surface &target) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    auto &src = *picture.source->memory, &dst = *target.memory;
    validate_layout(src, *picture.source);
    validate_layout(dst, target);
    check(src.size <= INT32_MAX && dst.size <= INT32_MAX,
          "Vulkan transform buffer exceeds signed offset range", VA_STATUS_ERROR_INVALID_SURFACE);
    TransformParameters p{};
    auto crop = picture.region;
    p.src_width = crop.width;
    p.src_height = crop.height;
    p.width = target.width;
    p.height = target.height;
    p.scaled_width = picture.rotation & 1 ? target.height : target.width;
    p.scaled_height = picture.rotation & 1 ? target.width : target.height;
    p.src_wide = picture.source->fourcc == VA_FOURCC_P010;
    p.dst_wide = target.fourcc == VA_FOURCC_P010;
    p.rotation = picture.rotation;
    p.enlarging = crop.width < p.scaled_width || crop.height < p.scaled_height;
    p.src_stride = src.stride;
    p.src_y = src.data_offset + crop.y * src.stride + crop.x * (1 + p.src_wide);
    p.src_uv = int32_t(src.chroma()) + src.stride * (crop.y / 2) + crop.x * (1 + p.src_wide);
    p.dst_stride = dst.stride;
    p.dst_y = dst.data_offset;
    p.dst_uv = int32_t(dst.chroma());
    bool scaling = p.src_width != p.scaled_width || p.src_height != p.scaled_height;
    if (scaling) {
        impl_->prepare_filters(p.src_width, p.src_height, p.scaled_width, p.scaled_height);
        p.filter_x = impl_->filter_offsets[0];
        p.filter_y = impl_->filter_offsets[1];
        p.filter_uv_x = impl_->filter_offsets[2];
        p.filter_uv_y = impl_->filter_offsets[3];
    }
    auto pipelines = impl_->transform_kernels(p);
    if (pipelines[1]) {
        VkDeviceSize size = VkDeviceSize(p.src_width) * p.scaled_height * 3;
        if (size > impl_->scratch_size) {
            impl_->scratch_size = 0;
            impl_->allocate_buffer(impl_->scratch_buffer, impl_->scratch_memory, size,
                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            impl_->scratch_size = size;
        }
    }
    impl_->dispatch(pipelines[0], &src, &dst, nullptr, &p, sizeof(p), (target.width + 15) / 16,
                    (target.height + 15) / 16, scaling, pipelines[1],
                    pipelines[1] ? (p.src_width + 15) / 16 : 0, (p.scaled_height + 15) / 16);
    trace("VPP Vulkan %ux%u -> %ux%u rotation=%u", crop.width, crop.height, target.width,
          target.height, picture.rotation * 90);
}

} // namespace irisva
