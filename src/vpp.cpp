// SPDX-License-Identifier: MIT
#include "internal.hpp"
#include "overlay-params.h"
#include <cmath>
#ifdef HAVE_CDSP_VPP
#include "cdsp/client.h"
#include <dlfcn.h>
#endif
#ifdef HAVE_VULKAN_VPP
#include "vulkan/vpp.hpp"
#endif
#if defined(HAVE_CDSP_VPP) || defined(HAVE_VULKAN_VPP)
#include <climits>
#include <cerrno>
#include <fcntl.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace irisva {
#ifdef HAVE_CDSP_VPP
namespace {
struct CdspApi {
    void *library = nullptr;
    decltype(&cdsp_vpp_create) create = nullptr;
    decltype(&cdsp_vpp_destroy) destroy = nullptr;
    decltype(&cdsp_vpp_import) import = nullptr;
    decltype(&cdsp_vpp_release) release = nullptr;
    decltype(&cdsp_vpp_transform) transform = nullptr;
    decltype(&cdsp_vpp_overlay) overlay = nullptr;
    ~CdspApi() {
        if (library)
            dlclose(library);
    }
};
std::shared_ptr<CdspApi> load_cdsp_api() {
    static std::mutex lock;
    static std::weak_ptr<CdspApi> cached;
    std::lock_guard<std::mutex> guard(lock);
    if (auto api = cached.lock())
        return api;
    auto api = std::make_shared<CdspApi>();
    api->library = dlopen("libiris_vpp.so", RTLD_NOW | RTLD_LOCAL);
    if (!api->library)
        api->library = dlopen(IRIS_VPP_CLIENT_PATH, RTLD_NOW | RTLD_LOCAL);
    if (!api->library) {
        trace("CDSP VPP runtime unavailable: %s", dlerror());
        return {};
    }
#define LOAD(name)                                                                                 \
    api->name = reinterpret_cast<decltype(api->name)>(dlsym(api->library, "cdsp_vpp_" #name));     \
    if (!api->name)                                                                                \
        return {};
    LOAD(create)
    LOAD(destroy)
    LOAD(import)
    LOAD(release)
    LOAD(transform)
    LOAD(overlay)
#undef LOAD
    cached = api;
    return api;
}
} // namespace
#endif
bool vpp_supported(VAProfile profile, VAEntrypoint entrypoint) {
#ifdef HAVE_VULKAN_VPP
    return profile == VAProfileNone && entrypoint == VAEntrypointVideoProc;
#elif defined(HAVE_CDSP_VPP)
    return profile == VAProfileNone && entrypoint == VAEntrypointVideoProc && bool(load_cdsp_api());
#else
    (void)profile;
    (void)entrypoint;
    return false;
#endif
}
void vpp_caps(VAProcPipelineCaps &caps, bool tone_mapping) {
    // These are driver-owned arrays, as consumed by FFmpeg and other VA clients.
    // Do not read the incoming structure: clients may pass it uninitialized.
    static VAProcColorStandardType colors[] = {VAProcColorStandardNone, VAProcColorStandardBT601,
                                               VAProcColorStandardBT709, VAProcColorStandardBT2020,
                                               VAProcColorStandardExplicit};
    static uint32_t formats[] = {VA_FOURCC_NV12, VA_FOURCC_P010, VA_FOURCC_BGRA, VA_FOURCC_RGBA};
    caps = {};
    caps.input_color_standards = caps.output_color_standards = colors;
    caps.num_input_color_standards = caps.num_output_color_standards =
        sizeof(colors) / sizeof(colors[0]);
    caps.input_pixel_format = caps.output_pixel_format = formats;
    caps.num_input_pixel_formats = 4;
    caps.num_output_pixel_formats = 2;
    caps.blend_flags = VA_BLEND_GLOBAL_ALPHA | VA_BLEND_PREMULTIPLIED_ALPHA;
    if (tone_mapping) {
        caps.input_color_standards = &colors[3];
        caps.output_color_standards = &colors[2];
        caps.num_input_color_standards = caps.num_output_color_standards = 1;
        caps.input_pixel_format = &formats[1];
        caps.output_pixel_format = &formats[0];
        caps.num_input_pixel_formats = caps.num_output_pixel_formats = 1;
    }
    caps.min_input_width = caps.min_output_width = 2;
#if defined(HAVE_CDSP_VPP) || defined(HAVE_VULKAN_VPP)
    caps.rotation_flags = (1 << VA_ROTATION_NONE) | (1 << VA_ROTATION_90) | (1 << VA_ROTATION_180) |
                          (1 << VA_ROTATION_270);
#endif
    caps.min_input_height = caps.min_output_height = 2;
    caps.max_input_width = caps.max_output_width = 8192;
    caps.max_input_height = caps.max_output_height = 8192;
}
void vpp_filter(const Buffer &buffer) {
    check(buffer.type == VAProcFilterParameterBufferType && buffer.elements == 1 &&
              buffer.element_size >= sizeof(VAProcFilterParameterBufferHDRToneMapping),
          "invalid VPP filter buffer", VA_STATUS_ERROR_INVALID_BUFFER);
    VAProcFilterParameterBufferHDRToneMapping filter{};
    std::memcpy(&filter, buffer.data.data(), sizeof(filter));
    check(filter.type == VAProcFilterHighDynamicRangeToneMapping, "unsupported VPP filter",
          VA_STATUS_ERROR_UNSUPPORTED_FILTER);
}
VppPicture vpp_picture(const Buffer &buffer, const std::shared_ptr<Surface> &source,
                       const Surface &target, const Buffer *filter, bool overlay) {
    VAProcPipelineParameterBuffer p{};
    std::memcpy(&p, buffer.data.data(), sizeof(p));
    check(!p.num_forward_references && !p.num_backward_references && !p.mirror_state &&
              !p.num_additional_outputs && !(p.pipeline_flags & ~VA_PROC_PIPELINE_FAST) &&
              !(p.filter_flags & ~VA_FILTER_SCALING_MASK) && !p.input_surface_flag &&
              !p.output_surface_flag && !p.output_hdr_metadata,
          "unsupported VPP processing operation", VA_STATUS_ERROR_UNIMPLEMENTED);
    check(p.rotation_state <= VA_ROTATION_270, "invalid VPP rotation",
          VA_STATUS_ERROR_INVALID_PARAMETER);
    const auto &in = p.input_color_properties;
    const auto &out = p.output_color_properties;
    auto matches = [](unsigned a, unsigned b) { return !a || !b || a == b; };
    if (!filter && !overlay) {
        check(matches(p.surface_color_standard, p.output_color_standard) &&
                  matches(in.color_range, out.color_range) &&
                  matches(in.chroma_sample_location, out.chroma_sample_location) &&
                  matches(in.colour_primaries, out.colour_primaries) &&
                  matches(in.transfer_characteristics, out.transfer_characteristics) &&
                  matches(in.matrix_coefficients, out.matrix_coefficients),
              "VPP color conversion unsupported", VA_STATUS_ERROR_UNIMPLEMENTED);
    } else if (filter) {
        auto color = [](unsigned value, unsigned expected) {
            return !value || value == 2 || value == expected; // ISO unspecified = 2.
        };
        check((p.surface_color_standard == VAProcColorStandardNone ||
               p.surface_color_standard == VAProcColorStandardBT2020 ||
               p.surface_color_standard == VAProcColorStandardExplicit) &&
                  (p.output_color_standard == VAProcColorStandardNone ||
                   p.output_color_standard == VAProcColorStandardBT709 ||
                   p.output_color_standard == VAProcColorStandardExplicit) &&
                  color(in.colour_primaries, 9) && color(in.transfer_characteristics, 16) &&
                  color(in.matrix_coefficients, 9) && color(out.colour_primaries, 1) &&
                  color(out.transfer_characteristics, 1) && color(out.matrix_coefficients, 1) &&
                  matches(in.color_range, VA_SOURCE_RANGE_REDUCED) &&
                  matches(out.color_range, VA_SOURCE_RANGE_REDUCED) &&
                  matches(in.chroma_sample_location, out.chroma_sample_location),
              "tone mapping requires limited-range BT.2020/PQ to BT.709",
              VA_STATUS_ERROR_UNIMPLEMENTED);
        check(source->fourcc == VA_FOURCC_P010 && target.fourcc == VA_FOURCC_NV12,
              "tone mapping requires P010 input and NV12 output",
              VA_STATUS_ERROR_UNSUPPORTED_RT_FORMAT);
    }
    VppPicture picture;
    picture.color_standard = p.output_color_standard;
    picture.color_properties = p.output_color_properties;
    picture.output = p.output_region ? *p.output_region
                                     : VARectangle{0, 0, static_cast<uint16_t>(target.width),
                                                   static_cast<uint16_t>(target.height)};
    if (p.blend_state)
        picture.blend = *p.blend_state;
    if (overlay) {
        check(!filter && !p.rotation_state && is_rgb(source->fourcc) && source.get() != &target,
              "overlay requires a distinct BGRA/RGBA surface without filters or rotation",
              VA_STATUS_ERROR_UNIMPLEMENTED);
        check(!(picture.blend.flags & ~(VA_BLEND_GLOBAL_ALPHA | VA_BLEND_PREMULTIPLIED_ALPHA)) &&
                  (!(picture.blend.flags & VA_BLEND_GLOBAL_ALPHA) ||
                   (std::isfinite(picture.blend.global_alpha) && picture.blend.global_alpha >= 0 &&
                    picture.blend.global_alpha <= 1)),
              "unsupported overlay blend state", VA_STATUS_ERROR_INVALID_PARAMETER);
        picture.source = source;
        picture.region = p.surface_region ? *p.surface_region
                                          : VARectangle{0, 0, static_cast<uint16_t>(source->width),
                                                        static_cast<uint16_t>(source->height)};
        auto r = picture.region;
        check(r.x >= 0 && r.y >= 0 && r.width && r.height &&
                  unsigned(r.x) + r.width <= source->width &&
                  unsigned(r.y) + r.height <= source->height && picture.output.width &&
                  picture.output.height && picture.output.width <= 8192 &&
                  picture.output.height <= 8192,
              "invalid overlay rectangle", VA_STATUS_ERROR_INVALID_PARAMETER);
        return picture;
    }
    check(!p.blend_state && !picture.output.x && !picture.output.y &&
              picture.output.width == target.width && picture.output.height == target.height,
          "VPP base layer must cover the output without blending", VA_STATUS_ERROR_UNIMPLEMENTED);
    if (filter) {
        vpp_filter(*filter);
        VAProcFilterParameterBufferHDRToneMapping hdr{};
        std::memcpy(&hdr, filter->data.data(), sizeof(hdr));
        check(hdr.data.metadata_type == VAProcHighDynamicRangeMetadataHDR10 && hdr.data.metadata &&
                  hdr.data.metadata_size == sizeof(VAHdrMetaDataHDR10),
              "invalid HDR10 input metadata", VA_STATUS_ERROR_INVALID_PARAMETER);
        // Snapshot caller-owned metadata at RenderPicture; never retain its pointer.
        VAHdrMetaDataHDR10 metadata{};
        std::memcpy(&metadata, hdr.data.metadata, sizeof(metadata));
        picture.tone_mapping = true;
        float peak = metadata.max_content_light_level;
        if (!peak)
            peak = metadata.max_display_mastering_luminance / 10000.0f;
        picture.source_peak = peak ? std::clamp(peak, 100.0f, 10000.0f) : 1000.0f;
    }
    picture.source = source;
    picture.rotation = p.rotation_state;
    picture.region = p.surface_region ? *p.surface_region
                                      : VARectangle{0, 0, static_cast<uint16_t>(source->width),
                                                    static_cast<uint16_t>(source->height)};
    auto r = picture.region;
    check(r.x >= 0 && r.y >= 0 && r.width && r.height && !(r.x & 1) && !(r.y & 1) &&
              !(r.width & 1) && !(r.height & 1) && !(target.width & 1) && !(target.height & 1) &&
              unsigned(r.x) + r.width <= source->width &&
              unsigned(r.y) + r.height <= source->height,
          "VPP requires even dimensions and crop coordinates", VA_STATUS_ERROR_INVALID_PARAMETER);
    unsigned scaled_width = (picture.rotation & 1) ? target.height : target.width;
    unsigned scaled_height = (picture.rotation & 1) ? target.width : target.height;
    bool down = scaled_width <= r.width && scaled_height <= r.height;
    bool up = scaled_width >= r.width && scaled_height >= r.height;
    check((down || up) && uint64_t(scaled_width) * 20 >= r.width &&
              uint64_t(scaled_height) * 20 >= r.height,
          "VPP cannot mix up/down scaling axes or downscale beyond 20:1",
          VA_STATUS_ERROR_RESOLUTION_NOT_SUPPORTED);
    check(source.get() != &target &&
              (source->fourcc == VA_FOURCC_NV12 || source->fourcc == VA_FOURCC_P010) &&
              (target.fourcc == VA_FOURCC_NV12 || target.fourcc == VA_FOURCC_P010),
          "VPP requires distinct NV12/P010 surfaces", VA_STATUS_ERROR_INVALID_SURFACE);
    return picture;
}
#if defined(HAVE_CDSP_VPP) || defined(HAVE_VULKAN_VPP)
namespace {
std::shared_ptr<Memory> allocate_vpp_memory(int render_fd, const Surface &surface) {
    auto memory = std::make_shared<Memory>();
    memory->origin = MemoryOrigin::DmaHeap;
    memory->width = surface.width;
    memory->height = surface.height;
    memory->fourcc = surface.fourcc;
    unsigned bytes = surface.fourcc == VA_FOURCC_P010 ? 2 : 1;
    memory->stride = (surface.width * bytes + 127) & ~127u;
    memory->storage_height = (surface.height + 31) & ~31u;
    memory->size = (size_t(memory->stride) * memory->storage_height * 3 / 2 + 4095) & ~size_t(4095);
    int heap = open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
    const char *heap_operation = "open system DMA heap";
    int heap_error = errno;
    if (heap >= 0) {
        dma_heap_allocation_data data{};
        data.len = memory->size;
        data.fd_flags = O_RDWR | O_CLOEXEC;
        int result = ioctl(heap, DMA_HEAP_IOCTL_ALLOC, &data);
        heap_error = errno;
        close(heap);
        if (result == 0) {
            memory->fd = data.fd;
            return memory;
        }
        heap_operation = "allocate from system DMA heap";
    }
    trace("%s failed (errno=%d); trying coherent MSM GEM", heap_operation, heap_error);
    // Buffers can cross Vulkan/CDSP backends. WC GEM produces incorrect pixels
    // on the tested CDSP stack, so do not use the ordinary allocator's WC fallback.
    memory->fd = irisva_msm_allocate_coherent(render_fd, memory->size);
    if (memory->fd < 0) {
        int gem_error = errno;
        throw Error(VA_STATUS_ERROR_ALLOCATION_FAILED,
                    std::string(heap_operation) + " failed (errno=" + std::to_string(heap_error) +
                        "); allocate/export coherent MSM GEM failed (errno=" +
                        std::to_string(gem_error) + ")");
    }
    memory->origin = MemoryOrigin::MsmCoherent;
    return memory;
}
#ifdef HAVE_CDSP_VPP
cdsp_vpp_frame import_vpp_frame(Memory &memory, const Surface &surface,
                                const std::shared_ptr<CdspApi> &api) {
    unsigned row_bytes = surface.width * (is_rgb(surface.fourcc)             ? 4
                                          : surface.fourcc == VA_FOURCC_P010 ? 2
                                                                             : 1);
    check(memory.fd >= 0 && memory.fourcc == surface.fourcc && memory.size <= INT_MAX &&
              memory.width >= surface.width && memory.height >= surface.height &&
              memory.stride >= row_bytes && memory.storage_height >= surface.height &&
              (is_rgb(surface.fourcc) ? uint64_t(memory.data_offset) : uint64_t(memory.chroma())) +
                      uint64_t(memory.stride) *
                          (is_rgb(surface.fourcc) ? surface.height - 1 : surface.height / 2 - 1) +
                      row_bytes <=
                  memory.size,
          "unsupported CDSP DMA-BUF layout", VA_STATUS_ERROR_INVALID_SURFACE);
    if (!memory.vpp_registration) {
        cdsp_vpp_buffer *buffer = nullptr;
        int e = api->import(memory.fd, memory.size, &buffer);
        check(e == 0, "CDSP DMA-BUF import failed");
        // shared_ptr invokes this deleter even if allocating its control block fails.
        // Keep the runtime loaded until every retained map has been released.
        memory.vpp_registration = std::shared_ptr<void>(
            buffer, [api](void *p) { api->release(static_cast<cdsp_vpp_buffer *>(p)); });
    }
    return {static_cast<cdsp_vpp_buffer *>(memory.vpp_registration.get()),
            surface.width,
            surface.height,
            memory.stride,
            memory.data_offset,
            memory.chroma()};
}
#endif
} // namespace
struct Vpp::Impl {
    int render_fd;
    std::shared_ptr<Surface> intermediate;
#ifdef HAVE_VULKAN_VPP
    bool vulkan_probed = false;
    std::unique_ptr<VulkanVpp> vulkan;
#endif
#ifdef HAVE_CDSP_VPP
    std::shared_ptr<CdspApi> api;
    cdsp_vpp_context *context = nullptr;
    bool cdsp_unavailable = false;
    void initialize_cdsp() {
        if (context)
            return;
        api = load_cdsp_api();
        check(bool(api), "CDSP VPP runtime unavailable", VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT);
        int e = api->create(&context, CDSP_VPP_MAX_WORKERS, 1); // Capped by available HVX contexts.
        if (e) {
            trace("CDSP VPP initialization failed: %d", e);
            throw Error(VA_STATUS_ERROR_OPERATION_FAILED, "CDSP VPP initialization failed");
        }
        cdsp_unavailable = false;
    }
    ~Impl() {
        if (context)
            api->destroy(context);
    }
#endif
    explicit Impl(int fd) : render_fd(fd) {}
};
Vpp::Vpp(int render_fd) : impl_(std::make_unique<Impl>(render_fd)) {}
bool Vpp::vulkan_available() {
#ifdef HAVE_VULKAN_VPP
    if (!impl_->vulkan_probed) {
        impl_->vulkan_probed = true;
        try {
            impl_->vulkan = std::make_unique<VulkanVpp>(impl_->render_fd);
        } catch (const Error &error) {
            trace("Vulkan VPP unavailable: %s", error.what());
        }
    }
    return bool(impl_->vulkan);
#else
    return false;
#endif
}
bool Vpp::use_vulkan(bool overlay) {
    const char *requested = overlay ? std::getenv("IRIS_VAAPI_OVERLAY") : nullptr;
    if (!requested)
        requested = std::getenv("IRIS_VAAPI_VPP");
    if (!requested)
        requested = IRIS_VPP_DEFAULT_BACKEND;
    check(!std::strcmp(requested, "auto") || !std::strcmp(requested, "cdsp") ||
              !std::strcmp(requested, "vulkan"),
          "VPP backend must be auto, cdsp or vulkan", VA_STATUS_ERROR_INVALID_PARAMETER);
    bool automatic = !std::strcmp(requested, "auto");
    if (!std::strcmp(requested, "vulkan") || (automatic && overlay)) {
        if (vulkan_available())
            return true;
        check(automatic, "Vulkan VPP unavailable", VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT);
    }
#ifdef HAVE_CDSP_VPP
    if (automatic) {
        if (!impl_->cdsp_unavailable) {
            try {
                impl_->initialize_cdsp();
                return false;
            } catch (const Error &error) {
                impl_->cdsp_unavailable = true;
                trace("CDSP VPP unavailable: %s", error.what());
            }
        }
    } else {
        impl_->initialize_cdsp();
        return false;
    }
#else
    check(automatic, "CDSP VPP was disabled at build time", VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT);
#endif
    check(vulkan_available(), "No VPP backend available", VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT);
    return true;
}
bool Vpp::tone_mapping_supported() {
    return vulkan_available();
}
void Vpp::process_single(const VppPicture &picture, const std::shared_ptr<Surface> &target) {
    auto source = picture.source;
    auto region = picture.region;
    check(source && source->memory, "VPP source has no storage", VA_STATUS_ERROR_INVALID_SURFACE);
    if (!target->memory)
        target->memory = allocate_vpp_memory(impl_->render_fd, *target);
    auto &src = *source->memory;
    auto &dst = *target->memory;
    check(dst.origin != MemoryOrigin::Iris, "VPP cannot overwrite decoder reference storage",
          VA_STATUS_ERROR_INVALID_SURFACE);
    struct stat src_stat{}, dst_stat{};
    check(fstat(src.fd, &src_stat) == 0 && fstat(dst.fd, &dst_stat) == 0 &&
              (src_stat.st_dev != dst_stat.st_dev || src_stat.st_ino != dst_stat.st_ino),
          "VPP source and destination DMA-BUF alias", VA_STATUS_ERROR_INVALID_SURFACE);
#ifdef HAVE_VULKAN_VPP
    if (picture.tone_mapping) {
        check(tone_mapping_supported(), "Vulkan tone mapping unavailable",
              VA_STATUS_ERROR_UNSUPPORTED_FILTER);
        bool geometry =
            picture.rotation || region.width != target->width || region.height != target->height;
        if (!geometry) {
            impl_->vulkan->process(picture, *target);
            target->status = VA_STATUS_SUCCESS;
            return;
        }
        auto &intermediate = impl_->intermediate;
        if (!intermediate || intermediate->width != region.width ||
            intermediate->height != region.height) {
            auto resized = std::make_shared<Surface>();
            resized->width = region.width;
            resized->height = region.height;
            resized->fourcc = VA_FOURCC_NV12;
            resized->memory = allocate_vpp_memory(impl_->render_fd, *resized);
            intermediate = std::move(resized);
        }
        // Vulkan and CDSP share this allocation without CPU pixel access.
        impl_->vulkan->process(picture, *intermediate);
        source = intermediate;
        region.x = region.y = 0;
    }
#endif
    if (use_vulkan(false)) {
#ifdef HAVE_VULKAN_VPP
        auto transform = picture;
        transform.source = source;
        transform.region = region;
        impl_->vulkan->transform(transform, *target);
        target->status = VA_STATUS_SUCCESS;
        return;
#endif
    }
#ifdef HAVE_CDSP_VPP
    impl_->initialize_cdsp();
    auto in = import_vpp_frame(*source->memory, *source, impl_->api),
         out = import_vpp_frame(dst, *target, impl_->api);
    cdsp_vpp_crop crop{unsigned(region.x), unsigned(region.y), region.width, region.height};
    uint64_t dsp_us = 0;
    int e = impl_->api->transform(
        impl_->context, &in, source->fourcc == VA_FOURCC_P010 ? CDSP_VPP_P010 : CDSP_VPP_NV12,
        &crop, &out, target->fourcc == VA_FOURCC_P010 ? CDSP_VPP_P010 : CDSP_VPP_NV12,
        picture.rotation, &dsp_us);
    if (e) {
        trace("CDSP VPP processing failed: %d", e);
        throw Error(VA_STATUS_ERROR_OPERATION_FAILED, "CDSP VPP processing failed");
    }
    target->status = VA_STATUS_SUCCESS;
    trace("VPP CDSP %ux%u -> %ux%u rotation=%u src-fd=%d dst-fd=%d DSP=%llu us", region.width,
          region.height, target->width, target->height, picture.rotation * 90, src.fd, dst.fd,
          static_cast<unsigned long long>(dsp_us));
#else
    throw Error(VA_STATUS_ERROR_UNIMPLEMENTED, "processing requires CDSP VPP");
#endif
}
namespace {
OverlayParams overlay_parameters(const VppPicture &layer, const Surface &base,
                                 const Surface &target, const VppPicture &background) {
    const auto &src = *base.memory, &dst = *target.memory, &rgb = *layer.source->memory;
    auto r = layer.region, out = layer.output;
    OverlayParams p{};
    p.width = target.width;
    p.height = target.height;
    p.src_stride = src.stride;
    p.src_y = src.data_offset;
    p.src_uv = src.chroma();
    p.dst_stride = dst.stride;
    p.dst_y = dst.data_offset;
    p.dst_uv = dst.chroma();
    p.rgb_stride = rgb.stride;
    p.rgb_offset = rgb.data_offset + r.y * rgb.stride + r.x * 4;
    p.rgb_width = r.width;
    p.rgb_height = r.height;
    p.x = out.x;
    p.y = out.y;
    p.w = out.width;
    p.h = out.height;
    p.alpha = (layer.blend.flags & VA_BLEND_GLOBAL_ALPHA)
                  ? std::lround(layer.blend.global_alpha * 256)
                  : 256;
    p.flags = (target.fourcc == VA_FOURCC_P010 ? OVERLAY_P010 : 0) |
              (layer.source->fourcc == VA_FOURCC_BGRA ? OVERLAY_BGRA : 0) |
              (layer.blend.flags & VA_BLEND_PREMULTIPLIED_ALPHA ? OVERLAY_PREMULTIPLIED : 0);
    p.step_x = (p.rgb_width << 16) / p.w;
    p.step_y = (p.rgb_height << 16) / p.h;
    auto standard = background.color_standard;
    auto props = background.color_properties;
    double kr = 0.2126, kb = 0.0722;
    if (standard == VAProcColorStandardBT601 ||
        ((standard == VAProcColorStandardExplicit || standard == VAProcColorStandardNone) &&
         (props.matrix_coefficients == 5 || props.matrix_coefficients == 6))) {
        kr = 0.299;
        kb = 0.114;
    } else if (standard == VAProcColorStandardBT2020 ||
               ((standard == VAProcColorStandardExplicit || standard == VAProcColorStandardNone) &&
                props.matrix_coefficients == 9)) {
        kr = 0.2627;
        kb = 0.0593;
    } else {
        check((standard == VAProcColorStandardNone || standard == VAProcColorStandardBT709 ||
               standard == VAProcColorStandardExplicit) &&
                  (!props.matrix_coefficients || props.matrix_coefficients == 1 ||
                   props.matrix_coefficients == 2),
              "unsupported overlay output matrix", VA_STATUS_ERROR_UNIMPLEMENTED);
    }
    check(!props.color_range || props.color_range == VA_SOURCE_RANGE_REDUCED ||
              props.color_range == VA_SOURCE_RANGE_FULL,
          "unsupported overlay output range", VA_STATUS_ERROR_UNIMPLEMENTED);
    double kg = 1 - kr - kb;
    bool full = props.color_range == VA_SOURCE_RANGE_FULL;
    double ys = full ? 1 : 219.0 / 255, cs = full ? 1 : 224.0 / 255;
    double matrix[] = {kr * ys,
                       kg * ys,
                       kb * ys,
                       -kr * cs / (2 * (1 - kb)),
                       -kg * cs / (2 * (1 - kb)),
                       cs / 2,
                       cs / 2,
                       -kg * cs / (2 * (1 - kr)),
                       -kb * cs / (2 * (1 - kr))};
    for (unsigned i = 0; i < 9; ++i)
        p.matrix[i] = std::lround(matrix[i] * 16384);
    p.black = full ? 0 : 16;
    check(overlay_valid(&p, src.size, rgb.size, dst.size), "invalid overlay memory layout",
          VA_STATUS_ERROR_INVALID_SURFACE);
    return p;
}
} // namespace
void Vpp::process(const std::vector<VppPicture> &pictures, const std::shared_ptr<Surface> &target) {
    check(!pictures.empty(), "missing VPP base layer", VA_STATUS_ERROR_INVALID_PARAMETER);
    if (!target->memory)
        target->memory = allocate_vpp_memory(impl_->render_fd, *target);
    check(target->memory->origin != MemoryOrigin::Iris, "VPP cannot overwrite decoder references",
          VA_STATUS_ERROR_INVALID_SURFACE);
    struct stat dst_stat{};
    check(fstat(target->memory->fd, &dst_stat) == 0, "stat VPP output");
    for (const auto &picture : pictures) {
        check(picture.source && picture.source->memory, "VPP source has no storage",
              VA_STATUS_ERROR_INVALID_SURFACE);
        struct stat src_stat{};
        check(fstat(picture.source->memory->fd, &src_stat) == 0 &&
                  (src_stat.st_dev != dst_stat.st_dev || src_stat.st_ino != dst_stat.st_ino),
              "VPP source and destination DMA-BUF alias", VA_STATUS_ERROR_INVALID_SURFACE);
    }
    auto base = pictures[0].source;
    const auto &background = pictures[0];
    bool direct = !background.tone_mapping && !background.rotation && !background.region.x &&
                  !background.region.y && background.region.width == target->width &&
                  background.region.height == target->height && base->fourcc == target->fourcc;
    std::vector<OverlayParams> params;
    for (size_t i = 1; i < pictures.size(); ++i)
        params.push_back(overlay_parameters(pictures[i], direct && i == 1 ? *base : *target,
                                            *target, background));
    if (pictures.size() == 1) {
        process_single(background, target);
        return;
    }
    bool gpu = use_vulkan(true);
    if (!direct) {
        process_single(background, target);
        base = target;
    }
    for (size_t i = 1; i < pictures.size(); ++i) {
#ifdef HAVE_VULKAN_VPP
        if (gpu)
            impl_->vulkan->overlay(*base, *pictures[i].source, *target, params[i - 1]);
#endif
#ifdef HAVE_CDSP_VPP
        if (!gpu) {
            auto src = import_vpp_frame(*base->memory, *base, impl_->api);
            auto rgb =
                import_vpp_frame(*pictures[i].source->memory, *pictures[i].source, impl_->api);
            auto dst = import_vpp_frame(*target->memory, *target, impl_->api);
            uint64_t us = 0;
            int error = impl_->api->overlay(impl_->context, src.buffer, rgb.buffer, dst.buffer,
                                            &params[i - 1], &us);
            if (error)
                trace("CDSP overlay failed: %d", error);
            check(!error, "CDSP overlay failed");
            trace("VPP CDSP overlay %dx%d -> %ux%u DSP=%llu us", params[i - 1].w, params[i - 1].h,
                  target->width, target->height, (unsigned long long)us);
        }
#endif
        base = target;
    }
    target->status = VA_STATUS_SUCCESS;
}

#else
struct Vpp::Impl {};
Vpp::Vpp(int) {
    throw Error(VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT, "VPP was disabled at build time");
}
bool Vpp::tone_mapping_supported() {
    return false;
}
void Vpp::process(const std::vector<VppPicture> &, const std::shared_ptr<Surface> &) {
    throw Error(VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT, "VPP was disabled at build time");
}
#endif
Vpp::~Vpp() = default;
} // namespace irisva
