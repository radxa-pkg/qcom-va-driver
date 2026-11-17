// SPDX-License-Identifier: MIT
// Board test: c++ -std=c++17 prime-import.cpp -lva -lva-drm -o prime-import
#include <va/va.h>
#include <va/va_drm.h>
#include <va/va_drmcommon.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void require(bool ok, const char *message) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}
static void expect(VAStatus got, VAStatus wanted = VA_STATUS_SUCCESS) {
    if (got != wanted) {
        std::fprintf(stderr, "Expected %s, got %s\n", vaErrorStr(wanted), vaErrorStr(got));
        std::exit(1);
    }
}
static VAStatus import_surface(VADisplay display, unsigned format, unsigned memory_type,
                               void *descriptor, VASurfaceID *surface) {
    VASurfaceAttrib attrs[2]{};
    // Descriptor first also checks that parsing does not depend on attribute order.
    attrs[0].type = VASurfaceAttribExternalBufferDescriptor;
    attrs[0].flags = VA_SURFACE_ATTRIB_SETTABLE;
    attrs[0].value.type = VAGenericValueTypePointer;
    attrs[0].value.value.p = descriptor;
    attrs[1].type = VASurfaceAttribMemoryType;
    attrs[1].flags = VA_SURFACE_ATTRIB_SETTABLE;
    attrs[1].value.type = VAGenericValueTypeInteger;
    attrs[1].value.value.i = memory_type;
    return vaCreateSurfaces(display, format, 128, 128, surface, 1, attrs, 2);
}
static void test_format(VADisplay display, unsigned format, unsigned chroma_skew = 0) {
    bool p010 = format == VA_RT_FORMAT_YUV420_10;
    unsigned bytes = p010 ? 2 : 1;
    VASurfaceAttribExternalBuffers desc{};
    desc.pixel_format = p010 ? VA_FOURCC_P010 : VA_FOURCC_NV12;
    desc.width = desc.height = 128;
    desc.num_planes = 2;
    desc.pitches[0] = desc.pitches[1] = 256 * bytes;
    // Exercise padded rows, padded storage height, and a nonzero base offset.
    desc.offsets[0] = 128;
    // chroma_skew places chroma off the row grid, as an allocator that pads
    // luma to a page boundary does (Mesa's freedreno for linear NV12).
    desc.offsets[1] = desc.offsets[0] + desc.pitches[0] * 160 + chroma_skew;
    desc.data_size = (desc.offsets[1] + desc.pitches[1] * 64 + 4095) & ~4095u;
    int heap = open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
    require(heap >= 0, "open DMA heap");
    dma_heap_allocation_data allocation{};
    allocation.len = desc.data_size;
    allocation.fd_flags = O_RDWR | O_CLOEXEC;
    require(ioctl(heap, DMA_HEAP_IOCTL_ALLOC, &allocation) == 0, "allocate DMA-BUF");
    close(heap);
    uintptr_t handle = allocation.fd;
    desc.buffers = &handle;
    desc.num_buffers = 1;
    VASurfaceID surface;
    auto rejected = [&](VASurfaceAttribExternalBuffers bad) {
        expect(
            import_surface(display, format, VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME, &bad, &surface),
            VA_STATUS_ERROR_INVALID_PARAMETER);
    };
    auto bad = desc;
    bad.num_planes = 1;
    rejected(bad);
    bad = desc;
    bad.num_buffers = 2;
    rejected(bad);
    bad = desc;
    bad.buffers = nullptr;
    rejected(bad);
    bad = desc;
    uintptr_t invalid_fd = UINTPTR_MAX;
    bad.buffers = &invalid_fd;
    rejected(bad);
    const unsigned unsupported_flags[] = {VA_SURFACE_EXTBUF_DESC_ENABLE_TILING,
                                          VA_SURFACE_EXTBUF_DESC_PROTECTED};
    for (unsigned flags : unsupported_flags) {
        bad = desc;
        bad.flags = flags;
        rejected(bad);
    }
    bad = desc;
    bad.width = 127;
    rejected(bad);
    bad = desc;
    bad.pixel_format = p010 ? VA_FOURCC_NV12 : VA_FOURCC_P010;
    rejected(bad);
    bad = desc;
    bad.pitches[0] = bad.pitches[1] = 0;
    rejected(bad);
    bad = desc;
    bad.pitches[1] += 16;
    rejected(bad);
    bad = desc;
    bad.offsets[1] = desc.offsets[0] + desc.pitches[0] * 127;
    rejected(bad);
    bad = desc;
    bad.offsets[1] = UINT_MAX;
    rejected(bad);
    bad = desc;
    bad.data_size = desc.offsets[1];
    rejected(bad);
    bad = desc;
    bad.data_size += 4096; // Passes plane checks, but exceeds the actual DMA-BUF.
    rejected(bad);
    if (p010) {
        bad = desc;
        ++bad.offsets[0];
        ++bad.offsets[1];
        rejected(bad);
        bad = desc;
        ++bad.offsets[1]; // Chroma must stay aligned to a sample.
        rejected(bad);
    }

    expect(import_surface(display, format, VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME, &desc, &surface));
    for (unsigned flags : {VA_EXPORT_SURFACE_COMPOSED_LAYERS, VA_EXPORT_SURFACE_SEPARATE_LAYERS}) {
        VADRMPRIMESurfaceDescriptor exported{};
        expect(vaExportSurfaceHandle(display, surface, VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                                     VA_EXPORT_SURFACE_READ_WRITE | flags, &exported));
        require(exported.fourcc == desc.pixel_format && exported.num_objects == 1,
                "exported format and object count");
        struct stat original{}, shared{};
        require(!fstat(allocation.fd, &original) && !fstat(exported.objects[0].fd, &shared) &&
                    original.st_dev == shared.st_dev && original.st_ino == shared.st_ino,
                "import/export must share the original DMA-BUF");
        VASurfaceID reimported;
        expect(import_surface(display, format, VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2, &exported,
                              &reimported));
        close(exported.objects[0].fd);
        VAImage image;
        expect(vaDeriveImage(display, reimported, &image));
        require(image.format.fourcc == desc.pixel_format && image.num_planes == 2 &&
                    image.pitches[0] == desc.pitches[0] && image.pitches[1] == desc.pitches[1] &&
                    image.offsets[0] == desc.offsets[0] && image.offsets[1] == desc.offsets[1],
                "imported image preserves the plane layout");
        void *data;
        expect(vaMapBuffer(display, image.buf, &data));
        auto *pixels = static_cast<unsigned char *>(data);
        pixels[image.offsets[0]] = 42;
        pixels[image.offsets[1]] = 73;
        expect(vaUnmapBuffer(display, image.buf));
        expect(vaDestroyImage(display, image.image_id));
        expect(vaDestroySurfaces(display, &reimported, 1));
    }
    require(fcntl(allocation.fd, F_GETFD) >= 0, "driver must not close caller's FD");
    close(allocation.fd);
    // The imported surface retains its own FD and the same pixel storage.
    VAImage image;
    expect(vaDeriveImage(display, surface, &image));
    void *data;
    expect(vaMapBuffer(display, image.buf, &data));
    auto *pixels = static_cast<unsigned char *>(data);
    require(pixels[image.offsets[0]] == 42 && pixels[image.offsets[1]] == 73,
            "shared pixels survive closing the caller's FD");
    expect(vaUnmapBuffer(display, image.buf));
    expect(vaDestroyImage(display, image.image_id));
    expect(vaDestroySurfaces(display, &surface, 1));
}
int main() {
    int fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
    require(fd >= 0, "open DRM device");
    auto display = vaGetDisplayDRM(fd);
    int major, minor;
    expect(vaInitialize(display, &major, &minor));
    for (VAProfile profile : {VAProfileHEVCMain, VAProfileHEVCMain10}) {
        VAConfigID config;
        expect(vaCreateConfig(display, profile, VAEntrypointVLD, nullptr, 0, &config));
        unsigned n = 0;
        expect(vaQuerySurfaceAttributes(display, config, nullptr, &n));
        std::vector<VASurfaceAttrib> attrs(n);
        expect(vaQuerySurfaceAttributes(display, config, attrs.data(), &n));
        bool found = false;
        for (const auto &a : attrs) {
            if (a.type == VASurfaceAttribMemoryType) {
                unsigned types = VA_SURFACE_ATTRIB_MEM_TYPE_VA |
                                 VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME |
                                 VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2;
                found = (a.value.value.i & types) == types;
            }
        }
        require(found, "advertise native, PRIME1 and PRIME2 surfaces");
        expect(vaDestroyConfig(display, config));
    }
    test_format(display, VA_RT_FORMAT_YUV420);
    test_format(display, VA_RT_FORMAT_YUV420_10);
    test_format(display, VA_RT_FORMAT_YUV420, 96);
    test_format(display, VA_RT_FORMAT_YUV420_10, 96);
    expect(vaTerminate(display));
    close(fd);
    std::puts("PRIME1/PRIME2 NV12/P010 import tests passed");
}
