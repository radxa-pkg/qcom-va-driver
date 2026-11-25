// SPDX-License-Identifier: MIT
#include "internal.hpp"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl32.h>
#include <GLES2/gl2ext.h>
#include <gbm.h>
#include <drm_fourcc.h>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#include <climits>
#include <condition_variable>
#include <future>
#include <thread>

namespace irisva {
namespace {
// Adreno misplaces linear copies whose plane base is not 64-byte aligned
// without reporting an error, and Mesa rejects imports whose pitch is not a
// multiple of 64. Driver and V4L2 allocations always comply; an imported
// DMA-BUF may not, and is copied on the CPU instead.
bool gpu_copy_aligned(const Memory &m) {
    return !(m.stride % 64) && !(m.data_offset % 64) && !(m.chroma() % 64);
}
class GpuCopy {
    int fd_ = -1;
    gbm_device *gbm_ = nullptr;
    EGLDisplay display_ = EGL_NO_DISPLAY;
    EGLContext context_ = EGL_NO_CONTEXT;
    bool initialized_ = false;
    PFNEGLCREATEIMAGEKHRPROC create_image_ = nullptr;
    PFNEGLDESTROYIMAGEKHRPROC destroy_image_ = nullptr;
    PFNGLEGLIMAGETARGETTEXTURE2DOESPROC bind_image_ = nullptr;
    struct View {
        GpuCopy &owner;
        std::weak_ptr<Memory> memory;
        unsigned width, height, stride, storage_height, offset, skew, fourcc;
        EGLImageKHR images[2] = {EGL_NO_IMAGE_KHR, EGL_NO_IMAGE_KHR};
        GLuint textures[2] = {};
        unsigned columns[2] = {}, rows[2] = {};
        View(GpuCopy &gpu, const std::shared_ptr<Memory> &m, unsigned w, unsigned h)
            : owner(gpu), memory(m), width(w), height(h), stride(m->stride),
              storage_height(m->storage_height), offset(m->data_offset), skew(m->chroma_skew),
              fourcc(m->fourcc) {}
        ~View() {
            glDeleteTextures(2, textures);
            for (auto image : images)
                if (image != EGL_NO_IMAGE_KHR)
                    owner.destroy_image_(owner.display_, image);
        }
        bool matches(const std::shared_ptr<Memory> &m, unsigned w, unsigned h) const {
            return memory.lock() == m && width == w && height == h && stride == m->stride &&
                   storage_height == m->storage_height && offset == m->data_offset &&
                   skew == m->chroma_skew && fourcc == m->fourcc;
        }
    };
    // Weak ownership is essential: retaining CAPTURE Memory would prevent requeue().
    std::map<Memory *, std::unique_ptr<View>> views_;
    void cleanup() noexcept {
        if (context_ != EGL_NO_CONTEXT) {
            glFinish();
            views_.clear();
            eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            eglDestroyContext(display_, context_);
        }
        if (initialized_)
            eglTerminate(display_);
        if (gbm_)
            gbm_device_destroy(gbm_);
        if (fd_ >= 0)
            close(fd_);
        eglReleaseThread();
    }
    View &view(const std::shared_ptr<Memory> &m, unsigned w, unsigned h) {
        auto it = views_.find(m.get());
        if (it != views_.end()) {
            if (it->second->matches(m, w, h))
                return *it->second;
            views_.erase(it);
        }
        check(w && h && w <= m->width && h <= m->height && h <= m->storage_height && m->fd >= 0 &&
                  m->stride <= INT_MAX,
              "GPU copy invalid buffer layout");
        auto v = std::make_unique<View>(*this, m, w, h);
        unsigned bytes = m->fourcc == VA_FOURCC_P010 ? 2 : 1;
        for (unsigned p = 0; p < 2; ++p) {
            // RGBA8 views transport four raw bytes per texel; no YUV conversion.
            uint64_t row_bytes = uint64_t(p ? ((w + 1) & ~1u) : w) * bytes;
            unsigned columns = (row_bytes + 3) / 4, rows = p ? (h + 1) / 2 : h;
            uint64_t offset = p ? uint64_t(m->chroma()) : m->data_offset;
            check(uint64_t(columns) * 4 <= m->stride && offset <= INT_MAX &&
                      offset + uint64_t(rows - 1) * m->stride + uint64_t(columns) * 4 <= m->size,
                  "GPU copy packed plane exceeds buffer");
            EGLint attributes[] = {EGL_WIDTH,
                                   EGLint(columns),
                                   EGL_HEIGHT,
                                   EGLint(rows),
                                   EGL_LINUX_DRM_FOURCC_EXT,
                                   DRM_FORMAT_ABGR8888,
                                   EGL_DMA_BUF_PLANE0_FD_EXT,
                                   m->fd,
                                   EGL_DMA_BUF_PLANE0_OFFSET_EXT,
                                   EGLint(offset),
                                   EGL_DMA_BUF_PLANE0_PITCH_EXT,
                                   EGLint(m->stride),
                                   EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
                                   0,
                                   EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
                                   0,
                                   EGL_IMAGE_PRESERVED_KHR,
                                   EGL_TRUE,
                                   EGL_NONE};
            v->images[p] =
                create_image_(display_, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, nullptr, attributes);
            check(v->images[p] != EGL_NO_IMAGE_KHR, "GPU copy DMA-BUF import failed");
            glGenTextures(1, &v->textures[p]);
            glBindTexture(GL_TEXTURE_2D, v->textures[p]);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            bind_image_(GL_TEXTURE_2D, v->images[p]);
            check(glGetError() == GL_NO_ERROR, "GPU copy texture import failed");
            v->columns[p] = columns;
            v->rows[p] = rows;
        }
        trace("GPU copy imported buffer=%p %ux%u offset=%u", static_cast<void *>(m.get()), w, h,
              m->data_offset);
        auto &result = *v;
        views_.emplace(m.get(), std::move(v));
        return result;
    }

  public:
    explicit GpuCopy(int render_fd) {
        try {
            // Private GBM identity gives this engine its own EGLDisplay. Terminating
            // it must not terminate an application's shared surfaceless display.
            // Reuse the VA display's device, including inside a GPU sandbox.
            fd_ = fcntl(render_fd, F_DUPFD_CLOEXEC, 0);
            check(fd_ >= 0, "GPU copy duplicate VA DRM fd failed");
            gbm_ = gbm_create_device(fd_);
            check(gbm_, "GPU copy GBM device failed");
            auto get_display = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
                eglGetProcAddress("eglGetPlatformDisplayEXT"));
            create_image_ =
                reinterpret_cast<PFNEGLCREATEIMAGEKHRPROC>(eglGetProcAddress("eglCreateImageKHR"));
            destroy_image_ = reinterpret_cast<PFNEGLDESTROYIMAGEKHRPROC>(
                eglGetProcAddress("eglDestroyImageKHR"));
            bind_image_ = reinterpret_cast<PFNGLEGLIMAGETARGETTEXTURE2DOESPROC>(
                eglGetProcAddress("glEGLImageTargetTexture2DOES"));
            check(get_display && create_image_ && destroy_image_ && bind_image_,
                  "GPU copy EGL entrypoints missing");
            display_ = get_display(EGL_PLATFORM_GBM_KHR, gbm_, nullptr);
            check(display_ != EGL_NO_DISPLAY && eglInitialize(display_, nullptr, nullptr),
                  "GPU copy EGL initialize failed");
            initialized_ = true;
            check(eglBindAPI(EGL_OPENGL_ES_API), "GPU copy EGL bind API failed");
            EGLint attributes[] = {EGL_CONTEXT_MAJOR_VERSION_KHR, 3, EGL_CONTEXT_MINOR_VERSION_KHR,
                                   2, EGL_NONE};
            context_ = eglCreateContext(display_, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, attributes);
            check(context_ != EGL_NO_CONTEXT &&
                      eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, context_),
                  "GPU copy GLES 3.2 context failed");
            auto renderer = reinterpret_cast<const char *>(glGetString(GL_RENDERER));
            check(renderer && !strstr(renderer, "llvmpipe") && !strstr(renderer, "softpipe") &&
                      !strstr(renderer, "Software"),
                  "GPU copy requires hardware rendering");
            trace("GPU copy initialized renderer=%s (private GBM display, worker thread)",
                  renderer);
        } catch (...) {
            cleanup();
            throw;
        }
    }
    ~GpuCopy() {
        cleanup();
    }
    void copy(const std::shared_ptr<Memory> &dst, const std::shared_ptr<Memory> &src, unsigned w,
              unsigned h) {
        check(dst != src && dst->fourcc == src->fourcc &&
                  (src->fourcc == VA_FOURCC_NV12 || src->fourcc == VA_FOURCC_P010),
              "GPU copy format mismatch", VA_STATUS_ERROR_INVALID_IMAGE_FORMAT);
        for (auto it = views_.begin(); it != views_.end();) {
            if (it->second->memory.expired())
                it = views_.erase(it);
            else
                ++it;
        }
        wait_surface_access(*src, POLLIN);
        wait_surface_access(*dst, POLLOUT);
        auto &s = view(src, w, h), &d = view(dst, w, h);
        for (unsigned p = 0; p < 2; ++p)
            glCopyImageSubData(s.textures[p], GL_TEXTURE_2D, 0, 0, 0, 0, d.textures[p],
                               GL_TEXTURE_2D, 0, 0, 0, 0, s.columns[p], s.rows[p], 1);
        // The persistent export consumer may render without another VA sync call.
        glFinish();
        check(glGetError() == GL_NO_ERROR, "GPU copy failed; try IRIS_VAAPI_COPY=cpu");
    }
};
} // namespace

struct SurfaceCopier::Impl {
    bool cpu;
    std::mutex mutex;
    std::condition_variable wake;
    bool stop = false;
    std::packaged_task<void(GpuCopy *, std::exception_ptr)> task;
    std::thread worker;
    explicit Impl(int render_fd) {
        const char *mode = std::getenv("IRIS_VAAPI_COPY");
        check(!mode || !strcmp(mode, "gpu") || !strcmp(mode, "cpu"),
              "IRIS_VAAPI_COPY must be gpu or cpu", VA_STATUS_ERROR_INVALID_PARAMETER);
        cpu = mode && !strcmp(mode, "cpu");
        trace("surface copy selected mode=%s", cpu ? "cpu" : "gpu");
        if (!cpu)
            worker = std::thread([this, render_fd] {
                // All EGL calls, including teardown, stay on this thread. VA entry
                // points may migrate between threads with arbitrary caller GL state.
                std::unique_ptr<GpuCopy> gpu;
                std::exception_ptr init_error;
                try {
                    gpu = std::make_unique<GpuCopy>(render_fd);
                } catch (...) {
                    init_error = std::current_exception();
                }
                for (;;) {
                    std::unique_lock<std::mutex> lock(mutex);
                    wake.wait(lock, [&] { return stop || task.valid(); });
                    if (stop)
                        break;
                    auto current = std::move(task);
                    lock.unlock();
                    // The packaged task conveys errors without throwing out of the worker.
                    current(gpu.get(), init_error);
                }
            });
    }
    ~Impl() {
        if (worker.joinable()) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                stop = true;
            }
            wake.notify_one();
            worker.join();
        }
    }
};

SurfaceCopier::SurfaceCopier(int render_fd) : render_fd_(render_fd) {}
SurfaceCopier::~SurfaceCopier() = default;
void SurfaceCopier::copy(const std::shared_ptr<Memory> &dst, const std::shared_ptr<Memory> &src,
                         unsigned w, unsigned h) {
    try {
        if (!impl_)
            impl_ = std::make_unique<Impl>(render_fd_);
        if (impl_->cpu || !gpu_copy_aligned(*src) || !gpu_copy_aligned(*dst)) {
            if (!impl_->cpu)
                trace("surface copy mode=cpu %ux%u: layout not 64-byte aligned for GPU copy", w, h);
            copy_surface(*dst, *src, w, h);
            return;
        }
        auto start = std::chrono::steady_clock::now();
        std::packaged_task<void(GpuCopy *, std::exception_ptr)> task(
            [&](GpuCopy *gpu, std::exception_ptr error) {
                if (error)
                    std::rethrow_exception(error);
                gpu->copy(dst, src, w, h);
            });
        auto done = task.get_future();
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            impl_->task = std::move(task);
        }
        impl_->wake.notify_one();
        done.get();
        trace("surface copy mode=gpu %ux%u fourcc=%#x capture=%u time-us=%lld", w, h, src->fourcc,
              src->index,
              (long long)std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now() - start)
                  .count());
    } catch (const Error &) {
        throw;
    } catch (const std::exception &e) {
        throw Error(VA_STATUS_ERROR_OPERATION_FAILED, std::string("surface copy: ") + e.what());
    }
}
} // namespace irisva
