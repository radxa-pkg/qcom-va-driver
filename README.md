# qcom-va-driver

Experimental VA-API backend for Qualcomm Iris via Linux V4L2 stateful M2M,
providing hardware decoding and encoding on **SC8280XP**, tested on
**Radxa Dragon Q8B**. AV1 decoding is also tested on **X1E80100**.
This is not a generic backend for all V4L2 devices.

## Support

| Codec | Decoding | Encoding |
| --- | --- | --- |
| H.264 / AVC | Constrained Baseline, Main, High (8-bit) | Same profiles (8-bit) |
| H.265 / HEVC | Main, Main10 (8/10-bit) | Main (8-bit) |
| VP9 | Profiles 0 and 2 (8/10-bit) | — |
| AV1 | Main (8/10-bit), when exposed by Iris | — |

- Progressive 4:2:0 video only. Decoding tested with FFmpeg, GStreamer, mpv,
  VLC, Chromium, and Kodi; encoding with FFmpeg, GStreamer and Sunshine.
- AV1: tested with FFmpeg and GStreamer from 720p to 4K; no film grain or
  large-scale tiles. Full reference maps can require an extra hidden decode
  because VA-API omits AV1 reference refresh flags.
  X1E80100 tested with Iris firmware `vfw-3.1:rel00101` from Windows qcdx8380
  driver version 31.0.174.0; the older tested firmware had intermittent decoder
  startup failures, including with direct V4L2.
- Encoding: CQP/CBR/VBR, even dimensions from 128×128 to 3840×2160, I/P frames
  with one reference; no B frames. I-frame requests produce IDR frames.
  HEVC uses Main tier; Main10 encoding is unsupported.
- Compatible DMA-BUF paths avoid raw-frame copies; cached surfaces may need
  GPU or CPU copying. PRIME1/PRIME2 import accepts single-buffer linear
  NV12/P010/BGRA/RGBA; export uses PRIME2.
- Optional CDSP/Vulkan VPP: NV12/P010 scaling, cropping, rotation, bit-depth
  conversion and RGB overlays with DMA-BUF zero-copy.
- Vulkan also supports HDR10 P010 to SDR NV12 tone mapping.
- Identified decode errors produce gray frames with the Iris error-timestamp
  patch below; stalls remain errors.

## Performance

HEVC Main (8-bit) average FPS on **Radxa Dragon Q8B — SC8280XP, Iris HFI Gen2**,
using FFmpeg 9.0.1 and GStreamer 1.28.6.

| Pipeline | Operation | 720p (1280×720) | 1080p (1920×1080) | 4K (3840×2160) |
| --- | --- | ---: | ---: | ---: |
| FFmpeg VA-API | Decode | 1111 | 806 | 297 |
| FFmpeg VA-API | Encode | 1289 | 714 | 206 |
| GStreamer VA | Decode | 1088 | 823 | 278 |
| GStreamer VA | Encode | 1285 | 709 | 206 |
| GStreamer direct V4L2 | Decode | 1199 | 882 | 330 |
| GStreamer direct V4L2 | Encode | 1287 | 724 | 210 |

Synthetic I/P streams, GOP 60, VBR at 5/10/20 Mbps respectively. Encoding includes
CPU NV12 upload; FFmpeg uses **`async_depth=4`**. Decoding excludes CPU download
and rendering. Three-run averages, except 4K encoding (one run); actual performance
depends on content and pipeline.

## Requirements

- Linux with Qualcomm Iris stateful and MSM DRM drivers, plus access to the
  Iris video devices and DRM render node.
- Kernel changes in [patches/](patches/), unless already present: decode-order
  output, larger CAPTURE pools, encoder deblocking controls, error timestamps,
  and NOSHOW buffer recycling (required for AV1).
- C++17 compiler, Meson ≥ 0.61, Ninja, pkg-config, libva ≥ 1.10 and libdrm
  development libraries. GPU-copy also requires EGL, GLES 3.2, and GBM.
- CDSP VPP is enabled by default and requires a Linux ARM64 build machine,
  GHC and `syb`.
  Use `-Dcdsp_vpp=disabled` to skip its dependencies and downloads.

## Build and install

The project and distribution package were previously named `libva-v4l2`.
The driver is installed as `qcom_drv_video.so`, with an automatic `msm` alias.
The legacy `v4l2` driver name is installed as a compatibility symlink unless
configured with `-Dv4l2_symlink=false`. Existing environment configuration
filenames are retained.

On Arch Linux:

```sh
sudo pacman -S --needed base-devel meson ninja pkgconf libva libdrm mesa libglvnd
python3 packaging/make-dist.py
cd packaging/arch
makepkg -si
```

Regenerate the archive after editing packaged sources. Alternatively, use Meson:

```sh
meson setup build --buildtype=release --prefix=/usr -Dlibdir=lib
meson compile -C build
sudo meson install -C build
```

- `-Dgpu_copy=auto` (default): enable GPU copying when dependencies are available;
  `disabled` uses CPU copying when needed, `enabled` requires dependencies.
  DMA-BUF zero-copy paths are unaffected. Distribution packages enable GPU-copy.

Adjust `libdir` if drivers belong somewhere other than `/usr/lib/dri`.
The installed MSM alias selects this backend automatically on the target platform.
Check installation with `vainfo` (`libva-utils` on Arch):

```sh
vainfo --display drm --device /dev/dri/renderD128
```

## CDSP video processing

Meson downloads pinned QAIC and Hexagon Open Access **19.0.02**, using system
FastRPC or a source fallback. No full SDK is needed. Arch packaging enables CDSP;
Debian disables it. See [CDSP build and runtime requirements](src/cdsp/README.md).

Both CDSP and Vulkan support NV12/P010 cropping, scaling, conversion and
clockwise 90°/180°/270° rotation, including combined operations. Dimensions
and crop coordinates must be even, maximum dimension 8192, reduction up to
20:1, both axes scaling in the same direction after
rotation. Color matrix/range is preserved; the base layer fills the output.
No mirroring or interlacing. CDSP workers follow the available HVX count,
up to eight.

Select `IRIS_VAAPI_VPP=auto|cdsp|vulkan` at runtime or `-Dvpp_backend=...`
at build time; the environment takes precedence. `auto` prefers CDSP for
geometry/conversion and Vulkan for overlays, falling back to the available
backend. `IRIS_VAAPI_OVERLAY` overrides only overlay selection. HDR tone mapping
always requires Vulkan.

```sh
ffmpeg -hwaccel vaapi -hwaccel_device /dev/dri/renderD128 \
  -hwaccel_output_format vaapi -i input.mp4 \
  -vf scale_vaapi=w=3840:h=2160:format=nv12 -c:v hevc_vaapi -bf 0 output.mp4
```

## Vulkan video processing

Use `tonemap_vaapi=format=nv12` for limited-range HDR10 P010 (BT.2020/PQ) to
SDR NV12 (BT.709). Scaling/rotation can use either backend.

Both backends support `overlay_vaapi`: BGRA/RGBA layers over NV12/P010, including
crop, bilinear scaling, clipping, global and per-pixel alpha. Processing shares
DMA-BUFs without CPU pixel copies. YUV overlays and luma keying are unsupported.

`-Dvulkan_vpp=enabled` (default: `auto`) requires Vulkan ≥ 1.3 headers/loader
and `glslangValidator`. Runtime needs a compatible Vulkan driver; tested with
Mesa Turnip on Adreno 690. Both VPP backends use `/dev/dma_heap/system`, falling
back to coherent MSM GEM on failure, never write-combined GEM.

## Usage

Playback:

```sh
mpv --hwdec=vaapi input.mp4
```

H.264 encoding (for HEVC, replace the codec/profile with `hevc_vaapi`/`main`):

```sh
ffmpeg -vaapi_device /dev/dri/renderD128 -i input.mp4 -an \
  -vf format=nv12,hwupload -c:v h264_vaapi -profile:v high \
  -bf 0 -g 60 -rc_mode CBR -b:v 6M -async_depth 4 output.mp4
```

Hardware transcoding of supported 8-bit input:

```sh
ffmpeg -hwaccel vaapi -hwaccel_device /dev/dri/renderD128 \
  -hwaccel_output_format vaapi -i input.mp4 -an \
  -c:v h264_vaapi -bf 0 -qp 24 -async_depth 4 output.mp4
```

- MP4 and raw H.264 output are supported; for Matroska, encode to raw H.264
  first and remux. FFmpeg packed-header warnings are expected because firmware
  produces the headers. Custom VUI/SEI metadata is not forwarded.
- GStreamer: set `GST_VA_ALL_DRIVERS=1` to enable this driver's `va` elements.
- Sunshine/Moonlight: use **H.264 / SDR**; HEVC/AV1 probes may fail while
  H.264 remains available.

## Configuration and development

| Variable | Purpose |
| --- | --- |
| `LIBVA_DRIVER_NAME=qcom` | Explicitly select this backend. |
| `LIBVA_DRIVERS_PATH=/path/to/build` | Load an uninstalled build. |
| `IRIS_VAAPI_DEVICE` / `IRIS_VAAPI_ENCODER_DEVICE` | Override video-device discovery. |
| `IRIS_VAAPI_COPY=gpu` or `cpu` | Default: GPU when built in, otherwise CPU. Explicit GPU requires GPU-copy support. Imports whose pitch or plane offsets are not 64-byte aligned are always copied on the CPU. |
| `IRIS_VAAPI_DEBUG=1` | Enable diagnostics. |

Use clang-format 22 with the checked-in `.clang-format` (Meson targets: `format`,
`format-check`). Hardware test instructions are in [tests/](tests/).

## License

[MIT](LICENSE) for the backend; GPL-2.0-only for bundled kernel patches.
External test frameworks and media retain their own licenses.
