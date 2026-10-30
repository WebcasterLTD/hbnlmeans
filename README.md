# hbnlmeans

HandBrake's NLMeans denoiser as a standalone library, **libhbnlmeans**, plus an
FFmpeg filter, `hbnlmeans`, that uses it. It works the same way zscale uses
zimg and vidstab uses vid.stab: install the library, add the filter to FFmpeg's
source, configure with `--enable-libhbnlmeans`.

The output is identical, byte for byte, to HandBrake's own NLMeans code for
every preset and tune, at 8 and 10 bit and at odd frame sizes
(`test/compare-handbrake.run` checks this).

FFmpeg already has a filter called `nlmeans`. That one is a different, spatial-only
implementation. `hbnlmeans` is HandBrake's version: it also searches the
following frames, and it has HandBrake's presets, tunes and prefilters.

## Install

### 1. The library

```bash
git clone https://github.com/WebcasterLTD/hbnlmeans.git
cd hbnlmeans
make -j"$(nproc)"
sudo make install
sudo ldconfig
pkg-config --modversion hbnlmeans
```

`make install PREFIX=/some/dir` installs elsewhere, and `SHARED=0` installs
the static library only. `make check` runs a quick self test.

### 2. FFmpeg (8.0 or newer)

```bash
cd /path/to/ffmpeg-source
/path/to/hbnlmeans/ffmpeg/add-to-ffmpeg.run .
./configure --enable-gpl --enable-libhbnlmeans   # plus your usual options
make -j"$(nproc)"
```

`add-to-ffmpeg.run` copies `ffmpeg/vf_hbnlmeans.c` into `libavfilter/` and adds
the lines FFmpeg needs to `configure`, `libavfilter/allfilters.c` and
`libavfilter/Makefile`. Running it twice is harmless. The library is GPL v2, so
FFmpeg must be configured with `--enable-gpl`.

## Use

```bash
# HandBrake preset and tune
ffmpeg -i in.mkv -vf hbnlmeans=preset=medium:tune=film out.mkv

# HandBrake custom string, pasted as is
ffmpeg -i in.mkv -vf hbnlmeans=y-strength=6:y-origin-tune=0.8:cb-strength=8 out.mkv
```

| Option | Meaning |
|---|---|
| `preset` | `ultralight`, `light`, `medium`, `strong` (default `custom`: only explicit values) |
| `tune` | `none`, `film`, `grain`, `highmotion`, `animation`, `tape`, `sprite` (with a preset) |
| `y-` / `cb-` / `cr-` `strength` | averaging weight decay, larger is smoother (0–63) |
| `…-origin-tune` | weight of the pixel's own patch, 0.01–1; lower for noisier sources and animation |
| `…-patch-size` | patch width in pixels, odd, up to 15 |
| `…-range` | search window width, odd; larger is better and much slower |
| `…-frame-count` | how many frames to search, 1–32; too many smears motion |
| `…-prefilter` | sum of the values below |

Explicit values override the preset. Unset values cascade like HandBrake: Cr
takes Cb, Cb takes Y, and Y falls back to `6:1:7:3:2:0` (strength, origin tune,
patch, range, frames, prefilter).

Prefilter values: 1 mean 3×3, 2 mean 5×5, 4 median 3×3, 8 median 5×5, 16 CSM 3×3,
32 CSM 5×5, 256 reduce by 25%, 512 reduce by 50%, 1024 edge boost, 2048 passthru
(the prefilter is the output, no NLMeans). Example: 1281 = mean 3×3, reduced by
25%, with edge boost.

Each frame is compared with the frames that follow it, so the filter holds back
`frame-count − 1` frames. It runs on all CPU threads (FFmpeg slice threading) and
uses SSE4.1 where available. Planar YUV, 8 to 16 bit.

## Using the library directly

```c
#include <hbnlmeans.h>

HBNLMParams p;
hbnlm_params_init(&p);
hbnlm_params_preset(&p, "medium", "film");
HBNLMeans *nlm = hbnlm_create(&p, width, height, 1, 1, 8);   /* 4:2:0, 8-bit */

/* queue frames; once hbnlm_frames_needed() are queued, output the head */
hbnlm_push(nlm, src_planes, src_strides);
if (hbnlm_queued(nlm) == hbnlm_frames_needed(nlm)) {
    hbnlm_filter(nlm, dst_planes, dst_strides);   /* or hbnlm_filter_slice per thread */
    hbnlm_pop(nlm);
}
/* at the end of the stream, keep filtering and popping until nothing is queued */
hbnlm_free(&nlm);
```

See `include/hbnlmeans.h` for the whole API.

## How it was checked

`test/compare-handbrake.run [ffmpeg] [video]` downloads HandBrake's current
`libhb` sources, builds a reference program from its NLMeans code unmodified,
and compares it with `ffmpeg -vf hbnlmeans` byte for byte. It covers all tunes,
the prefilters, passthru, a 15-pixel patch with an 11-pixel range, an odd frame
size (1917×1077) and 10-bit.

One deliberate difference, with prefilters only: HandBrake reads the current
frame's prefiltered image before it prefilters that frame. Depending on thread
timing, some frames then weigh patches on the unprefiltered image. The result
changes from run to run, so it can't be matched. hbnlmeans always prefilters
first, and the test compares against HandBrake's code with that order fixed.
Without prefilters (all presets and tunes) there is no difference.

Older HandBrake releases, such as 1.3.x, treated the pixels nearest the frame
edge differently. Inside the frame they match this filter too.

## License

GPL v2, like HandBrake, where this code comes from (`libhb/nlmeans.c`,
`libhb/templates/nlmeans_template.c`, `libhb/nlmeans_x86.c`, and the presets in
`libhb/param.c`). Copyright (c) 2013 Dirk Farin, (c) 2003-2026 HandBrake Team.
The library packaging, the FFmpeg filter, the scripts and the tests are
Copyright (c) 2026 Corneliu Ticu, under the same license. See `COPYING`.
This project is not affiliated with HandBrake.

## Author

[corneliu.ticu@gmail.com](mailto:corneliu.ticu@gmail.com)

Questions, ideas or interest in this project: write to me directly.
