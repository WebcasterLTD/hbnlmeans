/*
 * libhbnlmeans - HandBrake's NLMeans denoiser as a standalone library
 *
 * Copyright (c) 2013 Dirk Farin
 * Copyright (c) 2003-2026 HandBrake Team
 * Library packaging (c) 2026 Corneliu Ticu
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2, as published
 * by the Free Software Foundation. See COPYING.
 */

#ifndef HBNLMEANS_H
#define HBNLMEANS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HBNLM_VERSION_MAJOR 1
#define HBNLM_VERSION_MINOR 0
#define HBNLM_VERSION_PATCH 0

/* Prefilter modes; add them up. */
#define HBNLM_PREFILTER_MEAN3X3     1
#define HBNLM_PREFILTER_MEAN5X5     2
#define HBNLM_PREFILTER_MEDIAN3X3   4
#define HBNLM_PREFILTER_MEDIAN5X5   8
#define HBNLM_PREFILTER_CSM3X3     16
#define HBNLM_PREFILTER_CSM5X5     32
#define HBNLM_PREFILTER_REDUCE25  256
#define HBNLM_PREFILTER_REDUCE50  512
#define HBNLM_PREFILTER_EDGEBOOST 1024
#define HBNLM_PREFILTER_PASSTHRU  2048  /* prefilter is the output, no NLMeans */

/*
 * Per-plane settings, index 0 = Y, 1 = Cb, 2 = Cr. -1 means unset.
 * Unset values cascade like HandBrake: Cr takes Cb, Cb takes Y,
 * Y falls back to strength 6, origin tune 1, patch 7, range 3, frames 2,
 * prefilter 0.
 */
typedef struct HBNLMParams {
    double strength[3];    /* averaging weight decay, larger is smoother, 0..63 */
    double origin_tune[3]; /* weight of the origin patch, 0.01..1 */
    int    patch_size[3];  /* patch width in pixels, odd, 1..15 */
    int    range[3];       /* spatial search window width, odd */
    int    frame_count[3]; /* temporal search depth in frames, 1..32 */
    int    prefilter[3];   /* HBNLM_PREFILTER_* sum */
} HBNLMParams;

/* Library version as (major << 16) | (minor << 8) | patch. */
unsigned hbnlm_version(void);

/* Set every value to -1 (unset). */
void hbnlm_params_init(HBNLMParams *p);

/*
 * Fill the Y and Cb values that are still unset from a HandBrake preset
 * ("ultralight", "light", "medium", "strong") and tune ("none", "film",
 * "grain", "highmotion", "animation", "tape", "sprite"; NULL = "none").
 * Returns 0, or -1 for an unknown preset or tune.
 */
int hbnlm_params_preset(HBNLMParams *p, const char *preset, const char *tune);

/*
 * Parse a HandBrake custom string, e.g. "y-strength=6:y-origin-tune=0.8:cb-strength=8".
 * Keys: {y,cb,cr}-{strength,origin-tune,patch-size,range,frame-count,prefilter}.
 * Returns 0, or -1 on an unknown key or bad value.
 */
int hbnlm_params_parse(HBNLMParams *p, const char *settings);

typedef struct HBNLMeans HBNLMeans;

/*
 * Create a denoiser for planar YUV frames.
 * log2_chroma_w/h: chroma subsampling (4:2:0 = 1,1; 4:2:2 = 1,0; 4:4:4 = 0,0).
 * depth: bits per sample, 8..16. Above 8, samples are 16-bit native endian.
 * Returns NULL on bad arguments or out of memory.
 */
HBNLMeans *hbnlm_create(const HBNLMParams *p, int width, int height,
                        int log2_chroma_w, int log2_chroma_h, int depth);

void hbnlm_free(HBNLMeans **ctx);

/* The values actually used, after cascading, defaults and clamping. */
void hbnlm_get_params(const HBNLMeans *ctx, HBNLMParams *out);

/*
 * Each frame is compared with the frames that follow it. This is how many
 * frames must be queued before the head frame can be output (the largest
 * frame count). At the end of the stream, output what is queued anyway.
 */
int hbnlm_frames_needed(const HBNLMeans *ctx);

/* Number of frames currently queued. */
int hbnlm_queued(const HBNLMeans *ctx);

/*
 * Queue a copy of a frame. src/stride: the three planes, stride in bytes.
 * Fails (-EAGAIN) when hbnlm_frames_needed() frames are already queued.
 * Returns 0 or a negative errno value.
 */
int hbnlm_push(HBNLMeans *ctx, const uint8_t *const src[3], const ptrdiff_t stride[3]);

/*
 * Denoise the head frame into dst, rows of slice `job` out of `nb_jobs`.
 * Distinct jobs may run at the same time on different threads; together
 * they write the whole frame. Uses whatever frames are queued.
 * Returns 0 or a negative errno value.
 */
int hbnlm_filter_slice(HBNLMeans *ctx, uint8_t *const dst[3], const ptrdiff_t stride[3],
                       int job, int nb_jobs);

/* Same as hbnlm_filter_slice(ctx, dst, stride, 0, 1). */
int hbnlm_filter(HBNLMeans *ctx, uint8_t *const dst[3], const ptrdiff_t stride[3]);

/* Drop the head frame from the queue. */
void hbnlm_pop(HBNLMeans *ctx);

#ifdef __cplusplus
}
#endif

#endif /* HBNLMEANS_H */
