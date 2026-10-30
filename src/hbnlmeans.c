/*
 * libhbnlmeans - HandBrake's NLMeans denoiser as a standalone library
 *
 * From HandBrake libhb/nlmeans.c, libhb/nlmeans_x86.c and the presets and
 * tunes in libhb/param.c
 * Copyright (c) 2013 Dirk Farin
 * Copyright (c) 2003-2026 HandBrake Team
 * Library packaging (c) 2026 Corneliu Ticu
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2, as published
 * by the Free Software Foundation. See COPYING.
 */

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "hbnlmeans.h"

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define HBNLM_X86 1
#include <smmintrin.h>
#else
#define HBNLM_X86 0
#endif

#define HBNLM_MIN(a, b) ((a) < (b) ? (a) : (b))
#define HBNLM_MAX(a, b) ((a) > (b) ? (a) : (b))

#define NLMEANS_PREFILTER_MODE_MEAN3X3     HBNLM_PREFILTER_MEAN3X3
#define NLMEANS_PREFILTER_MODE_MEAN5X5     HBNLM_PREFILTER_MEAN5X5
#define NLMEANS_PREFILTER_MODE_MEDIAN3X3   HBNLM_PREFILTER_MEDIAN3X3
#define NLMEANS_PREFILTER_MODE_MEDIAN5X5   HBNLM_PREFILTER_MEDIAN5X5
#define NLMEANS_PREFILTER_MODE_CSM3X3      HBNLM_PREFILTER_CSM3X3
#define NLMEANS_PREFILTER_MODE_CSM5X5      HBNLM_PREFILTER_CSM5X5
#define NLMEANS_PREFILTER_MODE_REDUCE25    HBNLM_PREFILTER_REDUCE25
#define NLMEANS_PREFILTER_MODE_REDUCE50    HBNLM_PREFILTER_REDUCE50
#define NLMEANS_PREFILTER_MODE_EDGEBOOST   HBNLM_PREFILTER_EDGEBOOST
#define NLMEANS_PREFILTER_MODE_PASSTHRU    HBNLM_PREFILTER_PASSTHRU

#define NLMEANS_SORT(a,b) { if (a > b) NLMEANS_SWAP(a, b); }
#define NLMEANS_SWAP(a,b) { a = (a ^ b); b = (a ^ b); a = (b ^ a); }

#define NLMEANS_STRENGTH_MAX   63
#define NLMEANS_PATCH_SIZE_MAX 15
#define NLMEANS_FRAMES_MAX     32
#define NLMEANS_EXPSIZE        128

/* Spare elements after each bordered plane for SIMD over-reads */
#define PLANE_TAIL 64

typedef struct BorderedPlane {
    void *mem;
    void *mem_pre;
    void *image;
    void *image_pre;
    int w;
    int h;
    int border;
} BorderedPlane;

typedef struct HBNFrame {
    BorderedPlane plane[3];
} HBNFrame;

struct PixelSum {
    float weight_sum;
    float pixel_sum;
};

typedef void (*build_integral_fn)(void *integral, int integral_stride,
                                  const void *src_pre, const void *compare_pre,
                                  int w, int border, int dst_w, int dst_h,
                                  int dx, int dy, int n);

struct HBNLMeans {
    HBNLMParams resolved;      /* for hbnlm_get_params, before depth scaling */

    double strength[3];
    double origin_tune[3];
    int    patch_size[3];
    int    range[3];
    int    nframes[3];
    int    prefilter[3];
    int    border[3];

    float    exptable[3][NLMEANS_EXPSIZE];
    float    weight_fact_table[3];
    uint64_t diff_max[3];

    int depth;
    int bps;
    int planewidth[3];
    int planeheight[3];
    int max_frames;

    build_integral_fn build_integral;
    int  (*nlmeans_alloc)(const uint8_t *src, int src_w, int src_s, int src_h,
                          BorderedPlane *dst, int border);
    int  (*nlmeans_prefilter)(BorderedPlane *src, int filter_type);
    void (*nlmeans_deborder)(const BorderedPlane *src, uint8_t *dst,
                             int w, int s, int y0, int y1);
    int  (*nlmeans_plane)(const struct HBNLMeans *s,
                          HBNFrame *const *frame, int plane, int nframes,
                          uint8_t *dst, int dst_w, int dst_s, int y0, int y1);

    HBNFrame *queue[NLMEANS_FRAMES_MAX];
    int nb_queued;
};

#define BIT_DEPTH 8
#include "hbnlmeans_template.c"
#undef BIT_DEPTH

#define BIT_DEPTH 16
#include "hbnlmeans_template.c"
#undef BIT_DEPTH

#if HBNLM_X86
/* Same integer sums as build_integral_scalar, 16 (8-bit) or 8 (16-bit) at a time */
__attribute__((target("sse4.1")))
static void build_integral_8_sse4(void *integral, int integral_stride,
                                  const void *in_src_pre, const void *in_compare_pre,
                                  int w, int border, int dst_w, int dst_h,
                                  int dx, int dy, int n)
{
    const int bw = w + 2 * border;
    const int n_half = (n-1) / 2;

    const uint8_t *src_pre     = (const uint8_t *)(in_src_pre);
    const uint8_t *compare_pre = (const uint8_t *)(in_compare_pre);

    for (int y = 0; y < dst_h + n; y++) {
        __m128i sum_prev = _mm_setzero_si128();

        const uint8_t *pixel_src     = src_pre     + (y-n_half   )*bw - n_half;
        const uint8_t *pixel_compare = compare_pre + (y-n_half+dy)*bw - n_half + dx;
        const __m128i *prev = (const __m128i *)((uint32_t *)(integral) + ((y > 0 ? y-1 : 0) * integral_stride));
        uint32_t *out = (uint32_t *)(integral) + (y*integral_stride);

        for (int x = 0; x < dst_w + n; x += 16) {
            __m128i p1, p1_lo, p1_hi;
            __m128i p2, p2_lo, p2_hi;
            __m128i df_lo, df_hi;
            __m128i sq_lo, sq_hi;
            __m128i sum_grp0, sum_grp1, sum_grp2, sum_grp3, sum_tmp;

            p1 = _mm_loadu_si128((const __m128i*)(pixel_src));
            p2 = _mm_loadu_si128((const __m128i*)(pixel_compare));

            p1_lo = _mm_cvtepu8_epi16(p1);
            p2_lo = _mm_cvtepu8_epi16(p2);
            df_lo = _mm_sub_epi16(p1_lo, p2_lo);
            sq_lo = _mm_mullo_epi16(df_lo, df_lo);
            sum_grp0 = _mm_cvtepu16_epi32(sq_lo);
            sum_grp1 = _mm_cvtepu16_epi32(_mm_srli_si128(sq_lo, 8));

            sum_grp0 = _mm_add_epi32(sum_grp0, _mm_slli_si128(sum_grp0, 4));
            sum_grp0 = _mm_add_epi32(sum_grp0, _mm_slli_si128(sum_grp0, 8));
            sum_grp0 = _mm_add_epi32(sum_grp0, sum_prev);
            sum_tmp  = _mm_shuffle_epi32(sum_grp0, _MM_SHUFFLE(3, 3, 3, 3));

            sum_grp1 = _mm_add_epi32(sum_grp1, _mm_slli_si128(sum_grp1, 4));
            sum_grp1 = _mm_add_epi32(sum_grp1, _mm_slli_si128(sum_grp1, 8));
            sum_grp1 = _mm_add_epi32(sum_grp1, sum_tmp);
            sum_prev = _mm_shuffle_epi32(sum_grp1, _MM_SHUFFLE(3, 3, 3, 3));

            p1_hi = _mm_cvtepu8_epi16(_mm_srli_si128(p1, 8));
            p2_hi = _mm_cvtepu8_epi16(_mm_srli_si128(p2, 8));
            df_hi = _mm_sub_epi16(p1_hi, p2_hi);
            sq_hi = _mm_mullo_epi16(df_hi, df_hi);
            sum_grp2 = _mm_cvtepu16_epi32(sq_hi);
            sum_grp3 = _mm_cvtepu16_epi32(_mm_srli_si128(sq_hi, 8));

            sum_grp2 = _mm_add_epi32(sum_grp2, _mm_slli_si128(sum_grp2, 4));
            sum_grp2 = _mm_add_epi32(sum_grp2, _mm_slli_si128(sum_grp2, 8));
            sum_grp2 = _mm_add_epi32(sum_grp2, sum_prev);
            sum_tmp  = _mm_shuffle_epi32(sum_grp2, _MM_SHUFFLE(3, 3, 3, 3));

            sum_grp3 = _mm_add_epi32(sum_grp3, _mm_slli_si128(sum_grp3, 4));
            sum_grp3 = _mm_add_epi32(sum_grp3, _mm_slli_si128(sum_grp3, 8));
            sum_grp3 = _mm_add_epi32(sum_grp3, sum_tmp);
            sum_prev = _mm_shuffle_epi32(sum_grp3, _MM_SHUFFLE(3, 3, 3, 3));

            if (y > 0) {
                sum_grp0 = _mm_add_epi32(sum_grp0, _mm_loadu_si128(prev));
                sum_grp1 = _mm_add_epi32(sum_grp1, _mm_loadu_si128(prev+1));
                sum_grp2 = _mm_add_epi32(sum_grp2, _mm_loadu_si128(prev+2));
                sum_grp3 = _mm_add_epi32(sum_grp3, _mm_loadu_si128(prev+3));
            }

            _mm_storeu_si128((__m128i*)(out),    sum_grp0);
            _mm_storeu_si128((__m128i*)(out+4),  sum_grp1);
            _mm_storeu_si128((__m128i*)(out+8),  sum_grp2);
            _mm_storeu_si128((__m128i*)(out+12), sum_grp3);

            prev          += 4;
            out           += 16;
            pixel_src     += 16;
            pixel_compare += 16;
        }
    }
}

__attribute__((target("sse4.1")))
static void build_integral_16_sse4(void *integral, int integral_stride,
                                   const void *in_src_pre, const void *in_compare_pre,
                                   int w, int border, int dst_w, int dst_h,
                                   int dx, int dy, int n)
{
    const int bw = w + 2 * border;
    const int n_half = (n-1) / 2;

    const uint16_t *src_pre     = (const uint16_t *)(in_src_pre);
    const uint16_t *compare_pre = (const uint16_t *)(in_compare_pre);

    for (int y = 0; y < dst_h + n; y++) {
        __m128i sum_prev = _mm_setzero_si128();

        const uint16_t *pixel_src     = src_pre     + (y-n_half   )*bw - n_half;
        const uint16_t *pixel_compare = compare_pre + (y-n_half+dy)*bw - n_half + dx;
        const __m128i *prev = (const __m128i *)((uint64_t *)(integral) + ((y > 0 ? y-1 : 0) * integral_stride));
        uint64_t *out = (uint64_t *)(integral) + (y*integral_stride);

        for (int x = 0; x < dst_w + n; x += 8) {
            __m128i p1, p1_lo, p1_hi;
            __m128i p2, p2_lo, p2_hi;
            __m128i df_lo, df_hi;
            __m128i sq_lo, sq_hi;
            __m128i sum_grp0, sum_grp1, sum_grp2, sum_grp3, sum_tmp;

            p1 = _mm_loadu_si128((const __m128i*)(pixel_src));
            p2 = _mm_loadu_si128((const __m128i*)(pixel_compare));

            p1_lo = _mm_cvtepu16_epi32(p1);
            p2_lo = _mm_cvtepu16_epi32(p2);
            df_lo = _mm_sub_epi32(p1_lo, p2_lo);
            sq_lo = _mm_mullo_epi32(df_lo, df_lo);
            sum_grp0 = _mm_cvtepu32_epi64(sq_lo);
            sum_grp1 = _mm_cvtepu32_epi64(_mm_srli_si128(sq_lo, 8));

            sum_grp0 = _mm_add_epi64(sum_grp0, _mm_slli_si128(sum_grp0, 8));
            sum_grp0 = _mm_add_epi64(sum_grp0, sum_prev);
            sum_tmp  = _mm_shuffle_epi32(sum_grp0, _MM_SHUFFLE(3, 2, 3, 2));

            sum_grp1 = _mm_add_epi64(sum_grp1, _mm_slli_si128(sum_grp1, 8));
            sum_grp1 = _mm_add_epi64(sum_grp1, sum_tmp);
            sum_prev = _mm_shuffle_epi32(sum_grp1, _MM_SHUFFLE(3, 2, 3, 2));

            p1_hi = _mm_cvtepu16_epi32(_mm_srli_si128(p1, 8));
            p2_hi = _mm_cvtepu16_epi32(_mm_srli_si128(p2, 8));
            df_hi = _mm_sub_epi32(p1_hi, p2_hi);
            sq_hi = _mm_mullo_epi32(df_hi, df_hi);
            sum_grp2 = _mm_cvtepu32_epi64(sq_hi);
            sum_grp3 = _mm_cvtepu32_epi64(_mm_srli_si128(sq_hi, 8));

            sum_grp2 = _mm_add_epi64(sum_grp2, _mm_slli_si128(sum_grp2, 8));
            sum_grp2 = _mm_add_epi64(sum_grp2, sum_prev);
            sum_tmp  = _mm_shuffle_epi32(sum_grp2, _MM_SHUFFLE(3, 2, 3, 2));

            sum_grp3 = _mm_add_epi64(sum_grp3, _mm_slli_si128(sum_grp3, 8));
            sum_grp3 = _mm_add_epi64(sum_grp3, sum_tmp);
            sum_prev = _mm_shuffle_epi32(sum_grp3, _MM_SHUFFLE(3, 2, 3, 2));

            if (y > 0) {
                sum_grp0 = _mm_add_epi64(sum_grp0, _mm_loadu_si128(prev));
                sum_grp1 = _mm_add_epi64(sum_grp1, _mm_loadu_si128(prev+1));
                sum_grp2 = _mm_add_epi64(sum_grp2, _mm_loadu_si128(prev+2));
                sum_grp3 = _mm_add_epi64(sum_grp3, _mm_loadu_si128(prev+3));
            }

            _mm_storeu_si128((__m128i*)(out),   sum_grp0);
            _mm_storeu_si128((__m128i*)(out+2), sum_grp1);
            _mm_storeu_si128((__m128i*)(out+4), sum_grp2);
            _mm_storeu_si128((__m128i*)(out+6), sum_grp3);

            prev          += 4;
            out           += 8;
            pixel_src     += 8;
            pixel_compare += 8;
        }
    }
}
#endif

unsigned hbnlm_version(void)
{
    return (HBNLM_VERSION_MAJOR << 16) | (HBNLM_VERSION_MINOR << 8) | HBNLM_VERSION_PATCH;
}

void hbnlm_params_init(HBNLMParams *p)
{
    for (int c = 0; c < 3; c++) {
        p->strength[c]    = -1;
        p->origin_tune[c] = -1;
        p->patch_size[c]  = -1;
        p->range[c]       = -1;
        p->frame_count[c] = -1;
        p->prefilter[c]   = -1;
    }
}

/* HandBrake's preset/tune table (libhb/param.c, generate_nlmeans_settings) */
int hbnlm_params_preset(HBNLMParams *p, const char *preset, const char *tune)
{
    double strength[2], origin_tune[2];
    int patch_size[2], range[2], frames[2], prefilter[2];
    int ul, li, st;

    if (!preset)
        return -1;
    ul = !strcasecmp(preset, "ultralight");
    li = !strcasecmp(preset, "light");
    st = !strcasecmp(preset, "strong");
    if (!ul && !li && !st && strcasecmp(preset, "medium"))
        return -1;

    if (!tune || !strcasecmp(tune, "none")) {
        strength[0]    = strength[1]    = 6;
        origin_tune[0] = origin_tune[1] = 1;
        patch_size[0]  = patch_size[1]  = 7;
        range[0]       = range[1]       = 3;
        frames[0]      = frames[1]      = 2;
        prefilter[0]   = prefilter[1]   = 0;
        if (ul)      { strength[0] = strength[1] = 1.5; }
        else if (li) { strength[0] = strength[1] = 3; }
        else if (st) { strength[0] = strength[1] = 10; }
    } else if (!strcasecmp(tune, "film")) {
        strength[0]    = 6; strength[1] = 8;
        origin_tune[0] = origin_tune[1] = 0.8;
        patch_size[0]  = patch_size[1]  = 7;
        range[0]       = range[1]       = 3;
        frames[0]      = frames[1]      = 2;
        prefilter[0]   = prefilter[1]   = 0;
        if (ul)      { strength[0] = 1.5; strength[1] = 2.4; origin_tune[0] = 0.9; origin_tune[1] = 0.9; }
        else if (li) { strength[0] = 3;   strength[1] = 4;   origin_tune[0] = 0.9; origin_tune[1] = 0.9; }
        else if (st) { strength[0] = 8;   strength[1] = 10;  origin_tune[0] = 0.6; origin_tune[1] = 0.6; }
    } else if (!strcasecmp(tune, "grain")) {
        strength[0]    = 0; strength[1] = 6;
        origin_tune[0] = origin_tune[1] = 0.8;
        patch_size[0]  = patch_size[1]  = 7;
        range[0]       = range[1]       = 3;
        frames[0]      = frames[1]      = 2;
        prefilter[0]   = prefilter[1]   = 0;
        if (ul)      { strength[0] = 0; strength[1] = 2.4; origin_tune[0] = 0.9; origin_tune[1] = 0.9; }
        else if (li) { strength[0] = 0; strength[1] = 3.5; origin_tune[0] = 0.9; origin_tune[1] = 0.9; }
        else if (st) { strength[0] = 0; strength[1] = 8;   origin_tune[0] = 0.6; origin_tune[1] = 0.6; }
    } else if (!strcasecmp(tune, "highmotion")) {
        strength[0]    = 6;   strength[1]    = 6;
        origin_tune[0] = 0.8; origin_tune[1] = 0.7;
        patch_size[0]  = 7;   patch_size[1]  = 7;
        range[0]       = 3;   range[1]       = 5;
        frames[0]      = 2;   frames[1]      = 1;
        prefilter[0]   = 0;   prefilter[1]   = 0;
        if (ul)      { strength[0] = 1.5; strength[1] = 2.4;  origin_tune[0] = 0.9; origin_tune[1] = 0.9; }
        else if (li) { strength[0] = 3;   strength[1] = 3.25; origin_tune[0] = 0.9; origin_tune[1] = 0.8; }
        else if (st) { strength[0] = 8;   strength[1] = 6.75; origin_tune[0] = 0.6; origin_tune[1] = 0.5; }
    } else if (!strcasecmp(tune, "animation")) {
        strength[0]    = 5; strength[1] = 4;
        origin_tune[0] = origin_tune[1] = 0.15;
        patch_size[0]  = patch_size[1]  = 5;
        range[0]       = range[1]       = 7;
        frames[0]      = frames[1]      = 4;
        prefilter[0]   = prefilter[1]   = 0;
        if (ul)      { strength[0] = 2.5; strength[1] = 2;    frames[0] = 2; frames[1] = 2; }
        else if (li) { strength[0] = 3;   strength[1] = 2.25; frames[0] = 3; frames[1] = 3; }
        else if (st) { strength[0] = 10;  strength[1] = 8; }
    } else if (!strcasecmp(tune, "tape")) {
        strength[0]    = 3;   strength[1]    = 6;
        origin_tune[0] = 0.8; origin_tune[1] = 0.8;
        patch_size[0]  = 3;   patch_size[1]  = 5;
        range[0]       = 5;   range[1]       = 5;
        frames[0]      = 2;   frames[1]      = 2;
        prefilter[0]   = 0;   prefilter[1]   = 0;
        if (ul)      { strength[0] = 1.5; strength[1] = 5; origin_tune[0] = 0.9; origin_tune[1] = 0.9; frames[0] = 1; frames[1] = 1; }
        else if (li) { strength[0] = 2;   strength[1] = 6; origin_tune[0] = 0.9; origin_tune[1] = 0.9; }
        else if (st) { strength[0] = 3.5; strength[1] = 8; origin_tune[0] = 0.6; origin_tune[1] = 0.6; patch_size[0] = 5; patch_size[1] = 5; }
    } else if (!strcasecmp(tune, "sprite")) {
        strength[0]    = 3;    strength[1]    = 4;
        origin_tune[0] = 0.15; origin_tune[1] = 0.5;
        patch_size[0]  = 5;    patch_size[1]  = 5;
        range[0]       = 5;    range[1]       = 9;
        frames[0]      = 2;    frames[1]      = 4;
        prefilter[0]   = 0;    prefilter[1]   = 0;
        if (ul)      { strength[0] = 1.5; strength[1] = 3; range[0] = 5; range[1] = 7;  frames[0] = 1; frames[1] = 2; }
        else if (li) { strength[0] = 2;   strength[1] = 4; frames[0] = 2; frames[1] = 2; }
        else if (st) { strength[0] = 3;   strength[1] = 4; range[0] = 7; range[1] = 11; }
    } else {
        return -1;
    }

    for (int c = 0; c < 2; c++) {
        if (p->strength[c]    == -1) p->strength[c]    = strength[c];
        if (p->origin_tune[c] == -1) p->origin_tune[c] = origin_tune[c];
        if (p->patch_size[c]  == -1) p->patch_size[c]  = patch_size[c];
        if (p->range[c]       == -1) p->range[c]       = range[c];
        if (p->frame_count[c] == -1) p->frame_count[c] = frames[c];
        if (p->prefilter[c]   == -1) p->prefilter[c]   = prefilter[c];
    }
    return 0;
}

int hbnlm_params_parse(HBNLMParams *p, const char *settings)
{
    const char *s = settings;
    while (s && *s) {
        const char *end = strchr(s, ':');
        size_t len = end ? (size_t)(end - s) : strlen(s);
        char tok[128], *eq, *rest;
        int c;

        if (len >= sizeof(tok))
            return -1;
        memcpy(tok, s, len);
        tok[len] = 0;
        s = end ? end + 1 : NULL;
        if (!len)
            continue;

        eq = strchr(tok, '=');
        if (!eq)
            return -1;
        *eq = 0;
        if      (!strncmp(tok, "y-",  2)) { c = 0; rest = tok + 2; }
        else if (!strncmp(tok, "cb-", 3)) { c = 1; rest = tok + 3; }
        else if (!strncmp(tok, "cr-", 3)) { c = 2; rest = tok + 3; }
        else return -1;

        char *num_end;
        double v = strtod(eq + 1, &num_end);
        if (num_end == eq + 1 || *num_end)
            return -1;
        if      (!strcmp(rest, "strength"))    p->strength[c]    = v;
        else if (!strcmp(rest, "origin-tune")) p->origin_tune[c] = v;
        else if (!strcmp(rest, "patch-size"))  p->patch_size[c]  = (int)v;
        else if (!strcmp(rest, "range"))       p->range[c]       = (int)v;
        else if (!strcmp(rest, "frame-count")) p->frame_count[c] = (int)v;
        else if (!strcmp(rest, "prefilter"))   p->prefilter[c]   = (int)v;
        else return -1;
    }
    return 0;
}

/* Cascade, defaults and clamping exactly as libhb/nlmeans.c nlmeans_init */
static void resolve_params(HBNLMeans *s, const HBNLMParams *in)
{
    for (int c = 0; c < 3; c++) {
        s->strength[c]    = in->strength[c];
        s->origin_tune[c] = in->origin_tune[c];
        s->patch_size[c]  = in->patch_size[c];
        s->range[c]       = in->range[c];
        s->nframes[c]     = in->frame_count[c];
        s->prefilter[c]   = in->prefilter[c];
    }

    // Cr not set; inherit Cb. Cb not set; inherit Y. Y not set; defaults.
    for (int c = 1; c < 3; c++) {
        if (s->strength[c]    == -1) s->strength[c]    = s->strength[c-1];
        if (s->origin_tune[c] == -1) s->origin_tune[c] = s->origin_tune[c-1];
        if (s->patch_size[c]  == -1) s->patch_size[c]  = s->patch_size[c-1];
        if (s->range[c]       == -1) s->range[c]       = s->range[c-1];
        if (s->nframes[c]     == -1) s->nframes[c]     = s->nframes[c-1];
        if (s->prefilter[c]   == -1) s->prefilter[c]   = s->prefilter[c-1];
    }

    s->max_frames = 0;
    for (int c = 0; c < 3; c++) {
        if (s->strength[c]    == -1) s->strength[c]    = 6;
        if (s->origin_tune[c] == -1) s->origin_tune[c] = 1;
        if (s->patch_size[c]  == -1) s->patch_size[c]  = 7;
        if (s->range[c]       == -1) s->range[c]       = 3;
        if (s->nframes[c]     == -1) s->nframes[c]     = 2;
        if (s->prefilter[c]   == -1) s->prefilter[c]   = 0;

        if (s->strength[c] < 0)                        s->strength[c] = 0;
        if (s->strength[c] > NLMEANS_STRENGTH_MAX)     s->strength[c] = NLMEANS_STRENGTH_MAX;
        if (s->origin_tune[c] < 0.01)                  s->origin_tune[c] = 0.01; // avoid black artifacts
        if (s->origin_tune[c] > 1)                     s->origin_tune[c] = 1;
        if (s->patch_size[c] % 2 == 0)                 s->patch_size[c]--;
        if (s->patch_size[c] < 1)                      s->patch_size[c] = 1;
        if (s->patch_size[c] > NLMEANS_PATCH_SIZE_MAX) s->patch_size[c] = NLMEANS_PATCH_SIZE_MAX;
        if (s->range[c] % 2 == 0)                      s->range[c]--;
        if (s->range[c] < 1)                           s->range[c] = 1;
        if (s->nframes[c] < 1)                         s->nframes[c] = 1;
        if (s->nframes[c] > NLMEANS_FRAMES_MAX)        s->nframes[c] = NLMEANS_FRAMES_MAX;
        if (s->prefilter[c] < 0)                       s->prefilter[c] = 0;

        if (s->max_frames < s->nframes[c])
            s->max_frames = s->nframes[c];

        s->resolved.strength[c]    = s->strength[c];
        s->resolved.origin_tune[c] = s->origin_tune[c];
        s->resolved.patch_size[c]  = s->patch_size[c];
        s->resolved.range[c]       = s->range[c];
        s->resolved.frame_count[c] = s->nframes[c];
        s->resolved.prefilter[c]   = s->prefilter[c];

        // Scale strength with bit depth
        s->strength[c] *= s->depth > 8 ? 1 << (s->depth - 8) : 1;

        // Border as HandBrake, widened only when the search range would read past it
        const int n_half = (s->patch_size[c] - 1) / 2;
        const int r_half = (s->range[c] - 1) / 2;
        s->border[c] = HBNLM_MAX(((s->patch_size[c] + 2) / 2 + 15) / 16 * 16,
                                 (n_half + r_half + 1 + 15) / 16 * 16);

        // Precompute exponential table
        float *exptable          = &s->exptable[c][0];
        float *weight_fact_table = &s->weight_fact_table[c];
        uint64_t *diff_max       = &s->diff_max[c];
        const float weight_factor       = 1.0/s->patch_size[c]/s->patch_size[c] / (s->strength[c] * s->strength[c]);
        const float min_weight_in_table = 0.0005;
        const float stretch             = NLMEANS_EXPSIZE / (-log(min_weight_in_table));
        *(weight_fact_table)            = weight_factor * stretch;
        const double diff_max_d         = NLMEANS_EXPSIZE / (double)(*weight_fact_table);
        *(diff_max)                     = diff_max_d >= (double)(UINT64_MAX) ? UINT64_MAX : (uint64_t)(diff_max_d);
        for (int i = 0; i < NLMEANS_EXPSIZE; i++)
            exptable[i] = exp(-i/stretch);
        exptable[NLMEANS_EXPSIZE-1] = 0;
    }
}

#if HBNLM_X86
static int have_sse41(void)
{
    __builtin_cpu_init();
    return __builtin_cpu_supports("sse4.1");
}
#endif

HBNLMeans *hbnlm_create(const HBNLMParams *p, int width, int height,
                        int log2_chroma_w, int log2_chroma_h, int depth)
{
    HBNLMParams unset;
    HBNLMeans *s;

    if (width < 1 || height < 1 || depth < 8 || depth > 16 ||
        log2_chroma_w < 0 || log2_chroma_w > 2 || log2_chroma_h < 0 || log2_chroma_h > 2)
        return NULL;
    if (!p) {
        hbnlm_params_init(&unset);
        p = &unset;
    }

    s = calloc(1, sizeof(*s));
    if (!s)
        return NULL;

    s->depth = depth;
    s->bps   = depth > 8 ? 2 : 1;
    s->planewidth[0]  = width;
    s->planeheight[0] = height;
    s->planewidth[1]  = s->planewidth[2]  = -((-width)  >> log2_chroma_w);
    s->planeheight[1] = s->planeheight[2] = -((-height) >> log2_chroma_h);

    if (depth > 8) {
        s->build_integral    = build_integral_scalar_16;
        s->nlmeans_alloc     = nlmeans_alloc_16;
        s->nlmeans_prefilter = nlmeans_prefilter_16;
        s->nlmeans_deborder  = nlmeans_deborder_16;
        s->nlmeans_plane     = nlmeans_plane_16;
    } else {
        s->build_integral    = build_integral_scalar_8;
        s->nlmeans_alloc     = nlmeans_alloc_8;
        s->nlmeans_prefilter = nlmeans_prefilter_8;
        s->nlmeans_deborder  = nlmeans_deborder_8;
        s->nlmeans_plane     = nlmeans_plane_8;
    }
#if HBNLM_X86
    if (have_sse41())
        s->build_integral = depth > 8 ? build_integral_16_sse4 : build_integral_8_sse4;
#endif

    resolve_params(s, p);
    return s;
}

static void free_frame(HBNFrame **pf)
{
    HBNFrame *f = *pf;
    if (!f)
        return;
    for (int c = 0; c < 3; c++) {
        if (f->plane[c].mem_pre != f->plane[c].mem)
            free(f->plane[c].mem_pre);
        free(f->plane[c].mem);
    }
    free(f);
    *pf = NULL;
}

void hbnlm_free(HBNLMeans **ctx)
{
    HBNLMeans *s = ctx ? *ctx : NULL;
    if (!s)
        return;
    for (int i = 0; i < s->nb_queued; i++)
        free_frame(&s->queue[i]);
    free(s);
    *ctx = NULL;
}

void hbnlm_get_params(const HBNLMeans *s, HBNLMParams *out)
{
    *out = s->resolved;
}

int hbnlm_frames_needed(const HBNLMeans *s)
{
    return s->max_frames;
}

int hbnlm_queued(const HBNLMeans *s)
{
    return s->nb_queued;
}

int hbnlm_push(HBNLMeans *s, const uint8_t *const src[3], const ptrdiff_t stride[3])
{
    HBNFrame *f;
    int ret;

    if (s->nb_queued >= s->max_frames)
        return -EAGAIN;
    f = calloc(1, sizeof(*f));
    if (!f)
        return -ENOMEM;

    for (int c = 0; c < 3; c++) {
        ret = s->nlmeans_alloc(src[c], s->planewidth[c], (int)(stride[c] / s->bps),
                               s->planeheight[c], &f->plane[c], s->border[c]);
        if (ret < 0)
            goto fail;
        // Prefilter once, up front, for the frame's whole life in the queue
        if (s->prefilter[c] && (s->strength[c] != 0 || s->prefilter[c] & NLMEANS_PREFILTER_MODE_PASSTHRU)) {
            ret = s->nlmeans_prefilter(&f->plane[c], s->prefilter[c]);
            if (ret < 0)
                goto fail;
        }
    }
    s->queue[s->nb_queued++] = f;
    return 0;
fail:
    free_frame(&f);
    return ret;
}

int hbnlm_filter_slice(HBNLMeans *s, uint8_t *const dst[3], const ptrdiff_t stride[3],
                       int job, int nb_jobs)
{
    if (!s->nb_queued || nb_jobs < 1 || job < 0 || job >= nb_jobs)
        return -EINVAL;

    for (int c = 0; c < 3; c++) {
        const int h  = s->planeheight[c];
        const int y0 = (int)(((int64_t)h *  job)      / nb_jobs);
        const int y1 = (int)(((int64_t)h * (job + 1)) / nb_jobs);
        const int dst_s = (int)(stride[c] / s->bps);
        int ret;

        if (y0 >= y1)
            continue;

        if (s->prefilter[c] & NLMEANS_PREFILTER_MODE_PASSTHRU || s->strength[c] == 0) {
            s->nlmeans_deborder(&s->queue[0]->plane[c], dst[c], s->planewidth[c], dst_s, y0, y1);
            continue;
        }

        ret = s->nlmeans_plane(s, s->queue, c, HBNLM_MIN(s->nframes[c], s->nb_queued),
                               dst[c], s->planewidth[c], dst_s, y0, y1);
        if (ret < 0)
            return ret;
    }
    return 0;
}

int hbnlm_filter(HBNLMeans *s, uint8_t *const dst[3], const ptrdiff_t stride[3])
{
    return hbnlm_filter_slice(s, dst, stride, 0, 1);
}

void hbnlm_pop(HBNLMeans *s)
{
    if (!s->nb_queued)
        return;
    free_frame(&s->queue[0]);
    memmove(&s->queue[0], &s->queue[1], (s->nb_queued - 1) * sizeof(*s->queue));
    s->queue[--s->nb_queued] = NULL;
}
