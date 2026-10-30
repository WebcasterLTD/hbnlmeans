/*
 * HandBrake NLMeans denoiser, through libhbnlmeans
 * Copyright (c) 2026 Corneliu Ticu
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * libhbnlmeans is GPL v2, so FFmpeg must be built with --enable-gpl.
 */

/*
 * Options use HandBrake's own names, so a HandBrake custom string works as is:
 *     hbnlmeans=y-strength=6:y-origin-tune=0.8:cb-strength=8
 * or a HandBrake preset and tune:
 *     hbnlmeans=preset=medium:tune=film
 * Explicit values override the preset.
 */

#include <hbnlmeans.h>

#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "filters.h"
#include "video.h"

static const char *const preset_names[] = { "custom", "ultralight", "light", "medium", "strong" };
static const char *const tune_names[]    = { "none", "film", "grain", "highmotion", "animation", "tape", "sprite" };

#define MAX_QUEUE 32

typedef struct HBNLMeansFilterContext {
    const AVClass *class;

    int    preset;
    int    tune;
    double strength[3];
    double origin_tune[3];
    int    patch_size[3];
    int    range[3];
    int    frame_count[3];
    int    prefilter[3];

    HBNLMeans *nlm;
    AVFrame   *props[MAX_QUEUE]; // queued frames' properties, no data
    int        nb_queued;
    int        eof;
    int64_t    eof_pts;
} HBNLMeansFilterContext;

#define OFFSET(x) offsetof(HBNLMeansFilterContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM

#define CHANNEL_OPTS(prefix, i)                                                                                           \
    { prefix "-strength",    "averaging weight decay, larger is smoother", OFFSET(strength[i]),    AV_OPT_TYPE_DOUBLE, {.dbl=-1}, -1, 1000, FLAGS }, \
    { prefix "-origin-tune", "weight of the origin patch, 0.01..1",         OFFSET(origin_tune[i]), AV_OPT_TYPE_DOUBLE, {.dbl=-1}, -1, 1,    FLAGS }, \
    { prefix "-patch-size",  "patch width in pixels (odd)",                 OFFSET(patch_size[i]),  AV_OPT_TYPE_INT,    {.i64=-1}, -1, 99,   FLAGS }, \
    { prefix "-range",       "spatial search window width (odd)",           OFFSET(range[i]),       AV_OPT_TYPE_INT,    {.i64=-1}, -1, 99,   FLAGS }, \
    { prefix "-frame-count", "temporal search depth in frames",             OFFSET(frame_count[i]), AV_OPT_TYPE_INT,    {.i64=-1}, -1, 32,   FLAGS }, \
    { prefix "-prefilter",   "prefilter mode (sum of mode values)",         OFFSET(prefilter[i]),   AV_OPT_TYPE_INT,    {.i64=-1}, -1, 4095, FLAGS }

static const AVOption hbnlmeans_options[] = {
    { "preset", "HandBrake preset", OFFSET(preset), AV_OPT_TYPE_INT, {.i64=0}, 0, 4, FLAGS, .unit = "preset" },
        { "custom",     "no preset, only the explicit values", 0, AV_OPT_TYPE_CONST, {.i64=0}, 0, 0, FLAGS, .unit = "preset" },
        { "ultralight", NULL, 0, AV_OPT_TYPE_CONST, {.i64=1}, 0, 0, FLAGS, .unit = "preset" },
        { "light",      NULL, 0, AV_OPT_TYPE_CONST, {.i64=2}, 0, 0, FLAGS, .unit = "preset" },
        { "medium",     NULL, 0, AV_OPT_TYPE_CONST, {.i64=3}, 0, 0, FLAGS, .unit = "preset" },
        { "strong",     NULL, 0, AV_OPT_TYPE_CONST, {.i64=4}, 0, 0, FLAGS, .unit = "preset" },
    { "tune", "HandBrake tune (with a preset)", OFFSET(tune), AV_OPT_TYPE_INT, {.i64=0}, 0, 6, FLAGS, .unit = "tune" },
        { "none",       NULL,                                    0, AV_OPT_TYPE_CONST, {.i64=0}, 0, 0, FLAGS, .unit = "tune" },
        { "film",       "most content, live action",             0, AV_OPT_TYPE_CONST, {.i64=1}, 0, 0, FLAGS, .unit = "tune" },
        { "grain",      "like film but preserves luma grain",    0, AV_OPT_TYPE_CONST, {.i64=2}, 0, 0, FLAGS, .unit = "tune" },
        { "highmotion", "like film but avoids color smearing",   0, AV_OPT_TYPE_CONST, {.i64=3}, 0, 0, FLAGS, .unit = "tune" },
        { "animation",  "cel animation such as cartoons, anime", 0, AV_OPT_TYPE_CONST, {.i64=4}, 0, 0, FLAGS, .unit = "tune" },
        { "tape",       "analog tape sources such as VHS",       0, AV_OPT_TYPE_CONST, {.i64=5}, 0, 0, FLAGS, .unit = "tune" },
        { "sprite",     "1-/4-/8-/16-bit 2-dimensional games",   0, AV_OPT_TYPE_CONST, {.i64=6}, 0, 0, FLAGS, .unit = "tune" },
    CHANNEL_OPTS("y",  0),
    CHANNEL_OPTS("cb", 1),
    CHANNEL_OPTS("cr", 2),
    { NULL }
};

AVFILTER_DEFINE_CLASS(hbnlmeans);

static const enum AVPixelFormat pix_fmts[] = {
    AV_PIX_FMT_YUV410P, AV_PIX_FMT_YUV411P, AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV440P, AV_PIX_FMT_YUV444P,
    AV_PIX_FMT_YUVJ411P, AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_YUVJ422P,
    AV_PIX_FMT_YUVJ440P, AV_PIX_FMT_YUVJ444P,
    AV_PIX_FMT_YUV420P9,  AV_PIX_FMT_YUV422P9,  AV_PIX_FMT_YUV444P9,
    AV_PIX_FMT_YUV420P10, AV_PIX_FMT_YUV422P10, AV_PIX_FMT_YUV440P10, AV_PIX_FMT_YUV444P10,
    AV_PIX_FMT_YUV420P12, AV_PIX_FMT_YUV422P12, AV_PIX_FMT_YUV440P12, AV_PIX_FMT_YUV444P12,
    AV_PIX_FMT_YUV420P14, AV_PIX_FMT_YUV422P14, AV_PIX_FMT_YUV444P14,
    AV_PIX_FMT_YUV420P16, AV_PIX_FMT_YUV422P16, AV_PIX_FMT_YUV444P16,
    AV_PIX_FMT_NONE
};

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    HBNLMeansFilterContext *s = ctx->priv;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);
    HBNLMParams p, used;

    for (int c = 0; c < 3; c++) {
        p.strength[c]    = s->strength[c];
        p.origin_tune[c] = s->origin_tune[c];
        p.patch_size[c]  = s->patch_size[c];
        p.range[c]       = s->range[c];
        p.frame_count[c] = s->frame_count[c];
        p.prefilter[c]   = s->prefilter[c];
    }
    if (s->preset)
        hbnlm_params_preset(&p, preset_names[s->preset], tune_names[s->tune]);
    else if (s->tune)
        av_log(ctx, AV_LOG_WARNING, "tune only applies with a preset; ignoring it.\n");

    hbnlm_free(&s->nlm);
    s->nlm = hbnlm_create(&p, inlink->w, inlink->h, desc->log2_chroma_w,
                          desc->log2_chroma_h, desc->comp[0].depth);
    if (!s->nlm)
        return AVERROR(ENOMEM);

    hbnlm_get_params(s->nlm, &used);
    for (int c = 0; c < 3; c++)
        av_log(ctx, AV_LOG_VERBOSE, "%s: strength %g origin-tune %g patch %d range %d frames %d prefilter %d\n",
               c == 0 ? "y" : c == 1 ? "cb" : "cr", used.strength[c], used.origin_tune[c],
               used.patch_size[c], used.range[c], used.frame_count[c], used.prefilter[c]);
    return 0;
}

static int filter_slice(AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs)
{
    HBNLMeansFilterContext *s = ctx->priv;
    AVFrame *out = arg;
    uint8_t *dst[3] = { out->data[0], out->data[1], out->data[2] };
    const ptrdiff_t stride[3] = { out->linesize[0], out->linesize[1], out->linesize[2] };
    int ret = hbnlm_filter_slice(s->nlm, dst, stride, jobnr, nb_jobs);
    return ret < 0 ? AVERROR(-ret) : 0;
}

// Denoise the head of the queue, send it, and drop it
static int output_frame(AVFilterContext *ctx)
{
    HBNLMeansFilterContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    AVFrame *out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    int ret;

    if (!out)
        return AVERROR(ENOMEM);
    ret = av_frame_copy_props(out, s->props[0]);
    if (ret >= 0)
        ret = ff_filter_execute(ctx, filter_slice, out, NULL,
                                FFMIN(AV_CEIL_RSHIFT(outlink->h, 2), ff_filter_get_nb_threads(ctx)));
    if (ret < 0) {
        av_frame_free(&out);
        return ret;
    }

    hbnlm_pop(s->nlm);
    av_frame_free(&s->props[0]);
    memmove(&s->props[0], &s->props[1], (s->nb_queued - 1) * sizeof(*s->props));
    s->props[--s->nb_queued] = NULL;

    return ff_filter_frame(outlink, out);
}

static int add_frame(AVFilterContext *ctx, AVFrame *in)
{
    HBNLMeansFilterContext *s = ctx->priv;
    const uint8_t *src[3] = { in->data[0], in->data[1], in->data[2] };
    const ptrdiff_t stride[3] = { in->linesize[0], in->linesize[1], in->linesize[2] };
    AVFrame *props = av_frame_alloc();
    int ret;

    if (!props) {
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }
    ret = av_frame_copy_props(props, in);
    if (ret >= 0) {
        ret = hbnlm_push(s->nlm, src, stride);
        ret = ret < 0 ? AVERROR(-ret) : 0;
    }
    av_frame_free(&in);
    if (ret < 0) {
        av_frame_free(&props);
        return ret;
    }
    s->props[s->nb_queued++] = props;
    return 0;
}

static int activate(AVFilterContext *ctx)
{
    HBNLMeansFilterContext *s = ctx->priv;
    AVFilterLink *inlink  = ctx->inputs[0];
    AVFilterLink *outlink = ctx->outputs[0];
    AVFrame *in;
    int64_t pts;
    int status, ret;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    if (!s->eof) {
        ret = ff_inlink_consume_frame(inlink, &in);
        if (ret < 0)
            return ret;
        if (ret > 0) {
            ret = add_frame(ctx, in);
            if (ret < 0)
                return ret;
            // Each frame is compared with the frames after it, so wait for them
            if (s->nb_queued >= hbnlm_frames_needed(s->nlm))
                ret = output_frame(ctx);
            ff_filter_set_ready(ctx, 100);
            return ret;
        }
        if (ff_inlink_acknowledge_status(inlink, &status, &pts)) {
            s->eof = 1;
            s->eof_pts = pts;
        }
    }

    if (s->eof) {
        // Flush: the last frames use only the frames that remain
        if (s->nb_queued) {
            ret = output_frame(ctx);
            ff_filter_set_ready(ctx, 100);
            return ret;
        }
        ff_outlink_set_status(outlink, AVERROR_EOF, s->eof_pts);
        return 0;
    }

    FF_FILTER_FORWARD_WANTED(outlink, inlink);
    return FFERROR_NOT_READY;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    HBNLMeansFilterContext *s = ctx->priv;
    for (int i = 0; i < s->nb_queued; i++)
        av_frame_free(&s->props[i]);
    s->nb_queued = 0;
    hbnlm_free(&s->nlm);
}

static const AVFilterPad hbnlmeans_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input,
    },
};

const FFFilter ff_vf_hbnlmeans = {
    .p.name        = "hbnlmeans",
    .p.description = NULL_IF_CONFIG_SMALL("HandBrake's NLMeans denoiser (libhbnlmeans)."),
    .p.priv_class  = &hbnlmeans_class,
    .p.flags       = AVFILTER_FLAG_SLICE_THREADS,
    .priv_size     = sizeof(HBNLMeansFilterContext),
    .activate      = activate,
    .uninit        = uninit,
    FILTER_INPUTS(hbnlmeans_inputs),
    FILTER_OUTPUTS(ff_video_default_filterpad),
    FILTER_PIXFMTS_ARRAY(pix_fmts),
};
