/*
 * HandBrake NLMeans denoiser, per-bit-depth template.
 *
 * From HandBrake libhb/templates/nlmeans_template.c
 * Copyright (c) 2013 Dirk Farin
 * Copyright (c) 2003-2026 HandBrake Team
 *
 * It may be used under the terms of the GNU General Public License v2.
 * The arithmetic is kept exactly as in HandBrake so the output matches
 * HandBrake bit for bit. The only change is that a plane can be processed
 * in horizontal slices (rows y0..y1), which gives the same result because
 * every output pixel depends only on its own patch sums.
 */

#undef pixel
#undef pixel_2
#undef integral_pixel
#undef pixel_diff
#undef pixel_diff_2
#undef FUNC

#if BIT_DEPTH > 8
#   define pixel          uint16_t
#   define pixel_2        uint32_t
#   define integral_pixel uint64_t
#   define pixel_diff     int32_t
#   define pixel_diff_2   int64_t
#   define FUNC(name) name##_16
#else
#   define pixel          uint8_t
#   define pixel_2        uint16_t
#   define integral_pixel uint32_t
#   define pixel_diff     int16_t
#   define pixel_diff_2   int32_t
#   define FUNC(name) name##_8
#endif

static void FUNC(nlmeans_border)(pixel *src, const int w, const int h,
                                 const int border)
{
    const int bw = w + 2 * border;
    pixel *image = src + border + bw * border;

    // Create faux borders using edge pixels
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < border; x++) {
            *(image + y*bw - x - 1) = *(image + y*bw + x);
            *(image + y*bw + x + w) = *(image + y*bw - x + (w-1));
        }
    }
    for (int y = 0; y < border; y++) {
        memcpy(image - border - (y+1)*bw, image - border +       y*bw, bw * sizeof(pixel));
        memcpy(image - border + (y+h)*bw, image - border + (h-y-1)*bw, bw * sizeof(pixel));
    }
}

static void FUNC(nlmeans_deborder)(const BorderedPlane *src, uint8_t *in_dst,
                                   const int w, const int s,
                                   const int y0, const int y1)
{
    pixel *dst = (pixel *)in_dst;
    const int bw = src->w + 2 * src->border;
    const pixel *image = (const pixel *)src->image;
    const int width = HBNLM_MIN(w, src->w);

    for (int y = y0; y < y1; y++)
        memcpy(dst + y * s, image + y * bw, width * sizeof(pixel));
}

static int FUNC(nlmeans_alloc)(const uint8_t *in_src, const int src_w,
                               const int src_s, const int src_h,
                               BorderedPlane *dst, const int border)
{
    const int bw = src_w + 2 * border;
    const int bh = src_h + 2 * border;
    const pixel *src = (const pixel *)in_src;

    // Extra tail so the SIMD integral may over-read the last row safely
    pixel *mem = malloc((bw * bh + PLANE_TAIL) * sizeof(pixel));
    if (!mem)
        return -ENOMEM;
    memset(mem + bw * bh, 0, PLANE_TAIL * sizeof(pixel));
    pixel *image = mem + border + bw * border;

    for (int y = 0; y < src_h; y++)
        memcpy(image + y * bw, src + y * src_s, src_w * sizeof(pixel));

    dst->mem    = mem;
    dst->image  = image;
    dst->w      = src_w;
    dst->h      = src_h;
    dst->border = border;

    FUNC(nlmeans_border)(dst->mem, dst->w, dst->h, dst->border);
    dst->mem_pre   = dst->mem;
    dst->image_pre = dst->image;
    return 0;
}

static void FUNC(nlmeans_filter_mean)(const pixel *src, pixel *dst,
                                      const int w, const int h,
                                      const int border, const int size)
{
    const int bw = w + 2 * border;
    const int offset_min = -((size - 1) / 2);
    const int offset_max =   (size + 1) / 2;
    const double pixel_weight = 1.0 / (size * size);
    pixel_2 pixel_sum;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            pixel_sum = 0;
            for (int k = offset_min; k < offset_max; k++)
                for (int j = offset_min; j < offset_max; j++)
                    pixel_sum = pixel_sum + *(src + bw*(y+j) + (x+k));
            *(dst + bw*y + x) = (pixel)(pixel_sum * pixel_weight);
        }
    }
}

static pixel FUNC(nlmeans_filter_median_opt)(pixel *pixels, const int size)
{
    // Sorting networks by Nicolas Devillard (opt_med9, opt_med25)
    if (size == 3) {
        NLMEANS_SORT(pixels[1], pixels[2]); NLMEANS_SORT(pixels[4], pixels[5]); NLMEANS_SORT(pixels[7], pixels[8]);
        NLMEANS_SORT(pixels[0], pixels[1]); NLMEANS_SORT(pixels[3], pixels[4]); NLMEANS_SORT(pixels[6], pixels[7]);
        NLMEANS_SORT(pixels[1], pixels[2]); NLMEANS_SORT(pixels[4], pixels[5]); NLMEANS_SORT(pixels[7], pixels[8]);
        NLMEANS_SORT(pixels[0], pixels[3]); NLMEANS_SORT(pixels[5], pixels[8]); NLMEANS_SORT(pixels[4], pixels[7]);
        NLMEANS_SORT(pixels[3], pixels[6]); NLMEANS_SORT(pixels[1], pixels[4]); NLMEANS_SORT(pixels[2], pixels[5]);
        NLMEANS_SORT(pixels[4], pixels[7]); NLMEANS_SORT(pixels[4], pixels[2]); NLMEANS_SORT(pixels[6], pixels[4]);
        NLMEANS_SORT(pixels[4], pixels[2]);
        return pixels[4];
    } else if (size == 5) {
        NLMEANS_SORT(pixels[0],  pixels[1]);  NLMEANS_SORT(pixels[3],  pixels[4]);  NLMEANS_SORT(pixels[2],  pixels[4]);
        NLMEANS_SORT(pixels[2],  pixels[3]);  NLMEANS_SORT(pixels[6],  pixels[7]);  NLMEANS_SORT(pixels[5],  pixels[7]);
        NLMEANS_SORT(pixels[5],  pixels[6]);  NLMEANS_SORT(pixels[9],  pixels[10]); NLMEANS_SORT(pixels[8],  pixels[10]);
        NLMEANS_SORT(pixels[8],  pixels[9]);  NLMEANS_SORT(pixels[12], pixels[13]); NLMEANS_SORT(pixels[11], pixels[13]);
        NLMEANS_SORT(pixels[11], pixels[12]); NLMEANS_SORT(pixels[15], pixels[16]); NLMEANS_SORT(pixels[14], pixels[16]);
        NLMEANS_SORT(pixels[14], pixels[15]); NLMEANS_SORT(pixels[18], pixels[19]); NLMEANS_SORT(pixels[17], pixels[19]);
        NLMEANS_SORT(pixels[17], pixels[18]); NLMEANS_SORT(pixels[21], pixels[22]); NLMEANS_SORT(pixels[20], pixels[22]);
        NLMEANS_SORT(pixels[20], pixels[21]); NLMEANS_SORT(pixels[23], pixels[24]); NLMEANS_SORT(pixels[2],  pixels[5]);
        NLMEANS_SORT(pixels[3],  pixels[6]);  NLMEANS_SORT(pixels[0],  pixels[6]);  NLMEANS_SORT(pixels[0],  pixels[3]);
        NLMEANS_SORT(pixels[4],  pixels[7]);  NLMEANS_SORT(pixels[1],  pixels[7]);  NLMEANS_SORT(pixels[1],  pixels[4]);
        NLMEANS_SORT(pixels[11], pixels[14]); NLMEANS_SORT(pixels[8],  pixels[14]); NLMEANS_SORT(pixels[8],  pixels[11]);
        NLMEANS_SORT(pixels[12], pixels[15]); NLMEANS_SORT(pixels[9],  pixels[15]); NLMEANS_SORT(pixels[9],  pixels[12]);
        NLMEANS_SORT(pixels[13], pixels[16]); NLMEANS_SORT(pixels[10], pixels[16]); NLMEANS_SORT(pixels[10], pixels[13]);
        NLMEANS_SORT(pixels[20], pixels[23]); NLMEANS_SORT(pixels[17], pixels[23]); NLMEANS_SORT(pixels[17], pixels[20]);
        NLMEANS_SORT(pixels[21], pixels[24]); NLMEANS_SORT(pixels[18], pixels[24]); NLMEANS_SORT(pixels[18], pixels[21]);
        NLMEANS_SORT(pixels[19], pixels[22]); NLMEANS_SORT(pixels[8],  pixels[17]); NLMEANS_SORT(pixels[9],  pixels[18]);
        NLMEANS_SORT(pixels[0],  pixels[18]); NLMEANS_SORT(pixels[0],  pixels[9]);  NLMEANS_SORT(pixels[10], pixels[19]);
        NLMEANS_SORT(pixels[1],  pixels[19]); NLMEANS_SORT(pixels[1],  pixels[10]); NLMEANS_SORT(pixels[11], pixels[20]);
        NLMEANS_SORT(pixels[2],  pixels[20]); NLMEANS_SORT(pixels[2],  pixels[11]); NLMEANS_SORT(pixels[12], pixels[21]);
        NLMEANS_SORT(pixels[3],  pixels[21]); NLMEANS_SORT(pixels[3],  pixels[12]); NLMEANS_SORT(pixels[13], pixels[22]);
        NLMEANS_SORT(pixels[4],  pixels[22]); NLMEANS_SORT(pixels[4],  pixels[13]); NLMEANS_SORT(pixels[14], pixels[23]);
        NLMEANS_SORT(pixels[5],  pixels[23]); NLMEANS_SORT(pixels[5],  pixels[14]); NLMEANS_SORT(pixels[15], pixels[24]);
        NLMEANS_SORT(pixels[6],  pixels[24]); NLMEANS_SORT(pixels[6],  pixels[15]); NLMEANS_SORT(pixels[7],  pixels[16]);
        NLMEANS_SORT(pixels[7],  pixels[19]); NLMEANS_SORT(pixels[13], pixels[21]); NLMEANS_SORT(pixels[15], pixels[23]);
        NLMEANS_SORT(pixels[7],  pixels[13]); NLMEANS_SORT(pixels[7],  pixels[15]); NLMEANS_SORT(pixels[1],  pixels[9]);
        NLMEANS_SORT(pixels[3],  pixels[11]); NLMEANS_SORT(pixels[5],  pixels[17]); NLMEANS_SORT(pixels[11], pixels[17]);
        NLMEANS_SORT(pixels[9],  pixels[17]); NLMEANS_SORT(pixels[4],  pixels[10]); NLMEANS_SORT(pixels[6],  pixels[12]);
        NLMEANS_SORT(pixels[7],  pixels[14]); NLMEANS_SORT(pixels[4],  pixels[6]);  NLMEANS_SORT(pixels[4],  pixels[7]);
        NLMEANS_SORT(pixels[12], pixels[14]); NLMEANS_SORT(pixels[10], pixels[14]); NLMEANS_SORT(pixels[6],  pixels[7]);
        NLMEANS_SORT(pixels[10], pixels[12]); NLMEANS_SORT(pixels[6],  pixels[10]); NLMEANS_SORT(pixels[6],  pixels[17]);
        NLMEANS_SORT(pixels[12], pixels[17]); NLMEANS_SORT(pixels[7],  pixels[17]); NLMEANS_SORT(pixels[7],  pixels[10]);
        NLMEANS_SORT(pixels[12], pixels[18]); NLMEANS_SORT(pixels[7],  pixels[12]); NLMEANS_SORT(pixels[10], pixels[18]);
        NLMEANS_SORT(pixels[12], pixels[20]); NLMEANS_SORT(pixels[10], pixels[20]); NLMEANS_SORT(pixels[10], pixels[12]);
        return pixels[12];
    }
    return pixels[(int)((size * size) / 2)];
}

static void FUNC(nlmeans_filter_median)(const pixel *src, pixel *dst,
                                        const int w, const int h,
                                        const int border, const int size)
{
    const int bw = w + 2 * border;
    const int offset_min = -((size - 1) / 2);
    const int offset_max =   (size + 1) / 2;
    pixel pixels[25];
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int index = 0;
            for (int k = offset_min; k < offset_max; k++)
                for (int j = offset_min; j < offset_max; j++)
                    pixels[index++] = *(src + bw*(y+j) + (x+k));
            *(dst + bw*y + x) = FUNC(nlmeans_filter_median_opt)(pixels, size);
        }
    }
}

static void FUNC(nlmeans_filter_csm)(const pixel *src, pixel *dst,
                                     const int w, const int h,
                                     const int border, const int size)
{
    const int bw = w + 2 * border;
    const int offset_min = -((size - 1) / 2);
    const int offset_max =   (size + 1) / 2;
    pixel min = 0, max = 0, min2, max2, min3, max3, median, pixel;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int k = offset_min; k < offset_max; k++) {
                for (int j = offset_min; j < offset_max; j++) {
                    if (k == 0 && j == 0)
                        goto end; // Ignore origin
                    pixel = *(src + bw*(y+j) + (x+k));
                    if (k == offset_min && j == offset_min) {
                        min = pixel;
                        max = min;
                        goto end;
                    }
                    if (pixel < min)
                        min = pixel;
                    if (pixel > max)
                        max = pixel;
                }
                end:
                    continue;
            }

            median = (min + max) / 2;
            min2 = (min  + median) / 2;
            max2 = (max  + median) / 2;
            min3 = (min2 + median) / 2;
            max3 = (max2 + median) / 2;

            pixel = *(src + bw*(y) + (x));
            if      (pixel < min)  *(dst + bw*y + x) = min;
            else if (pixel > max)  *(dst + bw*y + x) = max;
            else if (pixel < min2) *(dst + bw*y + x) = min2;
            else if (pixel > max2) *(dst + bw*y + x) = max2;
            else if (pixel < min3) *(dst + bw*y + x) = min3;
            else if (pixel > max3) *(dst + bw*y + x) = max3;
        }
    }
}

static int FUNC(nlmeans_filter_edgeboost)(const pixel *src, pixel *dst,
                                          const int w, const int h,
                                          const int border)
{
    const int bw = w + 2 * border;
    const int bh = h + 2 * border;

    const int kernel_size = 3;
    const int kernel[3][3] = {{-31, 0, 31},
                              {-44, 0, 44},
                              {-31, 0, 31}};
    const double kernel_coef = 1.0 / 126.42;

    const int offset_min = -((kernel_size - 1) / 2);
    const int offset_max =   (kernel_size + 1) / 2;
    pixel_2 pixel1;
    pixel_2 pixel2;
    pixel *mask_mem = calloc(bw * bh, sizeof(pixel));
    if (!mask_mem)
        return -ENOMEM;
    pixel *mask = mask_mem + border + bw * border;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            pixel1 = 0;
            pixel2 = 0;
            for (int k = offset_min; k < offset_max; k++) {
                for (int j = offset_min; j < offset_max; j++) {
                    pixel1 += kernel[j+1][k+1] * *(src + bw*(y+j) + (x+k));
                    pixel2 += kernel[k+1][j+1] * *(src + bw*(y+j) + (x+k));
                }
            }
            pixel1 = pixel1 > 0 ? pixel1 : -pixel1;
            pixel2 = pixel2 > 0 ? pixel2 : -pixel2;
            pixel1 = (pixel_2)(((double)pixel1 * kernel_coef) + 128);
            pixel2 = (pixel_2)(((double)pixel2 * kernel_coef) + 128);
            *(mask + bw*y + x) = (pixel)(pixel1 + pixel2);
            if (*(mask + bw*y + x) > 160)
                *(mask + bw*y + x) = 235;
            else if (*(mask + bw*y + x) > 16)
                *(mask + bw*y + x) = 128;
            else
                *(mask + bw*y + x) = 16;
        }
    }

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (*(mask + bw*y + x) > 16) {
                int pixels = 0;
                for (int k = offset_min; k < offset_max; k++)
                    for (int j = offset_min; j < offset_max; j++)
                        if (*(mask + bw*(y+j) + (x+k)) > 16)
                            pixels++;
                if (pixels < 3)
                    *(mask + bw*y + x) = 16; // Remove false positive
                if (*(mask + bw*y + x) > 16) {
                    if (*(mask + bw*y + x) == 235)
                        *(dst + bw*y + x) = (3 * *(src + bw*y + x) + 1 * *(dst + bw*y + x)) / 4;
                    else
                        *(dst + bw*y + x) = (2 * *(src + bw*y + x) + 3 * *(dst + bw*y + x)) / 5;
                }
            }
        }
    }

    free(mask_mem);
    return 0;
}

static int FUNC(nlmeans_prefilter)(BorderedPlane *src, const int filter_type)
{
    if (filter_type & NLMEANS_PREFILTER_MODE_MEAN3X3   ||
        filter_type & NLMEANS_PREFILTER_MODE_MEAN5X5   ||
        filter_type & NLMEANS_PREFILTER_MODE_MEDIAN3X3 ||
        filter_type & NLMEANS_PREFILTER_MODE_MEDIAN5X5 ||
        filter_type & NLMEANS_PREFILTER_MODE_CSM3X3    ||
        filter_type & NLMEANS_PREFILTER_MODE_CSM5X5) {

        const pixel *mem   = src->mem;
        const pixel *image = src->image;
        const int border   = src->border;
        const int w        = src->w;
        const int h        = src->h;
        const int bw       = w + 2 * border;
        const int bh       = h + 2 * border;

        pixel *mem_pre = malloc((bw * bh + PLANE_TAIL) * sizeof(pixel));
        if (!mem_pre)
            return -ENOMEM;
        pixel *image_pre = mem_pre + border + bw * border;
        memcpy(mem_pre, mem, (bw * bh + PLANE_TAIL) * sizeof(pixel));

        if (filter_type & NLMEANS_PREFILTER_MODE_CSM5X5)
            FUNC(nlmeans_filter_csm)(image, image_pre, w, h, border, 5);
        else if (filter_type & NLMEANS_PREFILTER_MODE_CSM3X3)
            FUNC(nlmeans_filter_csm)(image, image_pre, w, h, border, 3);
        else if (filter_type & NLMEANS_PREFILTER_MODE_MEDIAN5X5)
            FUNC(nlmeans_filter_median)(image, image_pre, w, h, border, 5);
        else if (filter_type & NLMEANS_PREFILTER_MODE_MEDIAN3X3)
            FUNC(nlmeans_filter_median)(image, image_pre, w, h, border, 3);
        else if (filter_type & NLMEANS_PREFILTER_MODE_MEAN5X5)
            FUNC(nlmeans_filter_mean)(image, image_pre, w, h, border, 5);
        else if (filter_type & NLMEANS_PREFILTER_MODE_MEAN3X3)
            FUNC(nlmeans_filter_mean)(image, image_pre, w, h, border, 3);

        if (filter_type & NLMEANS_PREFILTER_MODE_EDGEBOOST) {
            int ret = FUNC(nlmeans_filter_edgeboost)(image, image_pre, w, h, border);
            if (ret < 0) {
                free(mem_pre);
                return ret;
            }
        }

        // Blend source and destination for lesser effect
        int wet = 1;
        int dry = 0;
        if (filter_type & NLMEANS_PREFILTER_MODE_REDUCE50 &&
            filter_type & NLMEANS_PREFILTER_MODE_REDUCE25) {
            wet = 1; dry = 3;
        } else if (filter_type & NLMEANS_PREFILTER_MODE_REDUCE50) {
            wet = 1; dry = 1;
        } else if (filter_type & NLMEANS_PREFILTER_MODE_REDUCE25) {
            wet = 3; dry = 1;
        }
        if (dry > 0) {
            for (int y = 0; y < bh; y++)
                for (int x = 0; x < bw; x++)
                    *(mem_pre + bw*y + x) = (pixel)((wet * *(mem_pre + bw*y + x) + dry * *(mem + bw*y + x)) / (wet + dry));
        }

        FUNC(nlmeans_border)(mem_pre, w, h, border);

        src->mem_pre   = mem_pre;
        src->image_pre = image_pre;
        if (filter_type & NLMEANS_PREFILTER_MODE_PASSTHRU) {
            // The prefiltered plane becomes the output; the original is freed
            free(src->mem);
            src->mem   = src->mem_pre;
            src->image = src->image_pre;
        }
    }
    return 0;
}

static void FUNC(build_integral_scalar)(void *integral, int integral_stride,
                                        const void *in_src_pre,
                                        const void *in_compare_pre,
                                        int w, int border,
                                        int dst_w, int dst_h,
                                        int dx, int dy, int n)
{
    const int bw = w + 2 * border;
    const int n_half = (n-1) / 2;

    const pixel *src_pre     = (const pixel *)in_src_pre;
    const pixel *compare_pre = (const pixel *)in_compare_pre;

    for (int y = 0; y < dst_h + n; y++) {
        const pixel *p1 = src_pre     + (y-n_half   )*bw - n_half;
        const pixel *p2 = compare_pre + (y-n_half+dy)*bw - n_half + dx;
        const integral_pixel *prev = (const integral_pixel *)(integral) + ((y > 0 ? y-1 : 0) * integral_stride);
        integral_pixel *out = (integral_pixel *)(integral) + (y*integral_stride);

        integral_pixel pixel_sum = 0;
        for (int x = 0; x < dst_w + n; x++) {
            // Square the difference between the two pixels and add to the integral sum
            pixel_diff diff = (pixel_diff)(*p1) - (pixel_diff)(*p2);
            pixel_sum += (integral_pixel)((pixel_diff_2)(diff) * diff);
            *out = pixel_sum + (y > 0 ? *prev : 0);

            prev++;
            out++;
            p1++;
            p2++;
        }
    }
}

/*
 * Denoise rows y0..y1 of one plane. frame[0] is the frame being output,
 * frame[1..nframes-1] are the frames that follow it.
 */
static int FUNC(nlmeans_plane)(const HBNLMeans *s,
                               HBNFrame *const *frame, int plane, int nframes,
                               uint8_t *in_dst, int dst_w, int dst_s,
                               int y0, int y1)
{
    pixel *dst = (pixel *)in_dst;
    const double origin_tune       = s->origin_tune[plane];
    const int n                    = s->patch_size[plane];
    const int r                    = s->range[plane];
    const float *exptable          = s->exptable[plane];
    const float  weight_fact_table = s->weight_fact_table[plane];
    const uint64_t diff_max        = s->diff_max[plane];
    const int r_half = (r-1) / 2;
    const int dst_h  = y1 - y0;

    const BorderedPlane *bp = &frame[0]->plane[plane];
    const int w      = bp->w;
    const int border = bp->border;
    const int bw     = w + 2 * border;

    // This slice's rows, inside the bordered planes
    const pixel *src     = (const pixel *)bp->image     + y0 * bw;
    const pixel *src_pre = (const pixel *)bp->image_pre + y0 * bw;

    struct PixelSum *tmp_data = calloc(dst_w * dst_h, sizeof(struct PixelSum));

    const int integral_stride = ((dst_w + n + 15) / 16 * 16) + 2 * 16;
    integral_pixel *const integral_mem = calloc(integral_stride * (dst_h + n + 1), sizeof(integral_pixel));
    if (!tmp_data || !integral_mem) {
        free(tmp_data);
        free(integral_mem);
        return -ENOMEM;
    }
    integral_pixel *const integral = integral_mem + integral_stride + 16;

    for (int f = 0; f < nframes; f++) {
        const pixel *compare     = (const pixel *)frame[f]->plane[plane].image     + y0 * bw;
        const pixel *compare_pre = (const pixel *)frame[f]->plane[plane].image_pre + y0 * bw;

        for (int dy = -r_half; dy <= r_half; dy++) {
            for (int dx = -r_half; dx <= r_half; dx++) {

                // Apply special weight tuning to origin patch
                if (dx == 0 && dy == 0 && f == 0) {
                    for (int y = 0; y < dst_h; y++) {
                        for (int x = 0; x < dst_w; x++) {
                            tmp_data[y*dst_w + x].weight_sum += origin_tune;
                            tmp_data[y*dst_w + x].pixel_sum  += origin_tune * src[y*bw + x];
                        }
                    }
                    continue;
                }

                s->build_integral(integral, integral_stride, src_pre, compare_pre,
                                  w, border, dst_w, dst_h, dx, dy, n);

                // Average displacement
                for (int y = 0; y < dst_h; y++) {
                    const integral_pixel *integral_ptr1 = integral + (y  -1)*integral_stride - 1;
                    const integral_pixel *integral_ptr2 = integral + (y+n-1)*integral_stride - 1;

                    for (int x = 0; x < dst_w; x++) {
                        // Difference between patches
                        const integral_pixel diff = (integral_pixel)(integral_ptr2[n] - integral_ptr2[0] - integral_ptr1[n] + integral_ptr1[0]);

                        // Sum pixel with weight
                        if (diff < diff_max) {
                            const int diffidx = (int)(diff * weight_fact_table);
                            if (diffidx < NLMEANS_EXPSIZE) {
                                const float weight = exptable[diffidx];
                                tmp_data[y*dst_w + x].weight_sum += weight;
                                tmp_data[y*dst_w + x].pixel_sum  += weight * compare[(y+dy)*bw + x + dx];
                            }
                        }
                        integral_ptr1++;
                        integral_ptr2++;
                    }
                }
            }
        }
    }

    // Copy image without border
    pixel result;
    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            result = (pixel)(tmp_data[y*dst_w + x].pixel_sum / tmp_data[y*dst_w + x].weight_sum);
            *(dst + (y0 + y)*dst_s + x) = result ? result : *(src + y*bw + x);
        }
    }

    free(tmp_data);
    free(integral_mem);
    return 0;
}
