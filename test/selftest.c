/* make check: noise goes down, sliced output equals one-job output, API basics. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hbnlmeans.h"

#define W 211   /* odd sizes on purpose */
#define H 127
#define FRAMES 5

static int fails;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fails++; } else printf("ok:   %s\n", msg); } while (0)

static unsigned rng = 12345;
static int noise(void) { rng = rng * 1103515245 + 12345; return (int)((rng >> 16) % 13) - 6; }

static int clean(int x, int y, int c) { return 60 + ((x / 16 + y / 16) % 2) * 100 + c * 10; }

/* Run all frames through, return mean squared error against the clean image. */
static double run(int depth, const char *preset, const char *tune, int jobs, uint16_t *keep)
{
    const int cw = (W + 1) / 2, ch = (H + 1) / 2, bps = depth > 8 ? 2 : 1;
    const int pw[3] = { W, cw, cw }, ph[3] = { H, ch, ch };
    HBNLMParams p;
    hbnlm_params_init(&p);
    if (preset && hbnlm_params_preset(&p, preset, tune) < 0)
        return -1;
    HBNLMeans *nlm = hbnlm_create(&p, W, H, 1, 1, depth);
    if (!nlm)
        return -1;

    uint8_t *in[FRAMES][3], *out[3];
    ptrdiff_t stride[3];
    for (int c = 0; c < 3; c++) {
        stride[c] = (ptrdiff_t)pw[c] * bps + 32;
        out[c] = calloc(stride[c], ph[c]);
        for (int f = 0; f < FRAMES; f++) {
            in[f][c] = calloc(stride[c], ph[c]);
            for (int y = 0; y < ph[c]; y++)
                for (int x = 0; x < pw[c]; x++) {
                    int v = clean(x, y, c) + noise();
                    if (bps == 2) ((uint16_t *)(in[f][c] + y * stride[c]))[x] = (uint16_t)(v << (depth - 8));
                    else in[f][c][y * stride[c] + x] = (uint8_t)v;
                }
        }
    }

    double err = 0;
    long n = 0, k = 0;
    int pushed = 0, done = 0;
    while (done < FRAMES) {
        if (pushed < FRAMES && hbnlm_queued(nlm) < hbnlm_frames_needed(nlm)) {
            const uint8_t *src[3] = { in[pushed][0], in[pushed][1], in[pushed][2] };
            hbnlm_push(nlm, src, stride);
            pushed++;
            continue;
        }
        for (int j = 0; j < jobs; j++)
            hbnlm_filter_slice(nlm, out, stride, j, jobs);
        hbnlm_pop(nlm);
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < ph[c]; y++)
                for (int x = 0; x < pw[c]; x++) {
                    int v = bps == 2 ? ((uint16_t *)(out[c] + y * stride[c]))[x] >> (depth - 8)
                                     : out[c][y * stride[c] + x];
                    double d = v - clean(x, y, c);
                    err += d * d; n++;
                    if (keep) keep[k++] = (uint16_t)v;
                }
        done++;
    }

    for (int c = 0; c < 3; c++) {
        free(out[c]);
        for (int f = 0; f < FRAMES; f++)
            free(in[f][c]);
    }
    hbnlm_free(&nlm);
    return err / n;
}

int main(void)
{
    const double noise_mse = 14; /* uniform -6..6 */
    const long total = (long)FRAMES * (W * H + 2 * ((W + 1) / 2) * ((H + 1) / 2));
    uint16_t *a = malloc(total * sizeof(*a)), *b = malloc(total * sizeof(*b));

    printf("libhbnlmeans %u.%u.%u\n", hbnlm_version() >> 16, (hbnlm_version() >> 8) & 255, hbnlm_version() & 255);

    rng = 1; double m1 = run(8, "medium", "film", 1, a);
    printf("      noise MSE %.1f, after medium/film %.1f\n", noise_mse, m1);
    rng = 1; double m7 = run(8, "medium", "film", 7, b);
    CHECK(m1 > 0 && m1 < noise_mse / 2, "8-bit medium/film removes at least half the noise");
    CHECK(!memcmp(a, b, total * sizeof(*a)), "7 slices give the same output as 1");

    rng = 1; double m10 = run(10, "strong", "animation", 3, NULL);
    CHECK(m10 > 0 && m10 < noise_mse / 2, "10-bit strong/animation removes at least half the noise");

    HBNLMParams p;
    hbnlm_params_init(&p);
    CHECK(hbnlm_params_parse(&p, "y-strength=7:cb-origin-tune=0.5:cr-frame-count=3") == 0 &&
          p.strength[0] == 7 && p.origin_tune[1] == 0.5 && p.frame_count[2] == 3, "custom string parses");
    CHECK(hbnlm_params_parse(&p, "y-bogus=1") < 0, "unknown key is rejected");
    CHECK(hbnlm_params_preset(&p, "medium", "nonsense") < 0, "unknown tune is rejected");

    hbnlm_params_init(&p);
    p.strength[0] = 4;
    HBNLMeans *nlm = hbnlm_create(&p, 64, 64, 1, 1, 8);
    HBNLMParams used;
    hbnlm_get_params(nlm, &used);
    CHECK(used.strength[2] == 4 && used.patch_size[2] == 7 && hbnlm_frames_needed(nlm) == 2,
          "values cascade Y -> Cb -> Cr, defaults fill the rest");
    hbnlm_free(&nlm);

    free(a); free(b);
    printf(fails ? "%d FAILED\n" : "all passed\n", fails);
    return fails != 0;
}
