/* Drives HandBrake's unmodified NLMeans template over raw yuv420p 8-bit frames,
 * frame by frame the way libhb/nlmeans.c does (frame i compared with i..i+n-1,
 * fewer at the end). Usage: harness W H "y-strength=..:.." in.yuv out.yuv */

static void resolve(hb_filter_private_t *pv, const char *settings);

static void read_params(hb_filter_private_t *pv, const char *settings)
{
    char buf[1024];
    snprintf(buf, sizeof(buf), "%s", settings);
    for (char *tok = strtok(buf, ":"); tok; tok = strtok(NULL, ":")) {
        char *eq = strchr(tok, '=');
        if (!eq) continue;
        *eq = 0;
        const char *k = tok, *v = eq + 1;
        int c = !strncmp(k, "y-", 2) ? 0 : !strncmp(k, "cb-", 3) ? 1 : !strncmp(k, "cr-", 3) ? 2 : -1;
        if (c < 0) continue;
        const char *p = strchr(k, '-') + 1;
        if      (!strcmp(p, "strength"))    pv->strength[c]    = atof(v);
        else if (!strcmp(p, "origin-tune")) pv->origin_tune[c] = atof(v);
        else if (!strcmp(p, "patch-size"))  pv->patch_size[c]  = atoi(v);
        else if (!strcmp(p, "range"))       pv->range[c]       = atoi(v);
        else if (!strcmp(p, "frame-count")) pv->nframes[c]     = atoi(v);
        else if (!strcmp(p, "prefilter"))   pv->prefilter[c]   = atoi(v);
    }
}

int main(int argc, char **argv)
{
    if (argc < 6) return 1;
    const int W = atoi(argv[1]), H = atoi(argv[2]);
    static hb_filter_private_t pvs;
    hb_filter_private_t *pv = &pvs;
    const int D = getenv("DEPTH") ? atoi(getenv("DEPTH")) : 8; const int B = D > 8 ? 2 : 1;
    pv->depth = D; pv->bps = B; pv->max_value = (1<<D)-1;
    pv->functions.build_integral = D > 8 ? build_integral_scalar_16 : build_integral_scalar_8;
    resolve(pv, argv[3]);

    const int pw[3] = { W, (W + 1) / 2, (W + 1) / 2 };
    const int ph[3] = { H, (H + 1) / 2, (H + 1) / 2 };
    const size_t fsz = ((size_t)pw[0]*ph[0] + 2*(size_t)pw[1]*ph[1]) * B;

    FILE *fi = fopen(argv[4], "rb"), *fo = fopen(argv[5], "wb");
    if (!fi || !fo) return 2;
    fseek(fi, 0, SEEK_END);
    const int total = ftell(fi) / fsz;
    fseek(fi, 0, SEEK_SET);

    hb_lock_t lock = 0;
    Frame *frames = calloc(total, sizeof(Frame));
    uint8_t *raw = malloc(fsz), *out = malloc(fsz);
    for (int i = 0; i < total; i++) {
        if (fread(raw, 1, fsz, fi) != fsz) return 3;
        size_t off = 0;
        for (int c = 0; c < 3; c++) {
            const int border = ((pv->patch_size[c] + 2) / 2 + 15) / 16 * 16;
            frames[i].plane[c].mutex = &lock;
            (B>1?nlmeans_alloc_16:nlmeans_alloc_8)(raw + off, pw[c], pw[c], ph[c], &frames[i].plane[c], border);
            off += (size_t)pw[c]*ph[c]*B;
        }
    }
    for (int i = 0; i < total; i++) {
        size_t off = 0;
        for (int c = 0; c < 3; c++) {
            uint8_t *dst = out + off;
            void (*pre)(BorderedPlane*,const int) = B>1?nlmeans_prefilter_16:nlmeans_prefilter_8;
            void (*deb)(const BorderedPlane*,void*,const int,const int,const int) = B>1?nlmeans_deborder_16:nlmeans_deborder_8;
            if (pv->prefilter[c] & NLMEANS_PREFILTER_MODE_PASSTHRU) {
                pre(&frames[i].plane[c], pv->prefilter[c]);
                deb(&frames[i].plane[c], dst, pw[c], pw[c], ph[c]);
            } else if (pv->strength[c] == 0) {
                deb(&frames[i].plane[c], dst, pw[c], pw[c], ph[c]);
            } else {
                // HandBrake reads the current frame's prefiltered image before it prefilters it
                // (timing-dependent with its frame threads); PREFILTER_FIRST gives the intended order
                if (getenv("PREFILTER_FIRST")) pre(&frames[i].plane[c], pv->prefilter[c]);
                int nf = total - i < pv->nframes[c] ? total - i : pv->nframes[c];
                (B>1?nlmeans_plane_16:nlmeans_plane_8)(&pv->functions, &frames[i], pv->prefilter[c], c, nf,
                                dst, pw[c], pw[c], ph[c], pv->origin_tune[c],
                                pv->patch_size[c], pv->range[c], pv->exptable[c],
                                pv->weight_fact_table[c], pv->diff_max[c]);
            }
            off += (size_t)pw[c]*ph[c]*B;
        }
        fwrite(out, 1, fsz, fo);
    }
    fclose(fo);
    return 0;
}
