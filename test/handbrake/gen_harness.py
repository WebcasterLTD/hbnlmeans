# Build harness.c from HandBrake's own, unmodified libhb sources:
# the structs and parameter code of nlmeans.c plus templates/nlmeans_template.c.
#   python3 gen_harness.py /path/to/HandBrake/libhb > harness.c
import re, sys
hb = sys.argv[1]
src = open(hb + '/nlmeans.c').read()
head = src[src.index('#define NLMEANS_STRENGTH_LUMA_DEFAULT'):src.index('static int nlmeans_init(hb_filter_object_t')]
init = src[src.index('static int nlmeans_init(hb_filter_object_t *filter,'):]
body = init[init.index('    // Mark parameters unset'):init.index('    // Threads')]
body = re.sub(r'    // Read user parameters\n    if \(filter->settings != NULL\)\n    \{.*?\n    \}\n',
              '    read_params(pv, settings);\n', body, flags=re.S)
if 'read_params' not in body:
    sys.exit('gen_harness: HandBrake nlmeans.c changed shape; update this script')
sys.stdout.write('''#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
typedef int hb_lock_t;
static void hb_lock(hb_lock_t *l){(void)l;} static void hb_unlock(hb_lock_t *l){(void)l;}
typedef struct hb_buffer_s hb_buffer_t;
typedef struct { int segment; void *taskset; } taskset_thread_arg_t;
typedef struct { int d; } taskset_t;
typedef struct { int pix_fmt; } hb_filter_init_t;
typedef struct hb_filter_private_s hb_filter_private_t;
#include "handbrake/nlmeans.h"
''' + head + '''
#define BIT_DEPTH 8
#include "templates/nlmeans_template.c"
#undef BIT_DEPTH
#define BIT_DEPTH 16
#include "templates/nlmeans_template.c"
#undef BIT_DEPTH
#include "harness_main.c"
static void resolve(hb_filter_private_t *pv, const char *settings)
{
''' + body + '}\n')
