/* On-board latency of a .rknn with the RKNN C API (no PC link): the route UG232 p.175 recommends for a realistic frame rate.
 *
 * Build on the board (needs rknn_api.h from 3rdparty/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/include):
 *   gcc -O2 -I<include dir> rknn_perf_bench.c -o rknn_perf_bench -L/usr/lib64 -lrknnrt -lm
 * Run (zero UINT8 NHWC input, converted by the runtime outside the timed rknn_run):
 *   ./rknn_perf_bench <model.rknn> <core_mask: 1 = CORE_0, 7 = CORE_0_1_2> <warmup> <loops> <out_prefix> [detail|-] [gap_ms] [input]
 * gap_ms sleeps after every run so the NPU can idle between runs (tests cold start effects), default 0 = back to back.
 * input is a raw UINT8 NHWC file of exactly n_elems bytes, or "random" (fixed seed), default all zeros.
 * Writes <out_prefix>.csv (run, wall_us around rknn_run, run_duration_us from RKNN_QUERY_PERF_RUN, t_us start time on the
 * monotonic clock, cpu = sched_getcpu() after the run). With "detail" the
 * context is created with RKNN_FLAG_COLLECT_PERF_MASK and every run's RKNN_QUERY_PERF_DETAIL text is appended to
 * <out_prefix>_detail.txt (same table format as eval_perf with perf_debug=True). Detail mode slows the runs, like perf_debug.
 */
#define _GNU_SOURCE
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rknn_api.h"

static int64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

static unsigned char *read_file(const char *path, long *size)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(*size);
    if (buf && fread(buf, 1, *size, f) != (size_t)*size) { free(buf); buf = NULL; }
    fclose(f);
    return buf;
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        fprintf(stderr, "usage: %s model.rknn core_mask warmup loops out_prefix [detail|-] [gap_ms] [input file|random]\n", argv[0]);
        return 2;
    }
    const char *model_path = argv[1];
    uint32_t core_mask = (uint32_t)strtoul(argv[2], NULL, 10);
    int warmup = atoi(argv[3]), loops = atoi(argv[4]);
    const char *prefix = argv[5];
    int detail = argc > 6 && strcmp(argv[6], "detail") == 0;
    long gap_ms = argc > 7 ? atol(argv[7]) : 0;
    const char *input = argc > 8 ? argv[8] : NULL;

    long size = 0;
    unsigned char *model = read_file(model_path, &size);
    if (!model) { fprintf(stderr, "cannot read %s\n", model_path); return 1; }

    rknn_context ctx;
    int ret = rknn_init(&ctx, model, (uint32_t)size, detail ? RKNN_FLAG_COLLECT_PERF_MASK : 0, NULL);
    if (ret < 0) { fprintf(stderr, "rknn_init failed %d\n", ret); return 1; }
    ret = rknn_set_core_mask(ctx, (rknn_core_mask)core_mask);
    if (ret < 0) { fprintf(stderr, "rknn_set_core_mask failed %d\n", ret); return 1; }

    rknn_sdk_version ver;
    rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver));
    printf("api %s driver %s core_mask %u detail %d input %s\n", ver.api_version, ver.drv_version, core_mask, detail, input ? input : "zeros");

    rknn_tensor_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
    if (ret < 0) { fprintf(stderr, "query input attr failed %d\n", ret); return 1; }
    /* pass_through = 1 was rejected by rknn_inputs_set (expects 4915200 bytes for this 1228800 byte INT8 input), so the
       zero image is given as UINT8 NHWC and converted by the runtime inside rknn_inputs_set, outside the timed rknn_run. */
    unsigned char *zeros = calloc(1, attr.n_elems);
    if (input && strcmp(input, "random") == 0) {
        srand(12345);
        for (uint32_t k = 0; k < attr.n_elems; k++) zeros[k] = (unsigned char)(rand() & 0xff);
    } else if (input) {
        long isz = 0;
        unsigned char *img = read_file(input, &isz);
        if (!img || isz != (long)attr.n_elems) { fprintf(stderr, "input %s must be %u bytes\n", input, attr.n_elems); return 1; }
        memcpy(zeros, img, attr.n_elems);
        free(img);
    }
    rknn_input in;
    memset(&in, 0, sizeof(in));
    in.index = 0;
    in.buf = zeros;
    in.size = attr.n_elems;
    in.pass_through = 0;
    in.type = RKNN_TENSOR_UINT8;
    in.fmt = RKNN_TENSOR_NHWC;
    ret = rknn_inputs_set(ctx, 1, &in);
    if (ret < 0) { fprintf(stderr, "rknn_inputs_set failed %d\n", ret); return 1; }

    char path[1024];
    snprintf(path, sizeof(path), "%s.csv", prefix);
    FILE *csv = fopen(path, "w");
    FILE *det = NULL;
    if (detail) {
        snprintf(path, sizeof(path), "%s_detail.txt", prefix);
        det = fopen(path, "w");
    }
    fprintf(csv, "run,wall_us,run_duration_us,t_us,cpu\n");

    for (int i = 0; i < warmup + loops; i++) {
        int64_t t0 = now_us();
        ret = rknn_run(ctx, NULL);
        int64_t wall = now_us() - t0;
        if (ret < 0) { fprintf(stderr, "rknn_run failed %d\n", ret); return 1; }
        rknn_perf_run pr;
        memset(&pr, 0, sizeof(pr));
        rknn_query(ctx, RKNN_QUERY_PERF_RUN, &pr, sizeof(pr));
        if (i >= warmup) fprintf(csv, "%d,%lld,%lld,%lld,%d\n", i - warmup, (long long)wall, (long long)pr.run_duration, (long long)t0, sched_getcpu());
        if (detail && i >= warmup) {
            rknn_perf_detail pd;
            memset(&pd, 0, sizeof(pd));
            if (rknn_query(ctx, RKNN_QUERY_PERF_DETAIL, &pd, sizeof(pd)) >= 0 && pd.perf_data)
                fprintf(det, "=== run %d ===\n%s\n", i - warmup, pd.perf_data);
        }
        if (gap_ms > 0) {
            struct timespec gap = { gap_ms / 1000, (gap_ms % 1000) * 1000000L };
            nanosleep(&gap, NULL);
        }
    }
    fclose(csv);
    if (det) fclose(det);
    rknn_destroy(ctx);
    free(zeros);
    free(model);
    return 0;
}
