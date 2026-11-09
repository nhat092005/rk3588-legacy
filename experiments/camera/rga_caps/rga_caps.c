// What each RGA core (RGA3_0, RGA3_1, RGA2) accepts and how long it takes, with dma-buf buffers and the core forced by imconfig.
// The core that really ran is read from /proc/interrupts (irq 40 RGA3_0, 41 RGA3_1, 111 RGA2).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include "im2d.h"
#include "rga.h"

struct dh_alloc { uint64_t len; uint32_t fd; uint32_t fd_flags; uint64_t heap_flags; };
#define DH_ALLOC _IOWR('H', 0x0, struct dh_alloc)
typedef struct { void *va; int fd; size_t size; } buf_t;
static int heap;

static int alloc_buf(buf_t *b, size_t size) {
    struct dh_alloc a = {.len = size, .fd_flags = O_RDWR | O_CLOEXEC};
    if (ioctl(heap, DH_ALLOC, &a) < 0) return -1;
    b->fd = a.fd; b->size = size;
    b->va = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, a.fd, 0);
    return b->va == MAP_FAILED ? -1 : 0;
}
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static long irq_count(int irq) {
    FILE *f = fopen("/proc/interrupts", "r"); char line[1024]; long total = -1;
    while (f && fgets(line, sizeof line, f)) {
        int n; if (sscanf(line, " %d:", &n) == 1 && n == irq) {
            total = 0; char *p = strchr(line, ':') + 1; char *e;
            for (int i = 0; i < 8; i++) { long v = strtol(p, &e, 10); if (e == p) break; total += v; p = e; }
        }
    }
    if (f) fclose(f); return total;
}
static const uint64_t CORES[3] = {IM_SCHEDULER_RGA3_CORE0, IM_SCHEDULER_RGA3_CORE1, IM_SCHEDULER_RGA2_CORE0};
static const char *CNAME[3] = {"RGA3_0", "RGA3_1", "RGA2"};
static const int CIRQ[3] = {40, 41, 111};
static size_t bytes_of(int w, int h, int fmt) {
    switch (fmt) { case RK_FORMAT_YCbCr_420_SP: return (size_t)w * h * 3 / 2; case RK_FORMAT_RGB_888: return (size_t)w * h * 3; case RK_FORMAT_RGBA_8888: return (size_t)w * h * 4; default: return (size_t)w * h * 4; }
}

typedef IM_STATUS (*op_fn)(rga_buffer_t src, rga_buffer_t dst, int dw, int dh);
static IM_STATUS op_cvt_nv12_rgb(rga_buffer_t s, rga_buffer_t d, int dw, int dh) { return imcvtcolor(s, d, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGB_888); }
static IM_STATUS op_cvt_rgb_nv12(rga_buffer_t s, rga_buffer_t d, int dw, int dh) { return imcvtcolor(s, d, RK_FORMAT_RGB_888, RK_FORMAT_YCbCr_420_SP); }
static IM_STATUS op_resize(rga_buffer_t s, rga_buffer_t d, int dw, int dh) { return imresize(s, d); }
static IM_STATUS op_letterbox(rga_buffer_t s, rga_buffer_t d, int dw, int dh) {
    rga_buffer_t pat; memset(&pat, 0, sizeof pat);
    im_rect full = {0, 0, dw, dh}, srect = {0, 0, 640, 360}, drect = {0, (dh - 360) / 2, 640, 360}, prect = {0, 0, 0, 0};
    IM_STATUS st = imfill(d, full, 0x727272); if (st != IM_STATUS_SUCCESS) return st;
    return improcess(s, d, pat, srect, drect, prect, IM_SYNC);
}
static IM_STATUS op_fill(rga_buffer_t s, rga_buffer_t d, int dw, int dh) { im_rect r = {100, 100, 200, 100}; return imfill(d, r, 0x00ff00); }
static IM_STATUS op_rect(rga_buffer_t s, rga_buffer_t d, int dw, int dh) { im_rect r = {100, 100, 200, 100}; return imrectangle(d, r, 0x0000ff, 2); }
static IM_STATUS op_rect_filled(rga_buffer_t s, rga_buffer_t d, int dw, int dh) { im_rect r = {100, 100, 200, 100}; return imrectangle(d, r, 0x0000ff, -1); }
static IM_STATUS op_blend(rga_buffer_t s, rga_buffer_t d, int dw, int dh) { return imblend(s, d); }
static IM_STATUS op_border(rga_buffer_t s, rga_buffer_t d, int dw, int dh) { return immakeBorder(s, d, 140, 140, 0, 0, IM_BORDER_CONSTANT, 114); }

static void run(const char *name, op_fn fn, int sw, int sh, int sfmt, int dw, int dh, int dfmt, int init_dst_from_src_like) {
    buf_t s, d;
    if (alloc_buf(&s, bytes_of(sw, sh, sfmt)) || alloc_buf(&d, bytes_of(dw, dh, dfmt))) { printf("%-46s alloc fail\n", name); return; }
    for (size_t i = 0; i < s.size; i++) ((uint8_t *)s.va)[i] = (uint8_t)(i % 251);
    for (int c = 0; c < 3; c++) {
        memset(d.va, 0x40, d.size);
        uint8_t *before = malloc(d.size); memcpy(before, d.va, d.size);
        rga_buffer_t src = wrapbuffer_fd(s.fd, sw, sh, sfmt), dst = wrapbuffer_fd(d.fd, dw, dh, dfmt);
        imconfig(IM_CONFIG_SCHEDULER_CORE, CORES[c]);
        IM_STATUS st = fn(src, dst, dw, dh);
        if (st != IM_STATUS_SUCCESS) { printf("%-46s %-7s FAIL %s\n", name, CNAME[c], imStrError(st)); free(before); imconfig(IM_CONFIG_SCHEDULER_CORE, 0); continue; }
        size_t changed = 0; for (size_t i = 0; i < d.size; i++) changed += before[i] != ((uint8_t *)d.va)[i];
        long i0[3]; for (int k = 0; k < 3; k++) i0[k] = irq_count(CIRQ[k]);
        enum { N = 100 }; double t[N];
        for (int i = 0; i < N; i++) { double a = now_ms(); fn(src, dst, dw, dh); t[i] = now_ms() - a; }
        long i1[3]; for (int k = 0; k < 3; k++) i1[k] = irq_count(CIRQ[k]);
        qsort(t, N, sizeof(double), cmpd);
        char ran[64] = ""; for (int k = 0; k < 3; k++) if (i1[k] - i0[k] > 0) { strcat(ran, CNAME[k]); strcat(ran, " "); }
        printf("%-46s %-7s OK median %.3f ms changed %.0f%% ran: %s\n", name, CNAME[c], t[N / 2], 100.0 * changed / d.size, ran[0] ? ran : "(none counted)");
        free(before); imconfig(IM_CONFIG_SCHEDULER_CORE, 0);
    }
    munmap(s.va, s.size); munmap(d.va, d.size); close(s.fd); close(d.fd);
}

int main(void) {
    heap = open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
    if (heap < 0) { perror("dma_heap"); return 1; }
    const int NV = RK_FORMAT_YCbCr_420_SP, RG = RK_FORMAT_RGB_888, RA = RK_FORMAT_RGBA_8888;
    run("cvt NV12 640x360 -> RGB888", op_cvt_nv12_rgb, 640, 360, NV, 640, 360, RG, 0);
    run("letterbox NV12 640x360 -> RGB888 640x640", op_letterbox, 640, 360, NV, 640, 640, RG, 0);
    run("letterbox NV12 640x360 -> RGB888 640x384", op_letterbox, 640, 360, NV, 640, 384, RG, 0);
    run("makeBorder NV12 640x360 -> RGB888? (src RGB) 640x640", op_border, 640, 360, RG, 640, 640, RG, 0);
    run("resize NV12 1920x1080 -> NV12 640x360", op_resize, 1920, 1080, NV, 640, 360, NV, 0);
    run("resize NV12 1920x1080 -> RGB888 640x360", op_resize, 1920, 1080, NV, 640, 360, RG, 0);
    run("cvt RGB888 -> NV12 1920x1080", op_cvt_rgb_nv12, 1920, 1080, RG, 1920, 1080, NV, 0);
    run("cvt RGB888 -> NV12 640x360", op_cvt_rgb_nv12, 640, 360, RG, 640, 360, NV, 0);
    run("imfill rect on NV12 640x360", op_fill, 640, 360, NV, 640, 360, NV, 0);
    run("imrectangle t=2 on NV12 640x360", op_rect, 640, 360, NV, 640, 360, NV, 0);
    run("imrectangle t=-1 on NV12 640x360", op_rect_filled, 640, 360, NV, 640, 360, NV, 0);
    run("imrectangle t=2 on NV12 1920x1080", op_rect, 1920, 1080, NV, 1920, 1080, NV, 0);
    run("imrectangle t=2 on RGB888 640x360", op_rect, 640, 360, RG, 640, 360, RG, 0);
    run("imrectangle t=2 on RGBA8888 640x360", op_rect, 640, 360, RA, 640, 360, RA, 0);
    run("imblend RGBA 640x360 over NV12 640x360", op_blend, 640, 360, RA, 640, 360, NV, 0);
    run("imblend RGBA 640x360 over RGB888 640x360", op_blend, 640, 360, RA, 640, 360, RG, 0);
    return 0;
}
