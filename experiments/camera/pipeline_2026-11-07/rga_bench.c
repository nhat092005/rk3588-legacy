// RGA timing on Orange Pi 5: NV12 -> RGB888 cases, virtual-address and dma-buf (dma_heap) buffers.
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

static int heap = -1;

static int alloc_buf(buf_t *b, size_t size, int use_fd) {
    b->size = size;
    if (!use_fd) {
        b->fd = -1;
        return posix_memalign(&b->va, 4096, size);
    }
    struct dh_alloc a = {.len = size, .fd_flags = O_RDWR | O_CLOEXEC};
    if (ioctl(heap, DH_ALLOC, &a) < 0) { perror("DH_ALLOC"); return -1; }
    b->fd = a.fd;
    b->va = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, a.fd, 0);
    return b->va == MAP_FAILED ? -1 : 0;
}

static rga_buffer_t wrap(buf_t *b, int w, int h, int fmt) {
    if (b->fd >= 0) return wrapbuffer_fd(b->fd, w, h, fmt);
    return wrapbuffer_virtualaddr(b->va, w, h, fmt);
}

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

enum { OP_RESIZE, OP_CVT, OP_LETTERBOX };

static int run_case(const char *name, int sw, int sh, int dw, int dh, int op, int use_fd) {
    buf_t s, d;
    if (alloc_buf(&s, (size_t)sw * sh * 3 / 2, use_fd) || alloc_buf(&d, (size_t)dw * dh * 3, use_fd)) return -1;
    for (size_t i = 0; i < s.size; i++) ((uint8_t *)s.va)[i] = (uint8_t)(i % 251);
    memset(d.va, 0, d.size);
    rga_buffer_t src = wrap(&s, sw, sh, RK_FORMAT_YCbCr_420_SP);
    rga_buffer_t dst = wrap(&d, dw, dh, RK_FORMAT_RGB_888);
    rga_buffer_t pat;
    memset(&pat, 0, sizeof(pat));
    im_rect srect = {0, 0, sw, sh}, drect = {0, (dh - 360) / 2, dw, 360}, prect = {0, 0, 0, 0};
    im_rect full = {0, 0, dw, dh};
    enum { N = 300, WARM = 30 };
    double t[N];
    IM_STATUS st = IM_STATUS_SUCCESS;
    for (int i = 0; i < N + WARM; i++) {
        double t0 = now_ms();
        if (op == OP_RESIZE) st = imresize(src, dst);
        else if (op == OP_CVT) st = imcvtcolor(src, dst, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGB_888);
        else {
            st = imfill(dst, full, 0x727272);
            if (st == IM_STATUS_SUCCESS) st = improcess(src, dst, pat, srect, drect, prect, IM_SYNC);
        }
        double t1 = now_ms();
        if (st != IM_STATUS_SUCCESS) { printf("%-40s %-4s FAIL %s\n", name, use_fd ? "fd" : "va", imStrError(st)); return -1; }
        if (i >= WARM) t[i - WARM] = t1 - t0;
    }
    qsort(t, N, sizeof(double), cmp);
    double sum = 0;
    for (int i = 0; i < N; i++) sum += t[i];
    printf("%-40s %-4s median %.3f  p95 %.3f  mean %.3f ms\n", name, use_fd ? "fd" : "va", t[N / 2], t[(int)(N * 0.95)], sum / N);
    return 0;
}

int main(void) {
    heap = open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
    if (heap < 0) perror("open dma_heap/system");
    printf("librga: %s\n", querystring(RGA_VERSION));
    for (int use_fd = 0; use_fd <= 1; use_fd++) {
        if (use_fd && heap < 0) break;
        run_case("NV12 1920x1080 -> RGB888 640x360 resize", 1920, 1080, 640, 360, OP_RESIZE, use_fd);
        run_case("NV12 1920x1080 -> RGB888 1920x1080 cvt", 1920, 1080, 1920, 1080, OP_CVT, use_fd);
        run_case("NV12 640x360 -> RGB888 640x360 cvt", 640, 360, 640, 360, OP_CVT, use_fd);
        run_case("NV12 640x360 -> RGB888 640x640 letterbox", 640, 360, 640, 640, OP_LETTERBOX, use_fd);
        run_case("NV12 1920x1080 -> RGB888 640x640 letterbox", 1920, 1080, 640, 640, OP_LETTERBOX, use_fd);
        run_case("NV12 4224x3136 -> RGB888 640x640 resize", 4224, 3136, 640, 640, OP_RESIZE, use_fd);
    }
    return 0;
}
