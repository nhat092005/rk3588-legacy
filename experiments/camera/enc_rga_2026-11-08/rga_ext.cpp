// RGA extra cases: usage rga_ext <op> <core: 0=RGA3_0 1=RGA3_1 2=RGA2> <W> <H> <N> <secs>
// ops: place, rect (N imrectangle calls), rectarr (one imrectangleArray with N rects), label (N small RGBA blends improcess)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <vector>
#include <linux/dma-buf.h>
#include <algorithm>
#include "im2d.h"
#include "rga.h"
struct dh_alloc { uint64_t len; uint32_t fd; uint32_t fd_flags; uint64_t heap_flags; };
#define DH_ALLOC _IOWR('H', 0x0, struct dh_alloc)
static int heap;
static int alloc_buf(size_t size, int *fd, void **va) {
    struct dh_alloc a = {}; a.len = size; a.fd_flags = O_RDWR | O_CLOEXEC;
    if (ioctl(heap, DH_ALLOC, &a) < 0) return -1;
    *fd = a.fd; *va = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, a.fd, 0);
    return *va == MAP_FAILED ? -1 : 0;
}
static double now_ms() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static long irq_count(int irq) {
    FILE *f = fopen("/proc/interrupts", "r"); char line[1024]; long total = -1;
    while (f && fgets(line, sizeof line, f)) { int n; if (sscanf(line, " %d:", &n) == 1 && n == irq) { total = 0; char *p = strchr(line, ':') + 1, *e; for (int i = 0; i < 8; i++) { long v = strtol(p, &e, 10); if (e == p) break; total += v; p = e; } } }
    if (f) fclose(f); return total;
}
static int alloc_low(size_t size, int *fd, void **va, int w, int h, int fmt) {
    static int drained = 0;
    if (!drained) { int f; void *v; for (int i = 0; i < 4; i++) { if (alloc_buf(64u << 20, &f, &v) == 0) memset(v, 1, 64u << 20); } drained = 1; } // hold dummy buffers: use up free pages above 4G first
    for (int t = 0; t < 400; t++) {
        if (alloc_buf(size, fd, va)) return -1;
        memset(*va, 0, size);
        rga_buffer_t b = wrapbuffer_fd(*fd, w, h, fmt);
        im_rect r = {0, 0, (w < 64 ? w : 64) & ~3, (h < 64 ? h : 64) & ~3};
        imconfig(IM_CONFIG_SCHEDULER_CORE, IM_SCHEDULER_RGA2_CORE0);
        IM_STATUS st = imfill(b, r, 0x0000ff);
        if (st == IM_STATUS_SUCCESS) { fprintf(stderr, "[alloc ok after %d tries]\n", t + 1); return 0; }
        /* keep failed buffer so the next allocation cannot reuse its pages */
    }
    return -1;
}
int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "usage\n"); return 2; }
    const char *op = argv[1]; int core = atoi(argv[2]), W = atoi(argv[3]), H = atoi(argv[4]), N = atoi(argv[5]); double secs = atof(argv[6]);
    static const uint64_t CORES[3] = {IM_SCHEDULER_RGA3_CORE0, IM_SCHEDULER_RGA3_CORE1, IM_SCHEDULER_RGA2_CORE0};
    static const char *CN[3] = {"RGA3_0", "RGA3_1", "RGA2"}; static const int CI[3] = {40, 41, 111};
    heap = open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC); if (heap < 0) { perror("heap"); return 1; }
    int sfd, dfd, lfd; void *sva, *dva, *lva;
    int NV = RK_FORMAT_YCbCr_420_SP, RG = RK_FORMAT_RGB_888, RA = RK_FORMAT_RGBA_8888;
    int isplace = !strcmp(op, "place");
    int dw = isplace ? 640 : W, dh = isplace ? 384 : H, dfmt = isplace ? RG : NV;
    int sw = 640, sh = 360;
    if (alloc_low((size_t)sw * sh * 3 / 2, &sfd, &sva, sw, sh, NV) || alloc_low(isplace ? (size_t)dw * dh * 3 : (size_t)dw * dh * 3 / 2, &dfd, &dva, dw, dh, dfmt) || alloc_low(96 * 20 * 4, &lfd, &lva, 96, 20, RA)) { printf("alloc under 4G failed\n"); return 1; }
    memset(sva, 0x55, (size_t)sw * sh * 3 / 2);
    { struct dma_buf_sync sy; sy.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE; ioctl(dfd, DMA_BUF_IOCTL_SYNC, &sy); memset(dva, 0x40, isplace ? (size_t)dw * dh * 3 : (size_t)dw * dh * 3 / 2); sy.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE; ioctl(dfd, DMA_BUF_IOCTL_SYNC, &sy); }
    for (int i = 0; i < 96 * 20; i++) ((uint32_t *)lva)[i] = 0xC0202020 + (i & 0xff);
    rga_buffer_t src = wrapbuffer_fd(sfd, sw, sh, NV), dst = wrapbuffer_fd(dfd, dw, dh, dfmt), lab = wrapbuffer_fd(lfd, 96, 20, RA);
    rga_buffer_t pat; memset(&pat, 0, sizeof pat);
    std::vector<im_rect> rects; srand(1);
    for (int i = 0; i < N; i++) { int rw = 16 + rand() % (dw / 6), rh = 16 + rand() % (dh / 4); int rx = rand() % (dw - rw - 8), ry = rand() % (dh - rh - 8); rx &= ~3; ry &= ~3; rw &= ~3; rh &= ~3; rects.push_back({rx, ry, rw, rh}); }
    imconfig(IM_CONFIG_SCHEDULER_CORE, CORES[core]);
    auto one = [&]() -> IM_STATUS {
        IM_STATUS st = IM_STATUS_SUCCESS;
        if (isplace) { im_rect sr = {0, 0, 640, 360}, dr = {0, 12, 640, 360}, pr = {0, 0, 0, 0}; return improcess(src, dst, pat, sr, dr, pr, IM_SYNC); }
        if (!strcmp(op, "rect")) { for (int i = 0; i < N; i++) { st = imrectangle(dst, rects[i], 0x0000ff, 2); if (st != IM_STATUS_SUCCESS) return st; } return st; }
        if (!strcmp(op, "rectarr")) return imrectangleArray(dst, rects.data(), N, 0x0000ff, 2);
        if (!strcmp(op, "rectjob")) { for (int b = 0; b < N; b += 12) { int n = N - b < 12 ? N - b : 12; im_job_handle_t j = imbeginJob(); if (!j) return IM_STATUS_FAILED; st = imrectangleTaskArray(j, dst, rects.data() + b, n, 0x0000ff, 2); if (st != IM_STATUS_SUCCESS) { imcancelJob(j); return st; } st = imendJob(j); if (st != IM_STATUS_SUCCESS) return st; } return st; }
        if (!strcmp(op, "labeljob")) { im_job_handle_t j = imbeginJob(); if (!j) return IM_STATUS_FAILED; for (int i = 0; i < N; i++) { im_rect sr = {0, 0, 96, 20}, dr = {rects[i].x, rects[i].y, 96, 20}, pr = {0, 0, 0, 0}; if (dr.x + 96 > dw) dr.x = dw - 96; if (dr.y + 20 > dh) dr.y = dh - 20; dr.x &= ~3; dr.y &= ~3; st = improcessTask(j, lab, dst, pat, sr, dr, pr, NULL, IM_ALPHA_BLEND_SRC_OVER); if (st != IM_STATUS_SUCCESS) { imcancelJob(j); return st; } } return imendJob(j); }
        if (!strcmp(op, "label")) { for (int i = 0; i < N; i++) { im_rect sr = {0, 0, 96, 20}, dr = {rects[i].x, rects[i].y, 96, 20}, pr = {0, 0, 0, 0}; if (dr.x + 96 > dw) dr.x = dw - 96; if (dr.y + 20 > dh) dr.y = dh - 20; dr.x &= ~3; dr.y &= ~3; st = improcess(lab, dst, pat, sr, dr, pr, IM_ALPHA_BLEND_SRC_OVER | IM_SYNC); if (st != IM_STATUS_SUCCESS) return st; } return st; }
        return IM_STATUS_NOT_SUPPORTED;
    };
    IM_STATUS st = one(); if (st != IM_STATUS_SUCCESS) { printf("%s %s %dx%d N=%d FAIL %s\n", op, CN[core], W, H, N, imStrError(st)); return 0; }
    { struct dma_buf_sync sy; sy.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ; ioctl(dfd, DMA_BUF_IOCTL_SYNC, &sy);
      if (getenv("DUMP") && !isplace) { char fn[128]; snprintf(fn, sizeof fn, "/dev/shm/enc/dump_%s_%d_%d.nv12", op, core, W); FILE *f = fopen(fn, "wb"); fwrite(dva, 1, (size_t)dw * dh * 3 / 2, f); fclose(f); } }
    { size_t n = isplace ? (size_t)dw * dh * 3 : (size_t)dw * dh * 3 / 2; unsigned long sum = 0, nz = 0; for (size_t i = 0; i < n; i++) { sum += ((uint8_t *)dva)[i] * (i % 255 + 1); nz += ((uint8_t *)dva)[i] != 0x40; } printf("chk=%lu changed=%lu ", sum, nz); }
    for (int i = 0; i < 10; i++) one();
    long i0[3]; for (int k = 0; k < 3; k++) i0[k] = irq_count(CI[k]);
    std::vector<double> t; double t0 = now_ms();
    long fails = 0; while (now_ms() - t0 < secs * 1e3) { double a = now_ms(); IM_STATUS q = one(); if (q != IM_STATUS_SUCCESS) { fails++; continue; } t.push_back(now_ms() - a); }
    long i1[3]; for (int k = 0; k < 3; k++) i1[k] = irq_count(CI[k]);
    double total = now_ms() - t0; if (t.empty()) { printf("fails=%ld all failed\n", fails); return 0; } std::sort(t.begin(), t.end());
    char ran[64] = ""; for (int k = 0; k < 3; k++) if (i1[k] - i0[k] > 0) { strcat(ran, CN[k]); strcat(ran, " "); }
    printf("fails=%ld ", fails); printf("%s %s %dx%d N=%d iters=%zu median=%.3f p95=%.3f ms rate=%.1f/s ran:%s\n", op, CN[core], W, H, N, t.size(), t[t.size() / 2], t[t.size() * 95 / 100], t.size() * 1e3 / total, ran);
    return 0;
}
