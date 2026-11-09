// Can RGA2 map buffers from V4L2 mainpath (EXPBUF) and from MPP groups? Fill 64x64 + one imrectangle on RGA2, report status.
#include <initializer_list>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include "im2d.h"
#include "rga.h"
#include <rockchip/rk_mpi.h>
#include <rockchip/mpp_buffer.h>
static IM_STATUS try_rga(int fd, int w, int h, uint64_t core) {
    rga_buffer_t b = wrapbuffer_fd(fd, w, h, RK_FORMAT_YCbCr_420_SP);
    imconfig(IM_CONFIG_SCHEDULER_CORE, core);
    im_rect r = {0, 0, 64, 64};
    IM_STATUS s = imfill(b, r, 0x0000ff); if (s != IM_STATUS_SUCCESS) return s;
    im_rect q = {128, 128, 200, 100};
    return imrectangle(b, q, 0x0000ff, 2);
}
static void report(const char *tag, int idx, int fd, int w, int h) {
    IM_STATUS a = try_rga(fd, w, h, IM_SCHEDULER_RGA2_CORE0), c = try_rga(fd, w, h, IM_SCHEDULER_RGA3_CORE0);
    printf("%-26s #%-2d RGA2:%s  RGA3_0(rect fails by design):%s\n", tag, idx, a == IM_STATUS_SUCCESS ? "OK" : "FAIL", c == IM_STATUS_SUCCESS ? "OK" : "FAIL");
}
int main(int argc, char **argv) {
    int W = 3840, H = 2160;
    int mode = argc > 1 ? atoi(argv[1]) : -1;
    if (mode < 0) {
    // V4L2 mainpath
    for (int count : {4, 8, 16}) {
        int vfd = open("/dev/video11", O_RDWR); if (vfd < 0) { perror("video11"); return 1; }
        v4l2_format f = {}; f.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        f.fmt.pix_mp.width = W; f.fmt.pix_mp.height = H; f.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12; f.fmt.pix_mp.num_planes = 1;
        if (ioctl(vfd, VIDIOC_S_FMT, &f) < 0) { perror("S_FMT"); return 1; }
        v4l2_requestbuffers rb = {}; rb.count = count; rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; rb.memory = V4L2_MEMORY_MMAP;
        if (ioctl(vfd, VIDIOC_REQBUFS, &rb) < 0) { perror("REQBUFS"); return 1; }
        for (unsigned i = 0; i < rb.count; i++) {
            v4l2_exportbuffer e = {}; e.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; e.index = i; e.plane = 0; e.flags = O_RDWR;
            if (ioctl(vfd, VIDIOC_EXPBUF, &e) < 0) { perror("EXPBUF"); continue; }
            char tag[64]; snprintf(tag, sizeof tag, "V4L2 mmap (count=%d)", count);
            report(tag, i, e.fd, f.fmt.pix_mp.width, f.fmt.pix_mp.height);
        }
        close(vfd);
    }
    }
    // MPP groups
    struct { const char *n; MppBufferType t; } types[] = {
        {"MPP DRM", (MppBufferType)MPP_BUFFER_TYPE_DRM},
        {"MPP DRM|CONTIG", (MppBufferType)(MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CONTIG)},
        {"MPP DMA_HEAP", (MppBufferType)MPP_BUFFER_TYPE_DMA_HEAP},
    };
    if (mode >= 0) { auto &t = types[mode]; int n = argc > 2 ? atoi(argv[2]) : 8;
        MppBufferGroup g = NULL;
        if (mpp_buffer_group_get_internal(&g, t.t) != MPP_OK) { printf("%s: group get failed\n", t.n); return 1; }
        for (int i = 0; i < n; i++) {
            MppBuffer b = NULL;
            if (mpp_buffer_get(g, &b, (size_t)W * H * 3 / 2) != MPP_OK) { printf("%s #%d: alloc failed\n", t.n, i); break; }
            int fd = mpp_buffer_get_fd(b); if (fd < 0) { printf("%s #%d: fd<0\n", t.n, i); break; }
            report(t.n, i, fd, W, H); fflush(stdout);
        }
    }
    return 0;
}
