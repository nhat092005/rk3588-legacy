// Can RGA read or draw on an FBC0 frame from rkisp_fbcpath? Real frame from /dev/video13, dst raster buffer from /dev/video11 (under 4G).
#include <initializer_list>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include "im2d.h"
#include "rga.h"
static int xioctl(int fd, unsigned long r, void *a) { int e; do e = ioctl(fd, r, a); while (e < 0 && errno == EINTR); return e; }
static const uint64_t CORES[3] = {IM_SCHEDULER_RGA3_CORE0, IM_SCHEDULER_RGA3_CORE1, IM_SCHEDULER_RGA2_CORE0};
static const char *CN[3] = {"RGA3_0", "RGA3_1", "RGA2"};
int main() {
    const int W = 4224, H = 3136;
    int ffd = open("/dev/video13", O_RDWR); if (ffd < 0) { perror("video13"); return 1; }
    v4l2_format f = {}; f.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; f.fmt.pix_mp.width = W; f.fmt.pix_mp.height = H; f.fmt.pix_mp.pixelformat = v4l2_fourcc('F','B','C','0'); f.fmt.pix_mp.num_planes = 1;
    if (xioctl(ffd, VIDIOC_S_FMT, &f) < 0) { perror("S_FMT fbc"); return 1; }
    printf("fbc fmt %ux%u sizeimage=%u\n", f.fmt.pix_mp.width, f.fmt.pix_mp.height, f.fmt.pix_mp.plane_fmt[0].sizeimage);
    v4l2_requestbuffers rb = {}; rb.count = 4; rb.type = f.type; rb.memory = V4L2_MEMORY_MMAP; if (xioctl(ffd, VIDIOC_REQBUFS, &rb) < 0) { perror("REQBUFS"); return 1; }
    int fds[8]; v4l2_plane pl[4][VIDEO_MAX_PLANES];
    for (unsigned i = 0; i < rb.count; i++) {
        v4l2_exportbuffer e = {}; e.type = f.type; e.index = i; e.flags = O_RDWR; if (xioctl(ffd, VIDIOC_EXPBUF, &e) < 0) { perror("EXPBUF"); return 1; } fds[i] = e.fd;
        v4l2_buffer b = {}; memset(pl[i], 0, sizeof pl[i]); b.type = f.type; b.memory = V4L2_MEMORY_MMAP; b.index = i; b.m.planes = pl[i]; b.length = 1; xioctl(ffd, VIDIOC_QBUF, &b);
    }
    int t = f.type; xioctl(ffd, VIDIOC_STREAMON, &t);
    int got = -1; unsigned used = 0;
    for (int k = 0; k < 6; k++) { v4l2_buffer b = {}; v4l2_plane p[VIDEO_MAX_PLANES] = {}; b.type = f.type; b.memory = V4L2_MEMORY_MMAP; b.m.planes = p; b.length = 1; pollfd pf = {ffd, POLLIN, 0}; poll(&pf, 1, 3000); if (xioctl(ffd, VIDIOC_DQBUF, &b) < 0) { perror("DQBUF"); break; } got = b.index; used = p[0].bytesused; printf("frame %d idx=%d bytesused=%u\n", k, b.index, used); if (k < 5) xioctl(ffd, VIDIOC_QBUF, &b); }
    if (got < 0) return 1;
    // dst raster buffer from mainpath (under 4G), NV12 4224x3136
    int vfd = open("/dev/video11", O_RDWR);
    v4l2_format g = {}; g.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; g.fmt.pix_mp.width = W; g.fmt.pix_mp.height = H; g.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12; g.fmt.pix_mp.num_planes = 1;
    if (xioctl(vfd, VIDIOC_S_FMT, &g) < 0) { perror("S_FMT main"); return 1; }
    v4l2_requestbuffers r2 = {}; r2.count = 1; r2.type = g.type; r2.memory = V4L2_MEMORY_MMAP; xioctl(vfd, VIDIOC_REQBUFS, &r2);
    v4l2_exportbuffer e2 = {}; e2.type = g.type; e2.index = 0; e2.flags = O_RDWR; if (xioctl(vfd, VIDIOC_EXPBUF, &e2) < 0) { perror("EXPBUF main"); return 1; }
    size_t rsz = (size_t)W * H * 3 / 2; void *rva = mmap(NULL, rsz, PROT_READ | PROT_WRITE, MAP_SHARED, e2.fd, 0);
    memset(rva, 0, rsz);
    rga_buffer_t src = wrapbuffer_fd(fds[got], W, H, RK_FORMAT_YCbCr_420_SP); src.rd_mode = IM_FBC_MODE;
    rga_buffer_t dst = wrapbuffer_fd(e2.fd, W, H, RK_FORMAT_YCbCr_420_SP);
    for (int c = 0; c < 3; c++) {
        imconfig(IM_CONFIG_SCHEDULER_CORE, CORES[c]);
        memset(rva, 0, rsz);
        IM_STATUS s = imcopy(src, dst);
        size_t nz = 0; for (size_t i = 0; i < rsz; i += 61) nz += ((unsigned char *)rva)[i] != 0;
        printf("%-7s read FBC -> raster NV12 (imcopy): %s (nonzero samples %zu)\n", CN[c], s == IM_STATUS_SUCCESS ? "OK" : imStrError(s), nz);
        if (c == 0 && s == IM_STATUS_SUCCESS) { FILE *o = fopen("/tmp/fbc_decoded.nv12", "wb"); fwrite(rva, 1, rsz, o); fclose(o); }
    }
    rga_buffer_t fb = wrapbuffer_fd(fds[got], W, H, RK_FORMAT_YCbCr_420_SP); fb.rd_mode = IM_FBC_MODE;
    for (int c = 0; c < 3; c++) {
        imconfig(IM_CONFIG_SCHEDULER_CORE, CORES[c]);
        im_rect r = {128, 128, 400, 300};
        IM_STATUS a = imrectangle(fb, r, 0x0000ff, 4), b2 = imfill(fb, r, 0x0000ff);
        printf("%-7s draw on FBC buffer: imrectangle %s, imfill %s\n", CN[c], a == IM_STATUS_SUCCESS ? "OK" : imStrError(a), b2 == IM_STATUS_SUCCESS ? "OK" : imStrError(b2));
    }
    return 0;
}
