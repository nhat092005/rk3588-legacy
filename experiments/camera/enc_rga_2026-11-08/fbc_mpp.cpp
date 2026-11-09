// Does the MPP H.265 encoder (VEPU580) accept an AFBC_V1 frame straight from rkisp_fbcpath? One IDR frame, dma-buf imported, output written to /tmp/fbc_enc.h265
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
#include <rockchip/rk_mpi.h>
#include <rockchip/mpp_buffer.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_packet.h>
#include <rockchip/rk_venc_cfg.h>
static int xioctl(int fd, unsigned long r, void *a) { int e; do e = ioctl(fd, r, a); while (e < 0 && errno == EINTR); return e; }
#define CHK(x) do { MPP_RET r_ = (x); if (r_ != MPP_OK) { printf("FAIL %s -> %d\n", #x, (int)r_); return 1; } } while (0)
int main(int argc, char **argv) {
    const int W = 4224, H = 3136; int codec_h265 = !(argc > 1 && !strcmp(argv[1], "h264"));
    int ffd = open("/dev/video13", O_RDWR);
    v4l2_format f = {}; f.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; f.fmt.pix_mp.width = W; f.fmt.pix_mp.height = H; f.fmt.pix_mp.pixelformat = v4l2_fourcc('F','B','C','0'); f.fmt.pix_mp.num_planes = 1;
    if (xioctl(ffd, VIDIOC_S_FMT, &f) < 0) { perror("S_FMT"); return 1; }
    unsigned size = f.fmt.pix_mp.plane_fmt[0].sizeimage;
    v4l2_requestbuffers rb = {}; rb.count = 4; rb.type = f.type; rb.memory = V4L2_MEMORY_MMAP; xioctl(ffd, VIDIOC_REQBUFS, &rb);
    int fds[8];
    for (unsigned i = 0; i < rb.count; i++) { v4l2_exportbuffer e = {}; e.type = f.type; e.index = i; e.flags = O_RDWR; xioctl(ffd, VIDIOC_EXPBUF, &e); fds[i] = e.fd; v4l2_buffer b = {}; v4l2_plane p[VIDEO_MAX_PLANES] = {}; b.type = f.type; b.memory = V4L2_MEMORY_MMAP; b.index = i; b.m.planes = p; b.length = 1; xioctl(ffd, VIDIOC_QBUF, &b); }
    int t = f.type; xioctl(ffd, VIDIOC_STREAMON, &t); int got = -1;
    for (int k = 0; k < 6; k++) { v4l2_buffer b = {}; v4l2_plane p[VIDEO_MAX_PLANES] = {}; b.type = f.type; b.memory = V4L2_MEMORY_MMAP; b.m.planes = p; b.length = 1; pollfd pf = {ffd, POLLIN, 0}; poll(&pf, 1, 3000); if (xioctl(ffd, VIDIOC_DQBUF, &b) < 0) break; got = b.index; if (k < 5) xioctl(ffd, VIDIOC_QBUF, &b); }
    if (got < 0) { printf("no frame\n"); return 1; }
    // keep stream on: buffer 'got' stays dequeued, others are queued
    MppCtx ctx; MppApi *mpi; CHK(mpp_create(&ctx, &mpi));
    CHK(mpp_init(ctx, MPP_CTX_ENC, codec_h265 ? MPP_VIDEO_CodingHEVC : MPP_VIDEO_CodingAVC));
    MppEncCfg cfg; CHK(mpp_enc_cfg_init(&cfg)); CHK(mpi->control(ctx, MPP_ENC_GET_CFG, cfg));
    int raw = argc > 2 && !strcmp(argv[2], "raw"); int rawfd = -1; unsigned rawsize = (unsigned)((size_t)W * H * 3 / 2);
    if (raw) {
        int vfd = open("/dev/video11", O_RDWR); v4l2_format g = {}; g.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; g.fmt.pix_mp.width = W; g.fmt.pix_mp.height = H; g.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12; g.fmt.pix_mp.num_planes = 1; xioctl(vfd, VIDIOC_S_FMT, &g);
        v4l2_requestbuffers r2 = {}; r2.count = 1; r2.type = g.type; r2.memory = V4L2_MEMORY_MMAP; xioctl(vfd, VIDIOC_REQBUFS, &r2);
        v4l2_exportbuffer e2 = {}; e2.type = g.type; e2.index = 0; e2.flags = O_RDWR; xioctl(vfd, VIDIOC_EXPBUF, &e2); rawfd = e2.fd;
        void *va = mmap(NULL, rawsize, PROT_READ | PROT_WRITE, MAP_SHARED, rawfd, 0); FILE *in = fopen("/tmp/fbc_decoded.nv12", "rb"); size_t n = fread(va, 1, rawsize, in); fclose(in); printf("raw control: loaded %zu bytes\n", n);
    }
    MppFrameFormat fmt = raw ? MPP_FMT_YUV420SP : (MppFrameFormat)(MPP_FMT_YUV420SP | MPP_FRAME_FBC_AFBC_V1);
    mpp_enc_cfg_set_s32(cfg, "prep:width", W); mpp_enc_cfg_set_s32(cfg, "prep:height", H);
    mpp_enc_cfg_set_s32(cfg, "prep:hor_stride", W); mpp_enc_cfg_set_s32(cfg, "prep:ver_stride", H);
    mpp_enc_cfg_set_s32(cfg, "prep:format", fmt);
    mpp_enc_cfg_set_s32(cfg, "rc:mode", MPP_ENC_RC_MODE_CBR); mpp_enc_cfg_set_s32(cfg, "rc:bps_target", 16000000);
    mpp_enc_cfg_set_s32(cfg, "rc:bps_max", 17000000); mpp_enc_cfg_set_s32(cfg, "rc:bps_min", 15000000);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_in_flex", 0); mpp_enc_cfg_set_s32(cfg, "rc:fps_in_num", 30); mpp_enc_cfg_set_s32(cfg, "rc:fps_in_denom", 1);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_out_flex", 0); mpp_enc_cfg_set_s32(cfg, "rc:fps_out_num", 30); mpp_enc_cfg_set_s32(cfg, "rc:fps_out_denom", 1); mpp_enc_cfg_set_s32(cfg, "rc:gop", 30);
    MPP_RET sr = mpi->control(ctx, MPP_ENC_SET_CFG, cfg); printf("SET_CFG (%s input, %ux%u): ret %d\n", raw ? "raw NV12" : "AFBC_V1", W, H, (int)sr); if (sr != MPP_OK) return 1;
    MppBufferInfo info = {}; info.type = MPP_BUFFER_TYPE_EXT_DMA; info.fd = raw ? rawfd : fds[got]; info.size = raw ? rawsize : size;
    MppBuffer buf = NULL; CHK(mpp_buffer_import(&buf, &info));
    MppFrame frame; CHK(mpp_frame_init(&frame));
    mpp_frame_set_width(frame, W); mpp_frame_set_height(frame, H); mpp_frame_set_hor_stride(frame, W); mpp_frame_set_ver_stride(frame, H); mpp_frame_set_fmt(frame, fmt); mpp_frame_set_buffer(frame, buf); mpp_frame_set_eos(frame, 0);
    MPP_RET pr = mpi->encode_put_frame(ctx, frame); printf("encode_put_frame: ret %d\n", (int)pr); if (pr != MPP_OK) return 1;
    FILE *o = fopen(raw ? "/tmp/raw_enc.h265" : (codec_h265 ? "/tmp/fbc_enc.h265" : "/tmp/fbc_enc.h264"), "wb"); int npk = 0; size_t tot = 0;
    for (int fr = 0; fr < 3; fr++) {
        if (fr > 0) { MppFrame f2; mpp_frame_init(&f2); mpp_frame_set_width(f2, W); mpp_frame_set_height(f2, H); mpp_frame_set_hor_stride(f2, W); mpp_frame_set_ver_stride(f2, H); mpp_frame_set_fmt(f2, fmt); mpp_frame_set_buffer(f2, buf); if (fr == 2) mpp_frame_set_eos(f2, 1); MPP_RET p2 = mpi->encode_put_frame(ctx, f2); printf("put_frame %d: ret %d\n", fr, (int)p2); }
        for (int tries = 0; tries < 400; tries++) {
            MppPacket pkt = NULL; MPP_RET gr = mpi->encode_get_packet(ctx, &pkt); if (gr != MPP_OK) { printf("get_packet ret %d\n", (int)gr); break; }
            if (pkt) { size_t len = mpp_packet_get_length(pkt); printf("frame %d packet %d: %zu bytes\n", fr, npk++, len); tot += len; fwrite(mpp_packet_get_pos(pkt), 1, len, o); int eos = mpp_packet_get_eos(pkt); mpp_packet_deinit(&pkt); if (len > 1000 || eos) break; } else usleep(5000);
        }
    }
    fclose(o); printf("total %zu bytes in %d packets\n", tot, npk);
    return 0;
}
