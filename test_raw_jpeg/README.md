# test_raw_jpeg — 原始图像 vs JPEG 压缩图像的画质影响测试

回答一个问题：录制端从「未压缩 `/quad_tile`」换成「JPEG q80 `/quad_tile/compressed`」之后，
训练数据的画质掉了多少，以及在这之后 LeRobot 的 CRF 档位还剩多少可选空间。

结果见 [`REPORT.md`](REPORT.md)。

## 为什么不直接对比 express_raw 和 express_mcap

两者是不同时间的两次录制，画面内容不同，逐像素指标没有意义。本测试改为构造对照组：
把 `express_raw` 分别按 q100 与 q80 重新编码成 JPEG 的 mcap（时间戳与关节数据逐字节
复制，只替换图像话题），让三条链路面对同一批帧。q80 是生产配置，q100 用来把「压了」
和「压狠了」分开。`express_mcap` 仅用于核对合成 JPEG 的码率是否与真实录制吻合。

## 流水线

| 阶段 | 脚本 | 产出 |
|---|---|---|
| 1 | `scripts/01_extract_and_jpeg.py` | 原始 tile 基准（memmap）+ JPEG q100/q80 指标 |
| 1b | `scripts/01b_reference_mcap_sizes.py` | `express_mcap` 真实 JPEG 码率 |
| 2 | `scripts/02_build_jpeg_bag.py --quality {100,80}` | `bags/express_jpeg*/` 合成 mcap |
| 3 | `scripts/03_convert_lerobot.sh` | 9 个 LeRobot v3 数据集（3 源 × CRF 0/20/30） |
| 4 | `scripts/04_eval_videos.py` | 视频体积、码率、逐帧画质 |
| 6 | `scripts/06_decode_videos_to_memmap.py` | 把 27 个视频解码成 memmap 供阶段 7 复用 |
| 7 | `scripts/07_dino_feature_eval.py` | DINOv3/DINOv2 特征漂移（需要 GPU） |
| 5 | `scripts/05_make_report.py` | 图与 `REPORT.md`（最后运行） |

```bash
./run_all.sh          # 各阶段在产物已存在时自动跳过
```

## 环境

- `envs/lerobot`：`rosbags`、`lerobot 0.6.0`、`av`、`opencv` — 阶段 1–4、6
- `envs/dino`：`torch`、`matplotlib` — 阶段 7、5

像素指标（PSNR / SSIM / MAE）取自 `test_lerobot/scripts/encode_and_pixel_eval.py`，
特征指标与预处理取自 `test_lerobot/scripts/dinov3_feature_eval.py`，模型与权重也是同三个。
两份报告的数值因此可以并排比较——差异只来自实验设计（本测试 11 档位 × 3 相机 × 1103 帧，
test_lerobot 3 档位 × 1 相机 × 220 帧，且编码器为 AV1 而非 h264）。

## 中间产物体积

`intermediate/*.npy` 约 36 GB：9 个来自阶段 1（原始 / JPEG-100 / JPEG-80 × 3 相机），
27 个来自阶段 6（9 个数据集 × 3 相机的解码帧）。删除后已生成的报告不受影响，
但重跑阶段 4 或 7 需要重新生成。
