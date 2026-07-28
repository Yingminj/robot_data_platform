# LeRobot AV1 compression evaluation

该目录包含头部相机 HDF5 单帧数据与 LeRobot AV1 CRF 0 / CRF 20 / CRF 50 视频的完整可复现实验，并横向比较：

- DINOv3 ViT-S/16
- DINOv3 ViT-B/16
- DINOv2 ViT-S/14

最终结论与图表见 [REPORT.md](REPORT.md)。

运行全部步骤：

```bash
./run_all.sh --force
```

目录：

- `config.json`：数据、编码器和 DINOv3 参数
- `scripts/`：编码、逐帧指标、DINOv3 特征评估和报告生成代码
- `videos/`：三档 MP4
- `results/per_frame_model_feature_metrics.csv`：三模型逐帧特征指标
- `results/model_feature_summary.json`：三模型汇总
- `results/model_global_embeddings.npz`：三模型全局特征
- `results/feature_maps/`：三模型完整逐帧 patch-token 特征（float16 NPZ）
- `figures/feature_maps/`：原始帧与三档压缩帧的 PCA-RGB 特征图
- `figures/`：指标图和最差帧对比

`run_all.sh` 使用已有的 `test` Conda 环境读取 HDF5/PyAV，使用 `dino` 环境运行 DINOv3，不会修改已有环境。
