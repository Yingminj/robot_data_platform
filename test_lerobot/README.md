# LeRobot AV1 compression evaluation

该目录包含头部相机 HDF5 单帧数据与 LeRobot AV1 CRF 0 / CRF 20 / CRF 50 视频的完整可复现实验。

最终结论与图表见 [REPORT.md](REPORT.md)。

运行全部步骤：

```bash
./run_all.sh --force
```

目录：

- `config.json`：数据、编码器和 DINOv3 参数
- `scripts/`：编码、逐帧指标、DINOv3 特征评估和报告生成代码
- `videos/`：三档 MP4
- `results/`：逐帧 CSV、汇总 JSON 和全局特征 NPZ
- `figures/`：指标图和最差帧对比

`run_all.sh` 使用已有的 `test` Conda 环境读取 HDF5/PyAV，使用 `dino` 环境运行 DINOv3，不会修改已有环境。
