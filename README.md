# camera_stream_remover

实现固定机位下的实时目标擦除：暂停画面框选目标后，恢复流时自动匹配并锁定 Track ID，使用累积背景覆盖目标区域，并做边缘修复。

## 运行环境

- Python 3.13+
- `opencv-python`
- `numpy`
- （可选）`ultralytics`：用于 YOLOv8/v11 Seg + Track

## 快速启动

```bash
python stream_remover.py --camera 0
```

如果有 ONNX/TensorRT 推理封装，可替换 `YoloSegTracker.infer()` 实现，保持输出为 `(track_id, bbox, mask)` 即可接入现有流程。

## 键盘与鼠标交互

- **鼠标左键单击**：选中 / 取消选中目标（点击已检测目标的 Mask 区域）
- `Space`: 暂停/恢复
- `C`: 清空已锁定擦除目标
- `L`: 开关背景学习
- `Q` / `Esc`: 退出

## 视觉反馈

- **淡绿色半透明 Mask**：所有检测到的目标（提示可点选）
- **亮绿色半透明 Mask**：已选中、正在被擦除的目标
