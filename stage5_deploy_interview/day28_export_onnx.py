# day28 YOLOv8 导出 ONNX
# 任务：导出 ONNX 模型，用 onnx 工具查看输入输出节点
# 思考题：导出 ONNX 的意义是什么？

from ultralytics import YOLO

# TODO: model = YOLO("best.pt"); model.export(format="onnx", opset=12, imgsz=640)
