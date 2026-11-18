# day29 ONNX Runtime 独立推理（脱离 ultralytics）
# 任务：预处理 -> onnxruntime 推理 -> 解析输出 -> NMS -> 画框
# 思考题：ONNX 推理与原生 PyTorch 推理有什么差别？

import numpy as np
import onnxruntime as ort

# TODO：在这里写代码
