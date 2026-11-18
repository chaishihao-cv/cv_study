# day5 卷积尺寸计算 + 单层 Conv2d 测试
# 任务：手算 Conv2d 输出尺寸，并与实际 shape 核对
# 思考题：padding=same 与 valid 区别？stride 变大特征图怎么变？
# 公式：out = (in + 2*padding - kernel) // stride + 1

import torch
import torch.nn as nn

# TODO：在这里写代码
