# day4 MLP 手写数字识别（MNIST 完整训练脚本）
# 任务：搭建 MLP 训练 MNIST；打印每层输出 shape；画 loss 曲线
# 思考题：反向传播更新的是什么？为什么推理时不需要计算梯度？

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# TODO：在这里写代码
