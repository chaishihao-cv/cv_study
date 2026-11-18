# day12 简易残差块实现
# 任务：手写 ResidualBlock；对照 torchvision ResNet18 结构
# 思考题：残差连接为什么能让深层网络更好训练？

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # TODO：conv-bn-relu-conv-bn + shortcut

    def forward(self, x):
        # TODO
        pass
