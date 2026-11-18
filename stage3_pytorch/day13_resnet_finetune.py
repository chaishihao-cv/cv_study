# day13 ResNet18 迁移学习与冻结主干实验
# 任务：加载预训练 ResNet18 做小数据集分类；对比冻结主干与全量微调
# 思考题：什么时候冻结主干？什么时候全部参数微调？

import torch
import torchvision.models as models

# TODO：在这里写代码
