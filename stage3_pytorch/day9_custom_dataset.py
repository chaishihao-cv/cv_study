# day9 自定义 Dataset + DataLoader
# 任务：手写 Dataset 读取本地图片文件夹+标签；DataLoader 分批加载
# 思考题：shuffle / batch_size / num_workers 各自作用？

import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        # TODO：返回 (image, label)
        pass
