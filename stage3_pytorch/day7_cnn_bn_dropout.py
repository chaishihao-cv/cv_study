# day7 CNN 中加入 BN、Dropout 对比实验
# 任务 1 手算 BN   |  任务 2 数 BN 家底  |  任务 3 train/eval 差异
# 任务 4 Dropout 三数 | 任务 5 收敛对比 | 任务 6 手写 BN
# 思考题：BN 在 train 与 eval 模式下行为差别在哪？
#
# 跑法：& "D:\Anaconda\anaconda3\python.exe" day7_cnn_bn_dropout.py
# 先跑 python -c "import torch;print(torch.cuda.is_available())" 确认环境

import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

torch.manual_seed(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {dev}")


# ============================================================
# 任务 1：手算 BN
# x = [[1.0, 2.0], [2.0, 2.0], [3.0, 2.0], [4.0, 10.0]]   shape (4,2)
# ============================================================
print("\n--- 任务 1: 手算 BN ---")

x1 = torch.tensor([[1.0, 2.0],
                   [2.0, 2.0],
                   [3.0, 2.0],
                   [4.0, 10.0]])

# 空1：通道 0 的 μ、σ²、σ（自己先写在下面，再用代码核对）
mu0_manual, var0_manual, sd0_manual = None, None, None
# 空2：通道 0 标准化后的 4 个值
xhat0_manual = None
# 空3：通道 1 的 μ、σ²
mu1_manual, var1_manual = None, None
# 空4：通道 1 标准化后的 4 个值
xhat1_manual = None

# TODO: 代码核对（提示：nn.BatchNorm1d(2)，注意 bn 默认就是 train 模式）
# bn = nn.BatchNorm1d(2)
# y = bn(x1) ...

print("任务 1 还没写")   # TODO 删掉这行


# ============================================================
# 任务 2：数 BN 家底
# ============================================================
print("\n--- 任务 2: BN 家底 ---")

# TODO: 对 BatchNorm2d(64) 和 BatchNorm1d(128) 各打印 6 个数：
#   weight/bias/running_mean/running_var/num_batches_tracked 的元素数
#   state_dict 总元素数、requires_grad=True 的个数
#   以及 BN 参数占 Conv2d(64,64,3) 的百分比（Conv 参数 = 36928）

print("任务 2 还没写")


# ============================================================
# 任务 3：train / eval 两种模式
# ============================================================
print("\n--- 任务 3: train/eval 差异 ---")

# 3.1 同输入、两模式
# TODO: 写
# bn = nn.BatchNorm1d(16)
# big = torch.randn(1000, 16) * 2 + 5
# bn(big)                       # 攒 running stats
# x1 = torch.randn(4, 16) * 2 + 5
# bn.train(); o_tr = bn(x1)
# bn.eval();  o_ev = bn(x1)
# 打印两种模式的均值/标准差，以及最大差（空5）

# 3.2 batch=1 的报错（这是要你复现并把原文抄进注释的，不要修）
# TODO: bn1 = nn.BatchNorm1d(16); bn1.train(); bn1(torch.randn(1, 16))
# 用 try/except 把 type(e).__name__ 和 str(e) 打出来
# 空7 原文：__________________________________________

# 3.3 batch=2 的极端情况
# TODO: bn2 = nn.BatchNorm1d(16); bn2.train()
#      print(bn2(torch.randn(2, 16)).round(decimals=2))
# 空8：你看到的规律 ________________________________

print("任务 3 还没写")


# ============================================================
# 任务 4：Dropout
# ============================================================
print("\n--- 任务 4: Dropout ---")

# 4.1 p=0.5，输入 ones(1000)，打印保留比例 / 置0比例 / 非零值
# 4.2 p=0.3，先手算非零值（空9）再跑核对
# 4.3 d.eval() 后打印输出（空10）
# 4.4 train 下连续两次 forward，torch.equal + 不同位置个数（空11）
# TODO: 写

print("任务 4 还没写")


# ============================================================
# 任务 5：BN 到底允许多大的学习率（今天的重头戏）
# ============================================================
print("\n--- 任务 5: 有/无 BN 的收敛对比 ---")

tf = transforms.ToTensor()
train_set = datasets.MNIST('D:/cv_study/data', train=True, download=False, transform=tf)
test_set = datasets.MNIST('D:/cv_study/data', train=False, download=False, transform=tf)
train_set = Subset(train_set, range(3000))     # 必须用子集，否则跑几分钟
test_set = Subset(test_set, range(1000))
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)


def make(use_bn):
    layers = [nn.Flatten(), nn.Linear(784, 256)]
    if use_bn:
        layers.append(nn.BatchNorm1d(256))
    layers += [nn.ReLU(), nn.Linear(256, 128)]
    if use_bn:
        layers.append(nn.BatchNorm1d(128))
    layers += [nn.ReLU(), nn.Linear(128, 10)]
    return nn.Sequential(*layers)


def run(use_bn, lr, epochs=3):
    """返回 [(epoch, train_acc, test_acc), ...]"""
    torch.manual_seed(0)                      # 每组必须重置种子
    net = make(use_bn).to(dev)
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        # TODO: 训练循环
        #   net.train()
        #   for xb, yb in train_loader: zero_grad -> forward -> loss -> backward -> step
        #   统计训练 acc
        # TODO: 评估
        #   net.eval()  + torch.no_grad()
        # 把 (ep+1, train_acc, test_acc) 记进 hist
        pass
    print(f"  use_bn={use_bn} lr={lr} 用时 {time.time()-t0:.1f}s")
    return hist


# TODO: 6 组循环
# for use_bn in (False, True):
#     for lr in (0.05, 0.5, 2.0):
#         h = run(use_bn, lr)
#         打印每轮的 train/test acc，填任务卡的表
print("任务 5 还没写")


# ============================================================
# 任务 6：手写 BN（不许用 nn.BatchNorm1d）
# ============================================================
print("\n--- 任务 6: 手写 BN ---")


class MyBN:
    def __init__(self, C, momentum=0.1, eps=1e-5):
        self.gamma = torch.ones(C)
        self.beta = torch.zeros(C)
        self.running_mean = torch.zeros(C)
        self.running_var = torch.ones(C)
        self.momentum = momentum
        self.eps = eps

    def forward(self, x, training=True):
        # x: (N, C)
        # TODO: training=True 时用当前 batch 的均值/方差（有偏，除以 N），并更新 running
        #       training=False 时用 self.running_mean / self.running_var
        #       最后 y = gamma * x_hat + beta
        raise NotImplementedError("任务 6 还没写")


# 验证（写完打开）
# torch.manual_seed(0)
# x = torch.randn(32, 8) * 3 + 10
# ref = nn.BatchNorm1d(8); ref.train()
# y_ref = ref(x)
# mine = MyBN(8)
# y_mine = mine.forward(x, training=True)
# print("train 最大差:", (y_ref - y_mine).abs().max().item())
# mine.running_mean = ref.running_mean.clone()
# mine.running_var = ref.running_var.clone()
# ref.eval()
# print("eval 最大差:", (ref(x) - mine.forward(x, training=False)).abs().max().item())


# ============================================================
# 思考题（写进注释，各 3~5 行）
# ============================================================
# 1. BN 只有标准化、去掉 γ β 会失去什么？举一个具体的层说明。
# 2. BatchNorm1d 在 train 模式 batch=1 报错，推理时 batch=1 却正常，为什么？
# 3. Dropout 训练时放大 vs 推理时乘 (1-p)，两种做法等价吗？各自代价？
# 4. Conv→BN→ReLU→Dropout 换成 Conv→ReLU→Dropout→BN 会怎样？小 batch 时呢？
