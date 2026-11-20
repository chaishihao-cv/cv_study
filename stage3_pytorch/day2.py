# 梯度消失实验（day2 任务4）
#
# 核心观察：梯度从输出层往回传，每穿一层就乘一次"该层导数"
#   sigmoid 每层最多 0.25 → 10 层只剩千万分之一
#   tanh    实际约 0.8    → 10 层还剩 0.107
#   ReLU    正区间恒为 1  → 完全不衰减
#
# 运行：D:\Anaconda\anaconda3\python.exe day2.py

import numpy as np
import matplotlib.pyplot as plt

# ---- 所有实验共用这一份数据（改这里，五个实验一起变）----
cases = {
    'sigmoid (理论上限 0.25)': 0.25,
    'sigmoid (远离0处的0.05)': 0.05,
    'tanh (实际0.8)': 0.8,
    'relu (正区间)': 1.0,
    '假设某层 1.5 (会爆炸)': 1.5,
}

LR = 0.1

# ============================================================
print('=' * 56)
print('实验1：不同激活函数，10 层连乘后每层还能收到多少梯度')
print('=' * 56)

for name, per_layer in cases.items():
    grad = 1.0
    for layer in range(10, 0, -1):
        grad *= per_layer
        print(f'  {name:24s} 第 {layer:2d} 层：{grad:.3e}')

# ============================================================
print()
print('=' * 56)
print('实验2：层数越多越致命 —— sigmoid 与 tanh 并排对比')
print('=' * 56)
print(f"  {'层数':>4}   {'sigmoid 0.25':>14}   {'tanh 0.8':>12}")

for layers in (5, 10, 20, 30, 40, 50):
    g_sig = 0.25 ** layers
    g_tanh = 0.8 ** layers
    print(f"  {layers:>4}   {g_sig:>14.3e}   {g_tanh:>12.3e}")

# ============================================================
print()
print('=' * 56)
print('实验3：逐层并排对比 —— 看两者差距怎么一层层拉开')
print('=' * 56)
print(f"  {'层':>3}   {'sigmoid 0.25':>14}   {'tanh 0.8':>12}   {'tanh/sigmoid':>13}")

g_sig, g_tanh = 1.0, 1.0
for layer in range(10, 0, -1):
    g_sig *= 0.25
    g_tanh *= 0.8
    print(f"  {layer:>3}   {g_sig:>14.3e}   {g_tanh:>12.3e}   {g_tanh / g_sig:>13.1e}")

# ============================================================
print()
print('=' * 56)
print("实验4：换算成'权重每次能移动多少'（lr=0.1）")
print('=' * 56)

for label, per_layer in cases.items():
    g = per_layer ** 10
    print(f'  {label:24s} 梯度 {g:.3e} → 每步移动 {LR * g:.3e}')

# ============================================================
print()
print('=' * 56)
print("实验5：最底层要走多少步才走得完 0.5 的权重变化（lr=0.1）")
print('=' * 56)

for label, per_layer in cases.items():
    grad = per_layer ** 10
    if grad >= 1:
        print(f'  {label:24s} 梯度 {grad:.3e} → 在放大，不适用')
    else:
        steps = 0.5 / (LR * grad)
        print(f'  {label:24s} 梯度 {grad:.3e} → 需要 {steps:.3e} 步')

# 附：把最大的那个换算成“按每秒 1000 步，要算多久”
worst_steps = 0.5 / (LR * 0.05 ** 10)
years = worst_steps / 1000 / 86400 / 365
print()
print(f'  参考：sigmoid(0.05) 那组的 {worst_steps:.3e} 步，')
print(f'        按每秒 1000 步算，需要约 {years:.0f} 年。')

# ============ 自己扩展 ============
# TODO:
# 1. 手算核对：第 1 层梯度 1e-7、lr=0.1 时，走完 0.5 需要多少步？
#    （对一下实验5 里 sigmoid 那一行）
#
# 2. 想一想：为什么"把 lr 调大 1000 倍"救不了第 1 层？
#    提示：别只看第 1 层，想想输出层同时会发生什么。
#
# 我的答案（第一次）：
# 因为第 1 层的梯度已经非常小了，乘以 1000 之后还是非常小，导致第 1 层的权重变化非常小。
#
# 正确答案要点：
# 1e-7 乘 1000 变成 1e-4，其实已经够用了。真正的问题在另一头 —— 同一个 lr 乘 1000 后，
# 输出层的梯度 0.25 会变成 250，训练直接爆炸。一个全局 lr 没法同时伺候"差 6 个数量级"的两端。
#
# 我的正确答案（打完字后写在这里）：
#
