"""
day2 任务4：梯度消失实验

观察：梯度从输出层往回传时，每经过一层就被乘一次"该层导数"
      sigmoid 每层最多 0.25 → 层数一多，梯度趋近 0
      ReLU    正区间每层 = 1 → 不衰减
      tanh    实际约 0.8    → 10 层还能剩 0.107

运行：D:\\Anaconda\\anaconda3\\python.exe day2_gradient_vanish.py
"""

import numpy as np

print("=" * 56)
print("实验1：不同激活函数，10 层连乘后第 1 层还能收到多少梯度")
print("=" * 56)

cases = {
    "sigmoid (理论上限 0.25)": 0.25,
    "sigmoid (远离 0 处的 0.05)": 0.05,
    "tanh (实际情况 0.8)": 0.8,
    "ReLU (正区间)": 1.0,
    "假设某层 1.5 (会爆炸)": 1.5,
}

for name, per_layer in cases.items():
    grad = 1.0
    for layer in range(10, 0, -1):
        grad = grad * per_layer
        print(f"  {name:28s} 第 {layer}层收到: {grad:.3e}")

print()
print("=" * 56)
print("实验2：层数越多越致命 —— sigmoid 与 tanh 并排对比")
print("=" * 56)
print(f"  {'层数':>4}   {'sigmoid 0.25':>14}   {'tanh 0.8':>12}")

for layers in (5, 10, 20, 30, 50):
    g_sig = 0.25 ** layers
    g_tanh = 0.8 ** layers
    print(f"  {layers:>4}   {g_sig:>14.3e}   {g_tanh:>12.3e}")

print()
print("=" * 56)
print("实验3：逐层并排对比 —— 看两者差距怎么一层层拉开")
print("=" * 56)
print(f"  {'层':>3}   {'sigmoid 0.25':>14}   {'tanh 0.8':>12}   {'tanh/sigmoid':>13}")

g_sig, g_tanh = 1.0, 1.0
for layer in range(10, 0, -1):
    g_sig = g_sig * 0.25
    g_tanh = g_tanh * 0.8
    print(f"  {layer:>3}   {g_sig:>14.3e}   {g_tanh:>12.3e}   {g_tanh / g_sig:>13.1e}")

print()
print("=" * 56)
print("实验4：换算成'权重每次能移动多少'")
print("=" * 56)

LR = 0.1
for label, per_layer in (("sigmoid 0.25", 0.25), ("tanh 0.8", 0.8), ("ReLU 1.0", 1.0)):
    g = per_layer ** 10
    print(f"  每层 {label}，10 层后梯度 = {g:.3e}，"
          f"lr={LR} 时每步移动 {LR * g:.3e}")

print()
print("=" * 56)
print("实验5：最底层要走多少步才走得完 0.5 的权重变化")
print("=" * 56)

for label, grad in (("sigmoid 理论上限", 0.25 ** 10),
                    ("sigmoid 真实情况", 1e-7),
                    ("tanh 实际", 0.8 ** 10),
                    ("ReLU 正区间", 1.0)):
    steps = 0.5 / (LR * grad)
    print(f"  {label:16s} 梯度 {grad:.3e} → 需要 {steps:>12.3e} 步")

# ============ 自己扩展（任务4 后半段）============
# TODO:
# 1. 算一下：如果第 1 层梯度只有 1e-7，lr=0.1，要走完 0.5 的权重变化需要多少步？
#    提示：步数 = 0.5 / (lr * grad)      （实验5 已经帮你算出来了，自己先手算核对一遍）
#
# 2. 想一想：为什么"把 lr 调大 1000 倍"救不了第 1 层？（写答案在下面）
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
