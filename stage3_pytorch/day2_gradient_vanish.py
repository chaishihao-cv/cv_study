"""
day2 任务4：梯度消失实验

观察：梯度从输出层往回传时，每经过一层就被乘一次"该层导数"
      sigmoid 每层最多 0.25 → 层数一多，梯度趋近 0
      ReLU    正区间每层 = 1 → 不衰减

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
print("实验2：sigmoid 的 0.25，层数越多越致命")
print("=" * 56)

for layers in (5, 10, 20, 30, 50):
    grad = 0.25 ** layers
    print(f"  {layers:2d} 层 → {grad:.3e}")

print()
print("=" * 56)
print("实验3：逐层打印，看它怎么一步步变小（这就是你之前跑的那段）")
print("=" * 56)

grad = 1.0
for layer in range(1, 11):
    grad = grad * 0.25
    print(f"  第 {layer:2d} 层传回的梯度: {grad:.10f}")

print()
print("=" * 56)
print("实验4：换算成'权重每次能移动多少'")
print("=" * 56)

LR = 0.1
for layers, per_layer in ((10, 0.25), (10, 1.0)):
    g = per_layer ** layers
    print(f"  每层 {per_layer}，{layers} 层后梯度 = {g:.3e}，"
          f"lr={LR} 时每步只移动 {LR * g:.3e}")

# ============ 自己扩展（任务4 后半段）============
# TODO:
# 1. 把 sigmoid 换成 tanh（每层 0.8），再跑一次实验1和实验3
# 2. 算一下：如果第 1 层梯度只有 1e-7，lr=0.1，要走完 0.5 的权重变化需要多少步？
#    提示：步数 = 0.5 / (lr * grad)
# 3. 想一想：为什么"把 lr 调大 1000 倍"救不了第 1 层？（写答案在下面）
#
# 我的答案：
