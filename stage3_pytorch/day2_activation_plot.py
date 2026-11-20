"""
day2 任务3：绘制 sigmoid / tanh / ReLU / Leaky ReLU 及其导数

画图部分已经写好，你只需要检查并理解上面 8 个函数体（尤其 dsigmoid 的 s*(1-s)）。

运行：D:\\Anaconda\\anaconda3\\python.exe day2_activation_plot.py
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 4 个激活函数 + 4 个导数
# 关键式子（必须记住的只有第一条）：
#   dsigmoid(x) = s * (1 - s)，其中 s = sigmoid(x)，峰值 0.25
#   dtanh(x)    = 1 - tanh(x)^2，峰值 1.0
#   drelu(x)    = x>0 时 1，否则 0
#   dleaky(x)   = x>0 时 1，否则 0.01
# ============================================================


def sigmoid(x):
   return 1 / (1+np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(x, 0)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def dsigmoid(x):
   s = sigmoid(x)
   return s * (1-s)


def dtanh(x):
    return 1 - np.tanh(x) ** 2


def drelu(x):
    return (x > 0).astype(float)


def dleaky_relu(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)


# ============================================================
# 画图部分
# ============================================================

x = np.linspace(-6, 6, 400)

funcs = [
    ("sigmoid", sigmoid(x), "tab:blue"),
    ("tanh", tanh(x), "tab:orange"),
    ("ReLU", relu(x), "tab:green"),
    ("Leaky ReLU", leaky_relu(x), "tab:red"),
]

derivs = [
    ("sigmoid'", dsigmoid(x), "tab:blue"),
    ("tanh'", dtanh(x), "tab:orange"),
    ("ReLU'", drelu(x), "tab:green"),
    ("Leaky ReLU'", dleaky_relu(x), "tab:red"),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

# ---- 左图：原函数 ----
for name, y, c in funcs:
    axes[0].plot(x, y, color=c, linewidth=2, label=name)
axes[0].axhline(0, color="gray", linewidth=0.8, alpha=0.5)
axes[0].axvline(0, color="gray", linewidth=0.8, alpha=0.5)
axes[0].set_title("Activation functions")
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=9)

# ---- 右图：导数 ----
for name, y, c in derivs:
    axes[1].plot(x, y, color=c, linewidth=2, label=name)
axes[1].axhline(0.25, color="black", linestyle="--", linewidth=1,
                label="0.25 = upper bound of sigmoid'")
axes[1].axhline(1.0, color="gray", linestyle=":", linewidth=1,
                label="1.0")
axes[1].set_title("Derivatives  (how much gradient survives per layer)")
axes[1].set_xlabel("x")
axes[1].set_ylabel("f'(x)")
axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=9, loc="upper right")

plt.tight_layout()
plt.show()
