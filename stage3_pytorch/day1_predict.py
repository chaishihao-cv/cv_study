"""
day1 追加：预测 w、b 的精确答案（最小二乘闭式解）

用途：做"先预测再验证"练习时，先用这个算出理论最优，再和梯度下降的结果对比。
结论：只要收敛，梯度下降的终点 = 闭式解；学习率只影响速度，不影响终点。

运行：D:\Anaconda\anaconda3\python.exe day1_predict.py
"""

import numpy as np

# ===== 改这里 =====
SEED = 42
NOISE = 0.3          # 试试 0.2 / 0.3 / 0.9
LR = 0.1
EPOCHS = 200
# ==================

x = np.linspace(-1, 1, 100)
rng = np.random.default_rng(SEED)
y = 2.0 * x + 3.0 + rng.standard_normal(100) * NOISE

mean_x = np.mean(x)
mean_x2 = np.mean(x ** 2)
mean_xy = np.mean(x * y)
mean_y = np.mean(y)

# 解二元一次方程：
#   w*mean_x2 + b*mean_x = mean_xy
#   w*mean_x  + b        = mean_y
denom = mean_x2 - mean_x ** 2          # 一般形式的分母
w_star = (mean_xy - mean_x * mean_y) / denom
b_star = mean_y - w_star * mean_x

loss_star = np.mean((w_star * x + b_star - y) ** 2)

# 梯度下降实际跑
w, b = 0.0, 0.0
for _ in range(EPOCHS):
    pred = w * x + b
    gw = np.mean(2 * (pred - y) * x)
    gb = np.mean(2 * (pred - y))
    w = w - LR * gw
    b = b - LR * gb
loss_gd = np.mean((w * x + b - y) ** 2)

print(f"噪声 = {NOISE}, 学习率 = {LR}, 轮数 = {EPOCHS}, seed = {SEED}")
print()
print("【闭式解：理论最优】")
print(f"  w* = {w_star:.4f}   b* = {b_star:.4f}   loss* = {loss_star:.4f}")
print()
print("【梯度下降实际】")
print(f"  w  = {w:.4f}   b  = {b:.4f}   loss  = {loss_gd:.4f}")
print()
print(f"参数偏差：w 差 {abs(w - w_star):.6f}，b 差 {abs(b - b_star):.6f}")
print(f"噪声实际方差 = {np.var(y - (2.0 * x + 3.0)):.4f}（loss 地板大致等于这个数）")
