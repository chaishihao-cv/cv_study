"""
day1 任务4：迁移测试 —— 把 MSE 换成 MAE

【任务】
1. 损失从 mean((pred-y)^2) 改成 mean(|pred-y|)
2. 自己推出 MAE 对 w、b 的梯度（不许直接照抄 MSE 的梯度！）
3. 跑通，观察收敛行为和 MSE 有什么不同
4. 加一个离群点，比较 MSE 和 MAE 谁更抗干扰

【规则】
- 梯度必须自己推。推导写在下面注释里。
- 跑之前先写预测，跑完对照。

【提示】
- 绝对值 |u| 的导数：u>0 时是 +1，u<0 时是 -1，即 sign(u)
- numpy 里有 np.sign()，也可以用 (pred > y) 这种布尔写法
- |u| 在 u=0 处不可导，但概率几乎为 0，实际不影响

运行：D:\Anaconda\anaconda3\python.exe day1_practice_mae.py
"""

import numpy as np

# ========== 我的推导（自己写，写完再往下看代码）==========
# L = mean(|pred - y|)
#L = np.mean(np.abs(pred - y))
#∂L/∂w = np.mean(np.sign(pred - y) * x)
#∂L/∂b = np.mean(np.sign(pred - y))
# ===========================================================

SEED = 42
NOISE = 0.2
LR = 0.1          # 提示：MAE 的梯度大小恒定，lr 可能要调小，试试 0.5 / 0.1 / 0.01
EPOCHS = 200

x = np.linspace(-1, 1, 100)
rng = np.random.default_rng(SEED)
y = 2.0 * x + 3.0 + rng.standard_normal(100) * NOISE

# TODO 1: 在下面加一个离群点（先不加，跑通后再加来对比）
OUTLIER = True
if OUTLIER:
    y[0] = 50.0

w, b = 0.0, 0.0
loss_history = []

for epoch in range(EPOCHS):
    pred = w * x + b

    # TODO 2: 把这里改成 MAE
    loss = np.mean(np.abs(pred - y))
    loss_history.append(loss)

    # TODO 3: 把梯度改成 MAE 的梯度（自己推的结果）
    grad_w = np.mean(np.sign(pred - y) * x)
    grad_b = np.mean(np.sign(pred - y))

    w = w - LR * grad_w
    b = b - LR * grad_b

    if epoch % 25 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch:3d} | loss {loss:8.4f} | w {w:6.3f} | b {b:6.3f}")

print()
print("训练结束")
print(f"真实直线：y = 2x + 3")
print(f"预测直线：y = {w:.2f}x + {b:.2f}")
print(f"起始 loss {loss_history[0]:.4f} -> 最终 loss {loss_history[-1]:.4f}")

# ========== 观察记录（跑完填）==========
# 1. MAE 最终 loss 是多少？和噪声水平（0.2）有什么关系？
# 2. 收敛速度和 MSE 比，谁快？
# 3. 曲线是平滑下降，还是带抖动？
# 4. 加离群点后，w 偏移多少？和 MSE 的结果比哪个更小？
# =========================================
