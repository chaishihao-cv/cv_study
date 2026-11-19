# day1 一元函数梯度下降拟合，画 loss
# 思考题：为什么必须划分验证集？过拟合有哪些直观现象？
#
# ---------------------------------------------------------------
# 今天要做的事：让电脑自己"猜"出一条直线 y = w*x + b
#   给它 100 个带噪声的点。它一开始完全不知道 w、b 是多少（都从 0 开始），
#   然后每轮算一下"我猜的和真实值差多少"，按差的方向微调一点点，重复 200 次。
#   这就是深度学习训练的全部本质：预测 -> 算损失 -> 求梯度 -> 更新参数。
#
# 运行方式（在 D:\cv_study\stage3_pytorch 目录下）：
#   python day1_gradient_descent.py
# ---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# 固定随机种子：保证每次运行生成的噪声数据一模一样，方便你对比实验
np.random.seed(42)

# ========== 第 1 步：造一批假数据 ==========
# 真实项目里答案未知，这里为了验证程序对不对，先设定一个"标准答案"
TRUE_W = 2.0   # 真实斜率
TRUE_B = 3.0   # 真实截距

x = np.linspace(-1, 1, 100)                 # 在 [-1, 1] 之间均匀取 100 个点
noise = np.random.randn(100) * 0.3          # 高斯噪声，模拟真实数据的抖动
y = TRUE_W * x + TRUE_B + noise             # 真实标签 = 真实直线 + 噪声

# ========== 第 2 步：初始化要学习的参数 ==========
# w 和 b 就是"模型参数"。深度网络里有几百万个这样的参数，原理完全一样
w = 0.0
b = 0.0

lr = 0.1          # 学习率：每轮走多大一步。太大容易跨过最优点，太小走得太慢
epochs = 200      # 训练轮数：把所有数据看多少遍

loss_history = []  # 记录每一轮的损失，最后画曲线用

# ========== 第 3 步：训练循环（这个结构以后每天都会见到）==========
for epoch in range(epochs):
    # (1) 前向传播：用当前参数算预测值
    y_pred = w * x + b

    # (2) 计算损失：均方误差 MSE，衡量"猜得有多离谱"
    #     loss = 平均((预测值 - 真实值)^2)，平方是为了让正负误差都变成正数
    loss = np.mean((y_pred - y) ** 2)
    loss_history.append(loss)

    # (3) 计算梯度：损失分别对 w、b 求导，告诉你"往哪个方向调最有效"
    #     这两行是链式法则手推的结果，先记住结论：
    #       dLoss/dw = 平均( 2*(y_pred - y)*x )
    #       dLoss/db = 平均( 2*(y_pred - y) )
    grad_w = np.mean(2 * (y_pred - y) * x)
    grad_b = np.mean(2 * (y_pred - y))

    # (4) 更新参数：沿梯度的反方向走一小步（所以是减号）
    w = w - lr * grad_w
    b = b - lr * grad_b

    # 每 50 轮打印一次，观察 loss 和参数怎么变化
    if epoch % 50 == 0 or epoch == epochs - 1:
        print(f"epoch {epoch:3d} | loss {loss:8.4f} | w {w:6.3f} | b {b:6.3f}")

# ========== 第 4 步：看结果 ==========
print("\n训练结束")
print(f"真实直线：y = {TRUE_W} x + {TRUE_B}")
print(f"学出来的：y = {w:.3f} x + {b:.3f}")
print(f"起始 loss {loss_history[0]:.4f} -> 最终 loss {loss_history[-1]:.4f}")

# ========== 第 5 步：画两张图 ==========
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# 左图：loss 随训练轮数下降，这是判断"模型有没有在学"最快的指标
axes[0].plot(loss_history, color="tab:blue")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("MSE loss")
axes[0].set_title("Loss curve")
axes[0].grid(alpha=0.3)

# 右图：散点是真实数据，红线是学出来的直线，看它有没有贴合
axes[1].scatter(x, y, s=12, alpha=0.6, label="data (with noise)")
axes[1].plot(x, w * x + b, color="red", linewidth=2, label=f"y = {w:.2f}x + {b:.2f}")
axes[1].plot(x, TRUE_W * x + TRUE_B, color="black", linestyle="--", linewidth=1,
             label=f"ground truth y = {TRUE_W}x + {TRUE_B}")
axes[1].set_title("Fitted line vs ground truth")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 跑通之后做三个实验，比看十篇教程有用：
#   实验1：lr 改成 0.01，观察 200 轮够不够收敛（loss 停在多少？）
#   实验2：lr 改成 1.5，观察 loss 是不是不降反升（步长太大，跨过了谷底）
#   实验3：epochs 改成 50，看 loss 曲线是不是还没到底就停了
# 每个实验记一句结论，写进笔记里。
# ---------------------------------------------------------------
