# 闭卷从零写：梯度下降训练循环
#
# 规则：
#   1. 不许打开 day1_gradient_descent.py
#   2. 只允许查 numpy 官方文档（不许查整段代码）
#   3. 写完运行，loss 必须能下降，否则回去 debug
#
# 你需要自己写出：
#   - 造数据：y = 2x + 3 + 噪声，100 个点
#   - 初始化 w、b
#   - 训练循环（前向 / 损失 / 梯度 / 更新 / 记录）
#   - 打印 loss 和参数
#   - 画 loss 曲线
#
# 及格线：最终 loss < 0.1，w ≈ 2.0，b ≈ 3.0
#
# 参考运行命令（在 D:\cv_study\stage3_pytorch 下）：
#   D:\Anaconda\anaconda3\python.exe day1_practice.py

# 在这里写代码（不允许看参考文件）
import numpy as np
import matplotlib.pyplot as plt


w = 0.0
b = 0.0
lr = 0.1
epochs = 200
loss_history = []
x = np.linspace(-1, 1, 100)
noise = np.random.randn(100) * 0.2
y = 2 * x + 3 + noise

for epoch in range(epochs):
    # 前向传播
    y_pred = w * x + b 
    # 计算损失
    loss = np.mean((y_pred - y) ** 2)
    loss_history.append(loss)
    # 计算梯度
    grad_w = np.mean(2 * (y_pred - y) * x)
    grad_b = np.mean(2 * (y_pred - y))
    # 更新参数
    w = w- lr * grad_w
    b = b- lr * grad_b

    # 打印
    if epoch % 25 == 0 or epoch == epochs - 1:
        print(f'epoch {epoch:3d} | loss {loss:8.4f} | w {w: 6.3f} | b {b: 6.3f}')

print('\n 训练结束')
print(f'真实直线: y = 2x + 3')
print(f'预测直线：y = {w:.2f}x + {b:.2f}')
print(f'起始 loss {loss_history[0]:.4f} -> 最终 loss {loss_history[-1]:.4f}')





