import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


TRUE_W = 2.0
TRUE_B = 3.0


x = np.linspace(-1, 1, 100)
noise = np.random.randn(100) * 0.3
y = TRUE_W * x + TRUE_B + noise

w = 0.0
b = 0.0


lr = 1.5
epochs = 50

loss_history = []


for epoch in range(epochs):

    y_pred = w * x + b

    loss = np.mean((y_pred - y) ** 2)
    loss_history.append(loss)

    grad_w = np.mean(2 * (y_pred - y) * x)
    grad_b = np.mean(2 * (y_pred - y))

    w = w -  lr * grad_w
    b = b -  lr * grad_b

    if epoch % 50 == 0 or epoch == epochs - 1:
        print(f'epoch {epoch:3d} | loss {loss:8.4f} | w {w: 6.3f} | b {b: 6.3f}')


print ('\n训练结束')
print(f'真是直线: y = {TRUE_W}x + {TRUE_B}')
print(f'预测直线: y = {w: .3f}x + {b: .3f}')
print(f'起始 loss {loss_history[0]:.4f} -> 最终 loss {loss_history[-1]:.4f}')


fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(loss_history,color='red')
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('MSE loss')
axes[0].set_title('Loss curve')
axes[0].grid(alpha=0.3)


axes[1].scatter(x, y, s=12, alpha=0.6, label="data (with noise)")
axes[1].plot(x, w * x + b, color="red", linewidth=2, label=f"y = {w:.2f}x + {b:.2f}")
axes[1].plot(x, TRUE_W * x + TRUE_B, color="black", linestyle="--", linewidth=1,
             label=f"ground truth y = {TRUE_W}x + {TRUE_B}")
axes[1].set_title("Fitted line vs ground truth")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()
