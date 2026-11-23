# day4 任务卡 · MLP 训练 MNIST

> 日期：2026-11-23　｜　阶段3：深度学习基础 + PyTorch
> 讲义：`D:\清美教育课件\计算机视觉3阶段学习\3阶段\day4-详细讲义.md`
> 脚本：`D:\cv_study\stage3_pytorch\day4_mlp_mnist.py`（骨架已存在，直接改）
> 数据：`D:\cv_study\data\MNIST`（**已下载好，不要再 download=True 重下**）
> 解释器：**必须是 `D:\Anaconda\anaconda3\python.exe`**

---

## 开始前必做（1 分钟）

1. 跑 `D:\cv_study\check_env.py`，最后一行要是 `✅ 环境正常`
2. `D:\cv_study\data\MNIST` 目录要存在（里面有 `raw` 和 `processed` 两个子目录）

**这两条不满足，今天不用动手。**

---

## 任务 1 · 看一眼数据（10 分钟）

在脚本里写：

```python
from torchvision import datasets, transforms

train_set = datasets.MNIST("D:/cv_study/data", train=True, download=False)
test_set  = datasets.MNIST("D:/cv_study/data", train=False, download=False)

print(len(train_set), len(test_set))
img, label = train_set[0]
print(img.shape, label)
```

**要看到：** `60000 10000` 和 `(1, 28, 28)` 和一个 `int` 标签。

**注意这里没加 `transform`** —— 所以拿到的 `img` 是 PIL 图片，不是张量。这一步就是让你体会：**`transform=ToTensor()` 到底做了什么转换。** 加了它之后 `img.shape` 会变成 `(1,28,28)` 的浮点张量、值域 0~1。

**验收**：能说出 `ToTensor()` 做了哪两件事（形状变成 CHW、像素值除以 255）。

---

## 任务 2 · 先搭 Sequential 版，再手算参数量（25 分钟）

**分两步，顺序不要跳：**

**第一步，先在纸上算**，把这三个数写下来（不许用代码）：

```
784 → 128 这层有多少参数？
128 → 10 这层有多少参数？
一共多少？
```

**第二步，写代码验证：**

```python
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 10),
)
print(model)

total = sum(p.numel() for p in model.parameters())
print("总参数:", total)

x = torch.randn(32, 1, 28, 28)
print("输入:", x.shape, "输出:", model(x).shape)
```

**要看到**：`总参数: 101770`、`输入 (32,1,28,28) 输出 (32,10)`。

再打印每个参数的名字和形状：

```python
for name, p in model.named_parameters():
    print(name, tuple(p.shape), p.numel())
```

**验收**：能解释为什么 weight 是 `(128, 784)` 而不是 `(784, 128)`。

---

## 任务 3 · 改写成 class（25 分钟）

把任务 2 的 Sequential 改写成：

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 在这里装层

    def forward(self, x):
        # 在这里接线
        return ...
```

**要求：**

1. 层在 `__init__` 里定义，运算在 `forward` 里发生
2. 用 `self.fc1`、`self.fc2` 命名（不要叫 `self.a`、`self.b`）
3. 跑一次 `model(x)` 确认输出 `(32,10)`
4. 打一次 `model`，对照着看 Sequential 版本打印的结构

**验收**：能说出 `model(x)` 和 `model.forward(x)` 的区别，以及为什么用前者。

---

## 任务 4 · 跑通 1 个 epoch（35 分钟，今天的核心）

**先只跑 1 个 epoch，不要一上来就 5 轮。** 跑通了再看准确率。

需要写全这些：

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.ToTensor()
train_set = datasets.MNIST("D:/cv_study/data", train=True,  download=False, transform=transform)
test_set  = datasets.MNIST("D:/cv_study/data", train=False, download=False, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False, num_workers=0)

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP().to(dev)
crit  = nn.CrossEntropyLoss()
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
```

训练循环七步（讲义第六节），每步都要有：

```python
model.train()
for x, y in train_loader:
    x, y = x.to(dev), y.to(dev)
    out  = model(x)
    loss = crit(out, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
```

**Windows 必须 `num_workers=0`**，否则会报 `DataLoader worker exited unexpectedly`。

**验收**：1 个 epoch 跑完，train loss ≈ **0.34**。如果 loss 是 2.x 或一直不降，先别继续，来找我。

---

## 任务 5 · 加评估，跑 5 轮，画曲线（25 分钟）

**加一个评估函数：**

```python
def evaluate():
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(dev), y.to(dev)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
    return correct / len(test_set)
```

然后每轮训练完调一次，打印：

```
epoch 1  loss=0.3437  test_acc=94.44%
```

**要看到的大致曲线**（实测参考）：

| epoch | loss | test_acc |
|---|---|---|
| 1 | 0.3437 | 94.44% |
| 2 | 0.1573 | 96.26% |
| 3 | 0.1096 | 97.15% |
| 4 | 0.0841 | 97.35% |
| 5 | 0.0668 | 97.58% |

**最后画 loss 曲线**（用 `loss_history` 列表）：

```python
plt.plot(loss_history)
plt.xlabel("epoch"); plt.ylabel("loss")
plt.title("MLP on MNIST")
plt.grid(alpha=0.3)
plt.show()
```

**验收**：末轮 test acc ≥ **97%**。

---

## 任务 6 · 保存 + 加载 + 单张推理（20 分钟）

```python
torch.save(model.state_dict(), "D:/cv_study/stage3_pytorch/day4_mlp.pth")
```

然后**新建一个模型实例**，把权重 load 回来，对测试集第 0 张图做推理：

```python
model2 = MLP().to(dev)
model2.load_state_dict(torch.load("D:/cv_study/stage3_pytorch/day4_mlp.pth", map_location=dev))
model2.eval()

img, true_label = test_set[0]
with torch.no_grad():
    logits = model2(img.unsqueeze(0).to(dev))   # 加 batch 维
    pred = logits.argmax(dim=1).item()
print(f"预测 {pred}，真值 {true_label}")
```

**注意 `img.unsqueeze(0)`** —— 单张图是 `(1,28,28)`，模型要的是 `(N,1,28,28)`。**这就是 day3 学的 `unsqueeze` 的真实用途。**

**验收**：预测和真值一致，且能说出 `state_dict()` 和整个 `model` 保存的区别。

---

## 思考题（写在脚本末尾当注释）

讲义骨架里原有的两个，加上新的两个：

1. **反向传播更新的是什么？** 是 loss 吗？是输入吗？是参数吗？
2. **为什么推理时不需要计算梯度？** 如果硬要算，会多花什么？
3. `optimizer.zero_grad()` 如果忘写，loss 曲线会变成什么样？为什么？
4. 为什么模型最后**不要**加 softmax 再传给 `CrossEntropyLoss`？

---

## 遇到报错怎么办

**先做两件事**：① `print(x.shape, y.shape, out.shape)` ② 把报错原文复制下来搜。

讲义第八节有 5 个必踩报错的对照表。**特别记住这一条：**

> **准确率卡在 10% = 模型完全没学到东西 = 管线 bug，不是调参问题。**

---

## 今天的完成标准

- [ ] 5 轮跑完，末轮 test acc ≥ 97%
- [ ] 打印过每层的输出 shape
- [ ] 手算参数量 101770，和代码对上了
- [ ] 画出了 loss 曲线
- [ ] 保存并成功 load 回来做了一次推理
- [ ] 4 道思考题能口述

**不要为了刷更高准确率去调参。** MLP 天花板约 98%，想上去要换 CNN —— 那是后面的事。

---

## 提交

```powershell
cd D:\cv_study
git add -A
git commit -m "day4: MLP 训练 MNIST"
git push
```
