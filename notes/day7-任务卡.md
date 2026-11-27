# day7 任务卡 · BatchNorm 与 Dropout

> 日期：2026-11-27　｜　预计 3 小时　｜　讲义：`notes/day7-详细讲义.md`
> 代码：`D:\cv_study\stage3_pytorch\day7_cnn_bn_dropout.py`（骨架已建好，6 个任务按顺序填）
> 规矩：**先手算 → 再跑代码核对 → 不一致就查**。不要先跑代码再看答案。

**时间预算**

| 环节 | 时间 |
|---|---|
| 任务 1 手算 BN | 20 分钟 |
| 任务 2 数 BN 家底 | 15 分钟 |
| 任务 3 train/eval + 报错 | 30 分钟 |
| 任务 4 Dropout | 25 分钟 |
| 任务 5 收敛对比（最长） | 50 分钟 |
| 任务 6 手写 BN | 40 分钟 |
| 思考题 + 记笔记 | 25 分钟 |

---

## 任务 1 · 手算一遍 BN（20 分钟）

给你 4 个样本、2 个通道（`shape (4, 2)`）：

```text
x = [[1.0,  2.0],
     [2.0,  2.0],
     [3.0,  2.0],
     [4.0, 10.0]]
```

`nn.BatchNorm1d(2)`，默认 `γ=1, β=0`，`eps=1e-5`（可忽略），`train` 模式。

**空 1**：通道 0 的 `μ` = ？　`σ²` = ？　`σ` = ？

**空 2**：通道 0 标准化后的 4 个值（保留 4 位小数）= ？

**空 3**：通道 1 的 `μ` = ？　`σ²` = ？（注意：三个 2.0，一个 10.0）

**空 4**：通道 1 标准化后的 4 个值 = ？（提示：三个相同的值会变成同一个数）

**然后代码核对**：

```python
bn = nn.BatchNorm1d(2)
y = bn(x)              # 注意 x 要 float，且 bn 默认在 train 模式
```

**核对标准**：4 个空全对，误差 < 1e-3（eps 会带来一点点偏差）。

**顺便回答（写进注释）**：为什么通道 1 里有三个数标准化后完全一样？

---

## 任务 2 · 数清 BN 的家底（15 分钟）

对 `nn.BatchNorm2d(64)` 和 `nn.BatchNorm1d(128)` 各数一遍，填表：

| 项 | `BatchNorm2d(64)` | `BatchNorm1d(128)` |
|---|---|---|
| `weight` 元素数 |  |  |
| `bias` 元素数 |  |  |
| `running_mean` 元素数 |  |  |
| `running_var` 元素数 |  |  |
| `num_batches_tracked` |  |  |
| `state_dict` 总元素 |  |  |
| 其中 requires_grad=True 的个数 |  |  |
| 占 `Conv2d(64,64,3)`（36,928 参数）的百分比 |  | — |

**代码提示**：

```python
sum(p.numel() for p in bn.parameters())        # 可学
sum(v.numel() for v in bn.state_dict().values())  # 全部
sum(p.numel() for p in bn.parameters() if p.requires_grad)
```

**要能一句话说**：BN 的可学参数是 `____ × C`，占卷积参数的百分之几 → 所以能无脑加。

---

## 任务 3 · train / eval 两种模式（30 分钟）

### 3.1 同输入、两模式，差多少

```python
bn = nn.BatchNorm1d(16)
big = torch.randn(1000, 16) * 2 + 5     # 先喂一大批，攒 running stats
bn(big)
x1 = torch.randn(4, 16) * 2 + 5         # 再来一个小 batch
bn.train(); o_tr = bn(x1)
bn.eval();  o_ev = bn(x1)
```

填表：

| | 输出均值 | 输出标准差 |
|---|---|---|
| train 模式 |  |  |
| eval 模式 |  |  |

**空 5**：两者最大差 = ？

**空 6**：为什么 eval 模式的输出均值不是 0，而是一个明显偏离 0 的数？（提示：`running_mean` 只被喂过 1 批，`0.1 × 第一批均值`）

### 3.2 复现 batch=1 的报错

```python
bn = nn.BatchNorm1d(16); bn.train()
bn(torch.randn(1, 16))
```

**空 7**：把报错原文完整抄下来（写进注释）。

**必答**：为什么 `BatchNorm1d` 怕 batch=1，而 `BatchNorm2d(8)` 输入 `(1, 8, 4, 4)` 不怕？

### 3.3 batch=2 的极端情况

```python
bn = nn.BatchNorm1d(16); bn.train()
print(bn(torch.randn(2, 16)).round(decimals=2))
```

**空 8**：你看到了什么规律？（提示：每行大概是 ±1 的组合）

**必答**：这说明 BN 在什么规模的数据上不可靠？检测/分割任务 batch 只有 2~4 时该怎么办？（讲义第五节）

---

## 任务 4 · Dropout 的三个数字（25 分钟）

### 4.1 p=0.5

```python
d = nn.Dropout(0.5); d.train()
one = torch.ones(1000)
o = d(one)
print((o > 0).float().mean())   # 保留比例
print((o == 0).float().mean())  # 丢掉的
print(o[o > 0][0])              # 非零值
```

| 项 | 实测 |
|---|---|
| 保留比例 |  |
| 被置 0 比例 |  |
| 非零值 |  |

### 4.2 p=0.3（**先手算再跑**）

**空 9**：p=0.3 时，非零值应该是多少？（写公式，再算数值）

### 4.3 eval 模式

```python
d.eval()
print(d(one))
```

**空 10**：eval 模式下输出是什么？为什么不需要再乘任何东西？（提示：缩放已经在训练时做完了）

### 4.4 两次 forward 不一样

```python
d.train()
a = d(one); b = d(one)
print(torch.equal(a, b), (a != b).sum())
```

**空 11**：两次结果相同吗？不同位置大约多少个？（预期约 500）

**必答**：这是 bug 吗？为什么？

---

## 任务 5 · BN 到底允许多大的学习率（50 分钟，今天的重头戏）

**固定条件**：MLP `784 → 256 → 128 → 10`，MNIST **子集 3000 训练 / 1000 测试**，SGD，**3 轮**，`torch.manual_seed(0)`。

**跑 6 组**：`use_bn ∈ {False, True}` × `lr ∈ {0.05, 0.5, 2.0}`。

**填表（训练 acc / 测试 acc）**：

| 配置 | ep1 | ep2 | ep3 |
|---|---|---|---|
| noBN lr=0.05 | / | / | / |
| noBN lr=0.5 | / | / | / |
| **noBN lr=2.0** | / | / | / |
| BN lr=0.05 | / | / | / |
| BN lr=0.5 | / | / | / |
| **BN lr=2.0** | / | / | / |

**填完必须能回答三个问题（写进注释）**：

1. `noBN lr=2.0` 停在多少？这叫什么现象？
2. `BN lr=2.0` 和 `BN lr=0.05` 差多少？说明什么？
3. 同样 `lr=0.05`，BN 比 noBN 高多少个百分点？

**代码提示**（结构，不是答案）：

```python
def make(use_bn):
    layers = [nn.Flatten(), nn.Linear(784, 256)]
    if use_bn: layers.append(nn.BatchNorm1d(256))
    layers += [nn.ReLU(), nn.Linear(256, 128)]
    if use_bn: layers.append(nn.BatchNorm1d(128))
    layers += [nn.ReLU(), nn.Linear(128, 10)]
    return nn.Sequential(*layers)
```

**两个必须记住的细节**：

```text
① 每组开跑前 torch.manual_seed(0)，否则 6 组初始化不同，没法比
② 评估前 net.eval()，评估代码包在 torch.no_grad() 里
```

**笔记本要能扛住**：6 组 × 3 轮 × 3000 张，约 1.7 秒/组，总共 10 秒左右。**如果超过 2 分钟，说明你用了全量 MNIST，去检查 `Subset`。**

---

## 任务 6 · 手写一个 BN（40 分钟，理解的分水岭）

**不许用 `nn.BatchNorm1d`**，自己实现一个 `MyBN`，要求：

```python
class MyBN:
    def __init__(self, C, momentum=0.1, eps=1e-5):
        self.gamma = torch.ones(C)
        self.beta  = torch.zeros(C)
        self.running_mean = torch.zeros(C)
        self.running_var  = torch.ones(C)
        ...
    def forward(self, x, training=True):
        ...
```

**四个要求**：

1. **train 模式**：用**当前 batch** 的均值/方差，**同时更新 running stats**
2. **eval 模式**：用 `running_mean / running_var`，**不更新**
3. 方差用**有偏**版本（除以 N，不是 N-1），和 PyTorch 一致
4. 输出和 `nn.BatchNorm1d` 对比，**最大差异 < 1e-5**

**验证代码**：

```python
torch.manual_seed(0)
x = torch.randn(32, 8) * 3 + 10

ref = nn.BatchNorm1d(8); ref.train()
y_ref = ref(x)

mine = MyBN(8)
y_mine = mine.forward(x, training=True)
print("train 最大差:", (y_ref - y_mine).abs().max().item())

# eval 对比：先把 ref 的 running stats 同步给 mine
mine.running_mean = ref.running_mean.clone()
mine.running_var  = ref.running_var.clone()
ref.eval()
print("eval 最大差:",
      (ref(x) - mine.forward(x, training=False)).abs().max().item())
```

**三个必答（写进注释）**：

- 为什么方差要除以 N 而不是 N-1？（提示：PyTorch 用的是总体方差）
- `running_var` 要不要用无偏修正？（提示：PyTorch 在 train 时用有偏更新 buffer，在 eval 时用的是 buffer 原值）
- 哪一行代码体现了"`γ`、`β` 可学"？（提示：`self.gamma * x_hat + self.beta`）

**做完这一步你就真懂 BN 了**——它没有任何魔法，就是 4 行公式 + 2 个 buffer。

---

## 4 道思考题（不用写代码，各写 3~5 行）

### 思考题 1

BN 里如果**只有标准化**（`x̂ = (x-μ)/σ`），去掉 γ、β，会失去什么？**举一个具体的层说明**（提示：如果某个层本来就想输出"均值 5、方差 0.1"的信号呢？）

### 思考题 2

`BatchNorm1d` 在 **train 模式 batch=1** 时直接报错，但**推理时 batch=1 却完全正常**。为什么？**这两种模式分别用谁算 μ、σ？**

### 思考题 3

Dropout 的"放大"可以放在两个地方：

```text
A（PyTorch 做法）：训练时放大 1/(1-p)，推理时不放大
B（老做法）      ：训练时不放大，推理时乘 (1-p)
```

**这两种做法在数学上等价吗？各自的代价是什么？为什么现在都用 A？**

### 思考题 4

`Conv → BN → ReLU → Dropout` 这个顺序，如果把 **BN 和 Dropout 换位置**（`Conv → ReLU → Dropout → BN`）会发生什么？**小 batch 时问题会放大还是缩小？**

---

## 完成标准（自己打勾）

- [ ] 任务 1：4 个空手算全对，代码核对误差 < 1e-3
- [ ] 任务 2：两张表填满，能说"BN 参数只有卷积的 0.35%"
- [ ] 任务 3：train/eval 差异填对，**报错原文抄下来了**，能解释 batch=1 与 batch=2
- [ ] 任务 4：三个数字（0.502 / 2.0 / 1.0）对得上，p=0.3 手算对
- [ ] 任务 5：6 组 18 个数字填满，**noBN lr=2.0 → 约 10%、BN lr=2.0 → 约 90%**
- [ ] 任务 6：手写 BN 与官方差 < 1e-5，train/eval 都对
- [ ] 4 道思考题各写了 3 行以上

---

## 卡住对照表

| 症状 | 先查这里 |
|---|---|
| BN 输出不是 0/1 | 是不是 `bn.train()` 忘了？（默认就是 train，但手写版要自己判断） |
| 6 组结果全都一样 | 是不是没在每组开头 `manual_seed(0)`？ |
| `noBN lr=2.0` 没有崩 | 是不是 BN 混进去了？`make(True/False)` 传错 |
| 手写 BN 差很大 | 方差除的是 N-1？还是 `running_var` 初始化成 0 了？ |
| `BatchNorm1d` 报 batch=1 错 | **这是正确行为**，抄下来，别修 |
| Dropout 两次一样 | 是不是 `d.eval()` 没切回 `train()`？ |
| 任务 5 跑了几分钟 | 数据没用 `Subset`，回讲义第九节 |
| 报 "Expected more than 1 value per channel" | 就是任务 3.2 要复现的那个，抄原文 |

---

## 提交命令

```powershell
$git = "C:\Program Files\Git\cmd\git.exe"
& $git -C "D:\cv_study" add stage3_pytorch/day7_cnn_bn_dropout.py
& $git -C "D:\cv_study" commit -m "day7: BN 手算+参数家底+train/eval 差异+Dropout 三数+BN 收敛对比+手写 BN"
& $git -C "D:\cv_study" push
```

**push 失败等 15~25 秒重试 3~5 次。**
