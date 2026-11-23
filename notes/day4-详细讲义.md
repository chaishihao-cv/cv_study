# day4 详细讲义 · MLP 训练 MNIST

> 日期：2026-11-23　｜　阶段3：深度学习基础 + PyTorch
> 脚本：`D:\cv_study\stage3_pytorch\day4_mlp_mnist.py`
> 数据：`D:\cv_study\data\MNIST`（已下载，60000 训练 + 10000 测试）
> 阅读时间：约 40 分钟。**读完再动手，别边读边抄。**

---

## 一、今天的跨度：从"一个 w"到"一个真正的任务"

day1~day3 你一直在拟合一个函数：真值 w=2、b=3，两个参数。

今天第一次面对真实的分类任务：**784 维输入、10 个类别、101,770 个参数。**

| | day3 | day4 |
|---|---|---|
| 参数 | `w`、`b` 两个张量 | `nn.Linear` 里的 4 个参数张量，共 101,770 个数 |
| 损失 | MSE（回归） | **CrossEntropy（分类）** |
| 更新 | `w -= lr*w.grad` 手写 | **`optimizer.step()`** |
| 评估 | loss 数值 | **准确率（%）** |
| 数据 | 自己造 100 个点 | **MNIST 60000 张图** |

**核心变化只有一个：终于用上"框架"了。** 手写更新变成优化器、手写网络变成模块。

---

## 二、`nn.Module`：三件事，没别的

继承 `nn.Module` 的类，你只需要管三件事：

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()              # ① 必须调用父类初始化
        self.fc1 = nn.Linear(784, 128)  # ② 在这里"装"层
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):               # ③ 在这里"接"线
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

### 为什么 `__init__` 里写了层，就自动被找到了

`nn.Module` 内部维护了一个"子模块清单"：**只要你在 `__init__` 里给 `self.xxx` 赋一个 `nn.` 开头的对象，它就会被自动登记。**

所以：

```python
model.parameters()      # 自动把所有层的 weight/bias 都列出来
model.cuda()            # 自动把所有层搬到 GPU
model.state_dict()      # 自动打包所有参数
```

**你不用挨个写 `w1, w2, b1, b2` —— 这是 `nn.Module` 唯一真正的价值。** 层本身只是"装参数的容器"。

### 两种写法，选一个

| 写法 | 代码 | 什么时候用 |
|---|---|---|
| **Sequential**（简单） | `nn.Sequential(nn.Flatten(), nn.Linear(...), nn.ReLU(), ...)` | 层是纯线性串联，没有分支 |
| **自定义类**（灵活） | 上面那个 `class Net` | 有分支、有跳连、要复用某层 |

**今天两种都写一遍**：先 Sequential（5 行搞定），再改成类（看清楚对应关系）。

### 一个必须记住的细节

```python
model = Net()
out = model(x)         # ✅ 应该这样调
out = model.forward(x) # ❌ 不要这样
```

**`model(x)` 会自动触发 `forward`，同时处理 hook 等机制。** 直接调 `.forward()` 会绕过这些，以后加调试钩子就失效了。

---

## 三、`nn.Linear` 到底算了什么（含参数量手算）

### 数学公式

```
y = x @ W.T + b
```

| 符号 | 形状 | 说明 |
|---|---|---|
| `x` | `(N, in)` | N 个样本，每个 in 维 |
| `W` | `(out, in)` | **注意是 out 在前！** |
| `b` | `(out,)` | |
| `y` | `(N, out)` | |

**权重形状是 `(out, in)` 不是 `(in, out)`** —— 这是最容易记反的一点。原因：`nn.Linear` 存的是 `W^T`，方便做 `x @ W.T`。

实测（第一层 784→128）：

```
1.weight  shape=(128, 784)   数量=100352
1.bias    shape=(128,)       数量=128
```

### 参数量手算（今天必须会的加法）

```
第一层 784 → 128：784*128 + 128 = 100352 + 128 = 100480
第二层 128 → 10 ：128*10  + 10  = 1280   + 10  = 1290
                                        总计 = 101770
```

实测总计 **101,770** ✅

**记忆公式：一层参数量 = `in*out + out`（后一个 `+out` 是 bias）。**

**这就是以后面试问"你这个模型多大"的标准答法** —— 不是背数字，是能现场算。

### 形状流转（今天要打印出来看）

```
输入图片 (N, 1, 28, 28)
  ↓ nn.Flatten()        拍平，保留 batch 维
       (N, 784)
  ↓ nn.Linear(784,128)
       (N, 128)
  ↓ nn.ReLU()
       (N, 128)
  ↓ nn.Linear(128,10)
       (N, 10)           ← 10 个类别的 logits
```

实测 `(32,1,28,28)` → `(32,10)` ✅

**28×28 = 784 是怎么来的**：MNIST 是 28×28 灰度图，单通道，拍平就是 784 个数。

**为什么用 `nn.Flatten()` 而不是 `x.view(-1, 784)`**：`flatten(start_dim=1)` 会自动保留第 0 维（batch），batch size 变了也不用改代码。

---

## 四、从回归到分类：`CrossEntropyLoss`

### 为什么不能继续用 MSE

分类任务的输出是"10 个类别的分数"。用 MSE 会有两个问题：

1. **梯度太小**：softmax 之后的概率都在 0~1 之间，MSE 的梯度 ∝ `(p - y)`，量级很小，学得很慢
2. **不是概率意义下的最优**：MSE 假设高斯噪声，分类需要的是"最大化正确类别的概率"

**分类任务的标准答案：`nn.CrossEntropyLoss`。**

### 它内部做了两件事

```
CrossEntropy = softmax + NLL（负对数似然）
```

也就是：

```
loss = -log( softmax(logits)[正确类别] )
```

**所以你传进去的必须是"未归一化的分数"（logits），不要在模型最后加 softmax！**

**加了 softmax 再传给 CrossEntropyLoss = 做了两次 softmax，梯度会变得极小。这是新手最典型的错误之一。**

### 手算一个例子（实测核对过）

logits = `[2.0, 1.0, 0.1]`，正确类别是 0：

```
softmax 概率 = [0.659001, 0.242433, 0.098566]
loss = -log(0.659001) = 0.41703
```

实测 `nn.CrossEntropyLoss()` 输出 **0.41703** ✅ 完全一致。

**记住这个手算过程，面试问"交叉熵怎么算"就有话说了。**

### 标签的两个硬性要求

| 要求 | 错误写法 | 报错 |
|---|---|---|
| **必须是 `long`（int64）** | `torch.tensor([0.0])` | `expected scalar type Long but found Float` |
| **形状必须是 `(N,)`**，不是 one-hot | `(N, 10)` | 不报错，但会被当成"软标签"处理 |

**MNIST 的标签天然就是 long，直接能用。** 但只要你自己造标签，就一定会踩这个坑。

### 输出形状必须对得上

```python
out  = model(x)      # (N, 10)
loss = crit(out, y)  # y 是 (N,)
```

**`out` 的第 0 维必须和 `y` 的第 0 维一致**，否则：

```
ValueError: Expected input batch_size (32) to match target batch_size (10).
```

---

## 五、优化器：`step()` 到底做了什么

### SGD 就是 day3 手写那行

实测：

```python
p = torch.tensor([1.0, 2.0], requires_grad=True)
opt = torch.optim.SGD([p], lr=0.1)
(p**2).sum().backward()      # grad = 2p = [2.0, 4.0]
opt.step()
# p 变成 [0.8, 1.6]
```

手算：`1.0 - 0.1*2.0 = 0.8`、`2.0 - 0.1*4.0 = 1.6` ✅

**`opt.step()` 就是 `p = p - lr*p.grad` 循环一遍所有参数。**

**你 day3 手写过这一行，所以今天应该觉得"这不就是我写的那个嘛"。**

### Adam 又多了什么

Adam = SGD + 两个机制：

| 机制 | 作用 |
|---|---|
| **动量（一阶矩）** | 记住之前梯度的方向，方向一致的加速，来回震荡的抵消 |
| **自适应学习率（二阶矩）** | 每个参数有自己的有效 lr：梯度大的自动调小，梯度小的自动调大 |

**定性理解就够了**：Adam 对学习率不敏感，`lr=1e-3` 开箱即用；SGD 需要调，但调好了泛化有时更好。

**今天用 Adam（`lr=1e-3`），省得调。**

### 为什么必须 `zero_grad()`

day3 你已经实测过：**`backward()` 是 `+=`，梯度会累加。**

```python
optimizer.zero_grad()   # 每轮开头清零
loss.backward()         # 算这一轮的梯度
optimizer.step()        # 用它更新
```

**顺序不能反。** 忘写 `zero_grad()` 的典型症状：**loss 前几轮降，然后突然不降了甚至往上走**（因为梯度被历史累积污染）。

### 优化器怎么知道要更新哪些参数

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

**`model.parameters()` 就是你手动把"要更新的东西"交给优化器。** 没传进去的参数不会被更新（这是"冻结某层"的原理）。

---

## 六、完整训练循环：七步

```python
for epoch in range(EPOCHS):
    model.train()                    # ① 切训练模式
    for x, y in train_loader:
        x, y = x.to(dev), y.to(dev)  # ② 数据上设备
        out  = model(x)              # ③ 前向
        loss = crit(out, y)          # ④ 算损失

        optimizer.zero_grad()        # ⑤ 清梯度
        loss.backward()              # ⑥ 反传
        optimizer.step()             # ⑦ 更新
```

**这七步是深度学习里出现频率最高的代码，以后每一份训练脚本都是它的变体。**

对照 day3：

| day3 手写 | day4 框架 |
|---|---|
| `pred = w*x + b` | `out = model(x)` |
| `loss = ((pred-y)**2).mean()` | `loss = crit(out, y)` |
| `with torch.no_grad(): w -= lr*w.grad` | `optimizer.step()` |
| `w.grad = None` | `optimizer.zero_grad()` |

### 评估要换两样东西

```python
model.eval()                        # ① 切评估模式
with torch.no_grad():               # ② 关掉计算图
    for x, y in test_loader:
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
```

**为什么 `no_grad`**：推理不需要梯度，关掉能省一大半显存和计算。**这是"推理比训练快"的直接原因之一。**

**为什么 `model.eval()`**：BN 层在训练时用 batch 统计量、评估时用滑动平均；Dropout 训练时随机丢、评估时全开。**不切 eval 模式，准确率会莫名其妙低几个点，而且很难查。**

### 准确率怎么算

```python
pred = out.argmax(dim=1)             # (N,) 每个样本分数最高的类别
correct = (pred == y).sum().item()   # 转成 python 整数
acc = correct / len(test_dataset)
```

**注意 `out.argmax(dim=1)`** —— `dim=1` 是"在 10 个类别这一维上比"，day3 学的 `dim` 语义在这里直接用上。

---

## 七、实测基线（跑之前心里有个数）

环境：RTX 3050 Laptop，batch=64，Adam lr=1e-3，MLP 784-128-10。

| epoch | train loss | test acc | 用时 |
|---|---|---|---|
| 1 | 0.3437 | **94.44%** | 9.9s |
| 2 | 0.1573 | 96.26% | 12.0s |
| 3 | 0.1096 | **97.15%** | 11.6s |
| 4 | 0.0841 | 97.35% | 11.4s |
| 5 | 0.0668 | **97.58%** | 10.9s |

**判断标准：**

- **第 1 轮结束 ≥ 93%** → 说明管线通了
- **5 轮结束 ≥ 97%** → 说明训练正常
- **低于 90%** → 有 bug（不是调参问题，先查标签、loss、数据）

**MLP 在 MNIST 上的天花板大约 98%。** 想更高就得上 CNN（day5 之后）。**这里不要纠结调参，能到 97% 就够了。**

---

## 八、五个必踩报错

| 报错 | 原因 | 怎么改 |
|---|---|---|
| `mat1 and mat2 shapes cannot be multiplied (32x784 and 128x784)` | 权重形状搞反 | `nn.Linear(in, out)` 参数顺序是"进→出" |
| `expected scalar type Long but found Float` | 标签不是 long | `y.long()` |
| `Expected input batch_size (32) to match target batch_size (10)` | 输出没 reshape 或模型输出维度错 | 检查最后一层是不是 `nn.Linear(..., 10)` |
| `RuntimeError: DataLoader worker ... exited unexpectedly` | Windows 下 `num_workers>0` | **Windows 一律 `num_workers=0`** |
| loss 不降，准确率卡在 10% | 忘 `zero_grad()`，或忘 `.to(dev)`，或模型最后加了 softmax | 逐项排查 |

**"准确率卡在 10%"这个特别值得记** —— MNIST 有 10 类，随机猜就是 10%。**卡在 10% 意味着模型完全没学到东西，一定是管线问题，不是超参数问题。**

---

## 九、自测题（能口述才算过）

1. `nn.Module` 里 `__init__` 和 `forward` 各自负责什么？为什么 `model(x)` 而不是 `model.forward(x)`？
2. `nn.Linear(784, 128)` 的 weight 形状是什么？为什么不是 `(784,128)`？
3. 手算 `784→128→10` 这个 MLP 一共多少参数？（不许查）
4. 为什么分类不用 MSE？CrossEntropyLoss 内部做了哪两步？
5. 为什么模型最后**不要**加 softmax？
6. 标签为什么要 `.long()`？
7. `optimizer.step()` 和 day3 的 `w -= lr*w.grad` 是什么关系？
8. `zero_grad()` 忘了会怎样？具体症状是什么？
9. 推理时为什么要 `model.eval()` + `torch.no_grad()`？两者分别解决什么？
10. `out.argmax(dim=1)` 里的 `dim=1` 指的是哪一维？
11. 准确率卡在 10% 最可能是什么问题？

---

## 十、收工检查（做完打勾）

- [ ] 脚本能跑完 5 个 epoch，末轮 test acc ≥ 97%
- [ ] 打印出了每层的输出 shape（`(N,1,28,28) → (N,784) → (N,128) → (N,10)`）
- [ ] 手算过参数量，和 `sum(p.numel())` 对上了
- [ ] 画出了 loss 曲线
- [ ] 能用一句话说清"`backward` 和 `step` 的分工"
- [ ] 保存了模型，并成功 load 回来做一次推理
- [ ] 自测题 11 道能口述

---

## 附：今日命令

```powershell
cd D:\cv_study
git add -A
git commit -m "day4: MLP 训练 MNIST"
git push
```
