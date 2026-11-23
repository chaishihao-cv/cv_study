# day4 · MLP 训练 MNIST

> 2026-11-23 · 第 1 周 · 实际用时约 5 小时
> 交付脚本：`stage3_pytorch/day4_mlp_mnist.py`（训练）+ `stage3_pytorch/day4_infer.py`（推理）
> 本笔记全部公式用纯文本写法（VS Code 预览不支持 LaTeX）

---

## 一、今天到底做了什么

从「手写梯度下降拟合一条线」跳到「让神经网络认手写数字」。

| 对比项 | day1~day3 | day4 |
|---|---|---|
| 数据 | 100 个由公式生成的点 | MNIST 60000 张 28×28 图片 |
| 模型 | 两个参数 w、b | 101770 个参数 |
| 求导 | 自己写梯度公式 / autograd 单变量 | autograd 自动跑整个网络 |
| 优化 | 手动 w = w - lr*grad | `optimizer.step()` |
| 评估 | 看 loss 数字 | 看测试集准确率 |

**五轮训练实测（本机 RTX 3050）**

```
epoch1  loss=0.3471  acc=0.9453
epoch2  loss=0.1585  acc=0.9622
epoch3  loss=0.1081  acc=0.9701
epoch4  loss=0.0821  acc=0.9739
epoch5  loss=0.0647  acc=0.9757
```

末轮 **97.57%**，loss 从 0.3471 降到 0.0647（降幅 81.4%）。每轮 9.9~12.0 秒。

**推理结果**：测试集第一张图，真值 7、预测 7、置信度 99.86%。

---

## 二、nn.Module 只要求你做三件事

1. `__init__` 里写 `super().__init__()`，然后**把要学的层当属性挂上**
2. `forward` 里写数据怎么流过去
3. **不要写 backward** —— autograd 从 forward 的执行过程自动建图

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.fc2(self.relu(self.fc1(x)))
```

### 三个关键点

**① `x.flatten(start_dim=1)` 而不是 `x.view(-1, 784)`**

输入是 (64, 1, 28, 28)，`flatten(start_dim=1)` 只压平后三维 → (64, 784)。
**第 0 维是 batch，不能动。** `view(-1, 784)` 也能算对，但写成 -1 属于「靠猜」，数据形状一变就出错。

**② 层只要挂成 `self.xxx`，就自动进 `parameters()`**

```python
sum(p.numel() for p in model.parameters())   # 101770
```

`nn.ReLU()` 没有参数，不贡献数量；`nn.Linear` 的 weight 和 bias 各算一份。

**③ 参数量手算（必须会）**

```text
fc1: 784 * 128 + 128 = 100352 + 128 = 100480
fc2: 128 * 10  + 10  = 1280  + 10  = 1290
合计: 100480 + 1290 = 101770
```

**公式：`输入维度 * 输出维度 + 输出维度`。** 加的那个是 bias，每个输出神经元一个。

### 自己踩的一个小坑

一开始用 `nn.Sequential(...)` 写模型能跑，但后来要打印中间 shape、要加自己的逻辑，Sequential 就不够用了。
**判断标准：只是想顺序堆层 → Sequential；要在 forward 里做 reshape / 分支 / 打印 → 自己写 class。**

---

## 三、CrossEntropyLoss 到底在算什么

`nn.CrossEntropyLoss()` 一步做了三件事：

```text
logits (10 个原始分数)
  → softmax（变成概率，和 = 1）
  → 取正确类别的概率 p
  → -log(p)
```

### 手算验证（和框架结果完全一致）

logits = [2.0, 1.0, 0.1]，target = 0

```text
e^2.0 = 7.389056
e^1.0 = 2.718282
e^0.1 = 1.105171
和    = 11.212509

softmax = [0.659001, 0.242433, 0.098566]
正确类概率 p0 = 0.659001
loss = -log(0.659001) = 0.41703
```

调 `nn.CrossEntropyLoss()(torch.tensor([[2.0,1.0,0.1]]), torch.tensor([0]))`
→ 输出 **0.41703**，一模一样。

### 三个必须知道的事

**① 传进去的必须是 logits，不是 softmax 之后的概率**

CrossEntropyLoss 内部已经做了 log_softmax。你如果先 softmax 再传进去，等于算了两次，loss 会变得很奇怪（而且梯度性质变了）。

**② 标签必须是 int64，shape (N,)**

报错原文：

```text
RuntimeError: expected scalar type Long but found Float
```

正确：`torch.tensor([0, 5, 3])`（dtype=torch.int64）
错误：`torch.tensor([0., 5., 3.])`（float32）

**③ one-hot 传进去居然不报错，值还一样**

```python
criterion(torch.tensor([[2.0,1.0,0.1]]), torch.tensor([[1.,0.,0.]]) )  # 0.41703
criterion(torch.tensor([[2.0,1.0,0.1]]), torch.tensor([0]) )           # 0.41703
```

两次都是 0.41703。**因为数学上等价**，但官方推荐用类别索引（int64），one-hot 会多占内存且容易踩 dtype 坑。

---

## 四、优化器那一步在做什么

```python
optimizer.zero_grad()   # 清空上一批的梯度
loss.backward()         # 算出这一批的梯度，写进 p.grad
optimizer.step()        # 按 lr 更新 p
```

`optimizer.step()` 内部等价于（SGD）：

```text
p = p - lr * p.grad
```

**实测验证**：参数 p=[1, 2]，grad=[2, 4]，lr=0.1

```text
p = [1 - 0.1*2, 2 - 0.1*4] = [0.8, 1.6]
```

跑出来正是 `[0.8, 1.6]`。

### 为什么 `zero_grad` 必须写在 `backward` 前面

这是 day3 踩过的坑的延续：**梯度会累加，不会自动清零。**

如果漏了 `zero_grad()`，第 2 批的梯度 = 第 1 批 + 第 2 批，参数更新方向就错了。
**它不会报错，只会让训练变差 —— 属于最难查的一类 bug。**

注意顺序细节：`zero_grad` 放在 `backward` 之前或之后都行（只要在下一批 `backward` 之前清过），但**行业惯例是写在最前面**，一眼能看出「每批都清」。

### Adam 和 SGD 的区别（今天只用到 Adam）

| | SGD | Adam |
|---|---|---|
| 更新公式 | 只看当前梯度 | 用梯度的一阶矩 + 二阶矩自适应调步长 |
| 需要调 lr | 敏感 | 不敏感，1e-3 就很好 |
| 本任务 | lr 要 0.1 左右才收敛快 | lr=1e-3 五轮就到 97% |

**新任务一律先试 Adam lr=1e-3**，跑通了再考虑换 SGD 追精度。

---

## 五、七步训练循环（背下来）

```python
for epoch in range(epochs):
    model.train()
    total, n = 0.0, 0                      # ← 每轮清零
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(dev), y.to(dev)

        out  = model(x)                    # 1 前向
        loss = criterion(out, y)           # 2 算损失

        optimizer.zero_grad()              # 3 清梯度
        loss.backward()                    # 4 反向
        optimizer.step()                   # 5 更新

        total += loss.item() * x.size(0)   # 6 累计统计
        n     += x.size(0)

    avg = total / n                        # 7 本轮平均
    acc = evaluate()
```

**顺口溜：前向 → 损失 → 清梯度 → 反向 → 更新 → 统计 → 出轮。**

---

## 六、统计量的作用域（今天踩了两个坑，最有价值的一节）

### 坑 1：`total, n` 写在 epoch 循环外面 → 打印出「历史累计平均」

原始写法：

```python
total, n = 0.0, 0          # ← 在外面，只初始化一次
for epoch in range(epochs):
    for batch_idx, (x, y) in enumerate(train_loader):
        ...
        total += loss.item() * x.size(0)
        n += x.size(0)
    avg = total / n
```

跑出来的 5 轮：

```text
epoch1 loss=0.3465
epoch2 loss=0.2510
epoch3 loss=0.2032
epoch4 loss=0.1729
epoch5 loss=0.1513      ← 看起来"降不下去了"
```

**这 5 个数不是本轮 loss，而是"跑到这一轮为止，所有样本的平均"。**

用累计关系反推真实的本轮值：

```text
第1轮真实 = 0.3465
第2轮真实 = 0.2510*2 - 0.3465 = 0.1555
第3轮真实 = 0.2032*3 - 0.5020 = 0.1076
第4轮真实 = 0.1729*4 - 0.6096 = 0.0820
第5轮真实 = 0.1513*5 - 0.6916 = 0.0649
```

**真实值 0.3465 → 0.0649（降 81%），打印值只降 56%。**

**后果**：曲线被历史拖平，会误判成「还远没收敛」。

**改法**：

```python
for epoch in range(epochs):
    total, n = 0.0, 0      # ← 挪到每轮开头
```

改完打印：`0.3471 / 0.1585 / 0.1081 / 0.0821 / 0.0647` —— 和参考基线几乎完全一致。

### 坑 2：`evaluate()` 一轮调了两次

```python
acc_history.append(evaluate())                       # 第 1 遍
print(f'... acc={evaluate():.4f}')                   # 第 2 遍
```

每轮把 10000 张测试图完整跑了两遍，5 轮 = 10 遍，白花约 20 秒。

```python
acc = evaluate()
acc_history.append(acc)
print(f'... acc={acc:.4f}')
```

**通用原则：函数调用提取成变量，尤其是「跑一整个数据集」这种重操作。**

### 作用域总表（以后照着查）

| 量 | 在哪初始化 | 在哪清零 |
|---|---|---|
| `loss_history` / `acc_history` 列表 | 所有轮之外，一次 | 不用清 |
| `total`、`n`（每轮 loss 统计） | **每轮开头** | 每轮开头 |
| `correct`（评估计数） | `evaluate()` 函数内部 | 每次调用自动重置 |
| `batch_idx` | `enumerate` 自动给 | 每轮自动归零 |
| optimizer 的 `grad` | autograd 建 | `zero_grad()` 每批清 |

**口诀：凡是「这一轮才能用」的统计量，必须每轮重新清零。**

忘了清，代码不报错、数字也漂亮，**但它是错的** —— 这是深度学习里最难查的一类 bug。

---

## 七、`avg` 为什么必须移到循环外（实测数据）

把 `avg = total / n` 放循环里，同时打印累计值，实测一轮：

```text
batch   0  本批loss=2.3023  累计avg=2.3023   已看样本=64
batch   1  本批loss=2.2402  累计avg=2.2712   已看样本=128
batch  10  本批loss=1.8246  累计avg=2.0566   已看样本=704
batch 200  本批loss=0.2696  累计avg=0.6624   已看样本=12864
batch 500  本批loss=0.1639  累计avg=0.4477   已看样本=32064
batch 900  本批loss=0.1583  累计avg=0.3499   已看样本=57664
batch 937  本批loss=0.2088  累计avg=0.3438   已看样本=60000

循环结束后 avg = 0.3438
```

**先说结论：不移，数值上不算错。** 循环结束时两者完全相等（都是 0.3438），因为「累计总和 / 累计样本数」走到最后一批自然就是整轮平均。

**那为什么还要移？三个理由：**

**① 循环内的 `avg` 语义不是「整轮平均」，而是「到目前为止的平均」**

`2.30 → 2.27 → 2.05 → 0.66 → 0.44 → 0.34` 这个下降**不是模型在进步，是分母在变大**。
前面只有 64 个样本时，一个坏批就能主导结果；60000 个样本才摊平波动。
看 batch 900 到 937 只从 0.3499 变 0.3438 —— **说明前面 900 批的值全是失真的。**

**② 一旦把 `append` 也留在循环里，图就废了**

```python
for batch_idx, (x, y) in enumerate(train_loader):
    ...
    loss_history.append(avg)      # ❌
```

每轮 append 938 次，5 轮 = **4690 个点**。
画出来横轴是 batch 序号不是 epoch，**5 条锯齿堆叠在一起，看不出「第 1 轮 0.34 → 第 5 轮 0.06」**。
数据没错，图错了 —— 这种错误查起来特别费劲。

**③ 白算 938 次除法**

除法本身不贵，但「在循环里做只该做一次的事」是坏习惯的苗头。以后遇到循环里做 `evaluate()`、`save`、`plot`，就是几十秒和几小时的差别。

**一句话规则：累加器留在里面，结果挪到外面。**

---

## 八、`batch_idx` 怎么来的

```python
for batch_idx, (x, y) in enumerate(train_loader):
```

`enumerate` 给可迭代对象自动编号，**从 0 开始**。

本任务实测：

```text
len(train_loader) = 938        # ceil(60000 / 64)
60000 % 64 = 32                # 最后一批只有 32 张
len(test_loader)  = 40         # ceil(10000 / 256)
```

手算验证编号能对上：

```text
batch_idx=937 时，前面已看的样本 = 937 * 64 = 59968
加上本批 32 张 = 60000   ✅
```

用途：**控制打印频率**。

```python
if batch_idx % 200 == 0:         # 命中 0, 200, 400, 600, 800
    print(...)
```

**注意 `batch_idx` 每轮自动归零**（因为每轮重新 `enumerate`），这点和 `total` 不一样 —— `total` 要你手动清。

---

## 九、`total += loss.item() * x.size(0)` 是什么

`loss` 是**这一批的平均损失**（`reduction="mean"`），不是这一批的总和。

要还原成总和再合并，就得乘回批内样本数：

```text
total += loss.item() * x.size(0)     # 这一批的总损失
n     += x.size(0)                   # 这一批的样本数
avg    = total / n                   # 按样本数加权平均
```

**类比：** 50 人考 90 分 + 10 人考 60 分，全班平均不是 (90+60)/2=75，而是
(50*90 + 10*60) / 60 = 85。

**诚实说明**：本任务每批都是 64 张（最后一批 32），差异只有约 0.0009，实际影响很小。
**但换成 batch 很大或末批很小时（比如 batch=10000，末批只剩 100 张），差异就成真问题了。**

**反例提醒**：把 loss 改成 `reduction="sum"` 会让 loss 直接放大 64 倍，**已经调好的 lr 全乱**，不要这么干。

---

## 十、保存 / 加载 / 单张推理（任务 6）

### 训练脚本里保存

```python
path = 'D:/cv_study/stage3_pytorch/day4_mlp_mnist.pth'
torch.save(model.state_dict(), path)
```

实测文件大小：**409,224 字节**。
101770 个参数 × 4 字节 = 407,080 字节，加文件头正好对上 —— **这个数字可以用来验证权重是否完整。**

**为什么存 `state_dict()` 而不是整个 `model`：**

| | `state_dict()` | 整个 `model` |
|---|---|---|
| 内容 | 只有权重字典 | 整段代码结构（pickle） |
| 体积 | 小 | 大 |
| 换环境 | 安全 | 类定义变了就加载失败 |

### 推理脚本（拆成了独立文件 `day4_infer.py`）

```python
model = MLP().to(dev)
model.load_state_dict(torch.load(path, weights_only=True))
model.eval()

img, label = test_set[0]                    # (1, 28, 28)
with torch.no_grad():
    out  = model(img.unsqueeze(0).to(dev))  # (1, 1, 28, 28)
    pred = torch.argmax(out, dim=1).item()
```

**四个要点：**

**① `model = MLP().to(dev)` 必须先建空模型**

`load_state_dict` 只「往已有的格子里填参数」，不「造格子」。
而且 **类里的层名必须和训练时一模一样**（`fc1`、`fc2`），名字变了报 `Missing key(s)`。

**② `img.unsqueeze(0)` 是今天最容易踩的坑**

`test_set[0]` 返回 `(1, 28, 28)`，模型要 `(N, 1, 28, 28)`。
`unsqueeze(0)` 在最前面插入一维 → `(1, 1, 28, 28)`。
**模型永远吃「带 batch 维」的输入，哪怕只有一张图，也要假装它是一批。**

**③ `model.eval()` + `torch.no_grad()`**

- `eval()`：切换模式。本模型没有 BN / Dropout 所以数值不变，但**这是铁的规矩**，以后加了就必须切。
- `no_grad()`：不建计算图，**省显存、省时间**。推理阶段不需要梯度。

**④ `weights_only=True`**

新版 torch 不加会有 FutureWarning，加上后只允许加载纯张量，更安全。

### 实测结果

```text
真值: 7   预测: 7   置信度: 99.86%
```

### 一个验证技巧：`allclose` 比推理更有价值

```python
with torch.no_grad():
    same = torch.allclose(model(x), model2(x), atol=1e-6)
print('加载后与原模型一致:', same)
```

推理对不对只有「对/错」两种答案；`allclose` 能告诉你**保存-加载链路是否完全无损**。
**注意**：如果训练循环被注释掉了（`model` 变成随机权重），这行会报 `False`，那是误报 —— 所以要拆成两个文件，别靠注释切换。

---

## 十一、今天踩的坑清单

| # | 现象 | 原因 | 修法 |
|---|---|---|---|
| 1 | 打印「训练损失 0.1627」，比预期低一半 | `print` 缩进在 for 里，打的是**最后一批**的 loss | print 退回一轮的层级，用 `avg` |
| 2 | `NameError: name 'total' is not defined`（第 80 行） | 漏了 `total, n, batch_idx` 初始化 | 在 epoch 开头写 `total, n = 0.0, 0` |
| 3 | 准确率算出 **24250%** 这种荒谬值 | `return correct / len(test_loader)` 除的是**批数 40**，不是样本数 10000 | 改成 `len(test_set)` |
| 4 | 5 轮 loss 打印出 `0.3465→0.1513`，曲线太平 | `total, n` 在 epoch 外初始化，变成**累计平均** | 每轮开头清零 |
| 5 | 每轮多花 3~4 秒 | `evaluate()` 调了两次（append 一次、print 一次） | 提取成 `acc = evaluate()` |
| 6 | `torch.load` 弹 FutureWarning | torch 2.4 的安全提示 | 加 `weights_only=True` |

### 教训总结

**第 3 条最危险** —— 它不报错，只给出一个离谱的百分比。**看到准确率 > 100% 就是分母错了。**
**第 4 条最隐蔽** —— 数字很「合理」，只是语义错了，不看反推根本发现不了。

**统一自查方法：跑完先手算一遍。** 938 批 × 平均 64 张 ≈ 60000、平均 loss 应在 0.3 左右、准确率应在 10%~99% 之间。**对不上就说明某个分母/分子错了。**

---

## 十二、自测题（能口述才算过）

1. `nn.Linear(784, 128)` 有几个参数？为什么是 100480 不是 100352？
2. `x.flatten(start_dim=1)` 里 `start_dim=1` 为什么不写 0？
3. CrossEntropyLoss 内部做了哪三步？为什么不能先 softmax 再传？
4. 标签传 float 会报什么错？one-hot 为什么传得进去？
5. 七步训练循环的七步分别是？`zero_grad` 漏了会怎样？
6. `total += loss.item() * x.size(0)` 为什么要乘样本数？不乘会怎样？
7. `avg` 为什么必须移到循环外？不移数值上错吗？真正的后果是什么？
8. `total` 为什么每轮要清零而 `batch_idx` 不用？
9. `load_state_dict` 之前为什么必须先 `MLP().to(dev)`？
10. `unsqueeze(0)` 把 `(1,28,28)` 变成什么？为什么必须补这一维？
11. 推理时为什么不需要 `loss.backward()`？`no_grad()` 省了什么？

### 我自己答出的关键三题

**第 1 题（参数量）**：784*128 + 128 = 100480，加的那个 128 是 bias。

**第 2 题（flatten）**：0 是 batch 维，压平它就把 64 张图混成一张了。

**第 7 题（avg 位置）**：数值上不移不算错（末值相等），但语义变成「到当前为止的平均」，前面全失真；而且 append 留里面会让图变成 4690 个点的锯齿。

---

## 十三、明日（day5）

主题：**卷积基础** —— 卷积核手算、输出尺寸公式、参数量对比 MLP。

脚本：`stage3_pytorch/day5_conv_calc.py`

今天预留的三个数（明天要用）：

```text
卷积输出尺寸 = (输入 + 2*padding - kernel) / stride + 1
MLP 第一层:  784 * 128 = 100480 个参数
同样感受野的 3x3 卷积(64通道): 3*3*1*64 + 64 = 640 个参数
```

**从 100480 到 640，这就是 CNN 存在的理由。**

---

## 附：今日命令

```powershell
# 训练
Remove-Item Env:PYTHONNOUSERSITE -ErrorAction SilentlyContinue
& "D:\Anaconda\anaconda3\python.exe" "D:\cv_study\stage3_pytorch\day4_mlp_mnist.py"

# 推理（秒级，不用等训练）
& "D:\Anaconda\anaconda3\python.exe" "D:\cv_study\stage3_pytorch\day4_infer.py"

# 提交
& "C:\Program Files\Git\cmd\git.exe" -C "D:\cv_study" add stage3_pytorch/day4_mlp_mnist.py stage3_pytorch/day4_infer.py notes/day4.md
& "C:\Program Files\Git\cmd\git.exe" -C "D:\cv_study" commit -m "day4: MLP 训练 MNIST + 独立推理脚本 + 笔记"
& "C:\Program Files\Git\cmd\git.exe" -C "D:\cv_study" push
```

**注意**：`.pth` 文件已被 `.gitignore` 排除，不会进仓库（权重文件不进 git 是行规）。MNIST 数据同样排除。
