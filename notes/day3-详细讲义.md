# day3 · Tensor 与 autograd（详细讲义）

> 日期：2026-11-21　｜　阶段3：深度学习基础 + PyTorch
> 对应脚本：`D:\cv_study\stage3_pytorch\day3_tensor_autograd.py`
> 任务卡：`D:\cv_study\notes\day3-任务卡.md`

---

## 第 0 节 · 今天要跨过的那道坎

前两天的梯度都是**你自己推出来的**：

- day1：手推 `dL/dw = mean(2*(pred-y)*x)`，然后手写 `w = w - lr*grad`
- day2：手算梯度连乘 10 次，看它衰减到 1e-7

**真实的网络有几十层、几百万个参数，手推是不可能的。** 所以要交给框架：

```
你只管写"前向怎么算" → torch 自动算出所有梯度
```

**今天就是把"手推梯度"换成"自动求导"，并且要能说清 torch 到底做了什么。**

---

## 第 1 节 · Tensor 是什么

一句话：**Tensor 就是 NumPy 数组 + 三个额外能力。**

| 能力 | NumPy 数组 | Tensor |
|---|---|---|
| 多维数值计算 | ✅ | ✅ |
| 放 GPU 上算 | ❌ | ✅ |
| **自动求导** | ❌ | ✅ |
| **存梯度** | ❌ | ✅（`.grad`） |

所以 Tensor 不是新东西，**你已有的 numpy 知识 90% 能直接用**：

```python
import torch
t = torch.arange(12).reshape(3, 4)   # 和 numpy 几乎一样
print(t.shape)      # torch.Size([3, 4])
print(t.dtype)      # torch.int64
```

### 四个必须记住的属性

| 属性 | 含义 | 为什么重要 |
|---|---|---|
| `.shape` | 形状，`torch.Size([3,4])` | 90% 的报错是形状不匹配 |
| `.dtype` | 数据类型（float32 / int64） | **模型参数必须是浮点**，整数不能求导 |
| `.device` | 在 CPU 还是 GPU 上 | 模型和数据**必须在同一个 device** |
| `.requires_grad` | 要不要对它求导 | 决定它参不参与计算图 |

**测试代码（可以现在就跑）：**

```python
import torch
a = torch.tensor([1, 2, 3])            # int64，不能求导
b = torch.tensor([1., 2., 3.])         # float32，可以求导
c = torch.tensor([1., 2., 3.], requires_grad=True)

print(a.dtype, a.requires_grad)        # torch.int64 False
print(b.dtype, b.requires_grad)        # torch.float32 False
print(c.dtype, c.requires_grad)        # torch.float32 True
print(a.device)                        # cpu
```

> **注意那个小数点**：`[1,2,3]` 是整数，`[1.,2.,3.]` 是浮点。**整数张量不能求导**，这是新手第一个坑。

---

## 第 2 节 · 常用操作（只学这七类够用）

### ① 变形：`reshape` / `view`

```python
x = torch.arange(12)
x.reshape(3, 4)      # 变成 3×4
x.view(3, 4)         # 效果一样
```

**区别（面试可能问）：**

| | `view` | `reshape` |
|---|---|---|
| 要求内存连续 | **必须连续**，否则报错 | 不连续会自动复制一份 |
| 速度 | 更快（只是换个"看法"） | 可能慢一点 |
| 建议 | 拿不准就用 `reshape` | **新手统一用 `reshape`** |

**`-1` 的用法**（自动推断那一维）：

```python
x.reshape(3, -1)     # 12 个数，3 行，列数自动算 = 4
x.reshape(-1,)       # 展平成 1 维
```

### ② 增删维度：`unsqueeze` / `squeeze`

```python
x = torch.randn(3, 4)      # shape (3,4)
x.unsqueeze(0).shape       # (1,3,4)  在位置 0 加一维
x.unsqueeze(1).shape       # (3,1,4)  在位置 1 加一维
y = torch.randn(1, 3, 4)
y.squeeze(0).shape         # (3,4)    去掉大小为 1 的维度
```

**什么时候用**：单张图是 `(3,224,224)`，模型要 `(N,3,224,224)`，就得 `unsqueeze(0)`。**这个操作在推理时天天用。**

### ③ 换轴：`transpose` / `permute`

```python
x = torch.randn(3, 224, 224)      # (C,H,W) —— PyTorch 图像默认格式
x.permute(1, 2, 0).shape          # (224,224,3) —— 变成 (H,W,C)，画图用
```

**`transpose` 只能换两个轴，`permute` 可以换任意多个。**

> 记住这个方向：**PyTorch 是 (C,H,W)，OpenCV / matplotlib 是 (H,W,C)**。你在 day17 画检测框时一定会遇到这个转换。

### ④ 拼接：`cat` / `stack`

```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)
torch.cat([a, b], dim=0).shape      # (4,3)  沿着已有维度接
torch.stack([a, b], dim=0).shape    # (2,2,3) 新建一个维度
```

| | `cat` | `stack` |
|---|---|---|
| 做什么 | 沿现有维度**接起来** | **新建**一维再叠 |
| `(2,3)+(2,3)` | `(4,3)` | `(2,2,3)` |

### ⑤ 索引与切片（和 numpy 一模一样）

```python
x = torch.randn(4, 5)
x[0]          # 第 0 行，shape (5,)
x[:, 0]       # 第 0 列，shape (4,)
x[1:3, :2]    # 第 1~2 行、前 2 列
x[x > 0]      # 布尔索引，把正数挑出来
```

**还有一个 numpy 没有的**：

```python
x[[0, 2]]     # 取第 0 和第 2 行（花式索引）
```

### ⑥ 广播（broadcasting）——最容易出错的一节

规则：**从最后一维往左比，能对上（相等或其中一个是 1）就能算，否则报错。**

**手算三个例子：**

```
例1:  (3,4) + (4,)      → (3,4)     ✅  (4) 自动补成 (1,4) 再复制 3 份
例2:  (3,1) + (1,4)     → (3,4)     ✅  各自复制到对方的大小
例3:  (3,4) + (3,)      → ❌ 报错   最后一维 4 vs 3 不相等，也不是 1
```

**为什么例3 会报错**：广播是**从右往左**比的。`(3,4)` 的最后一维是 4，`(3,)` 的最后一维是 3，对不上。

**修正办法**：把第二个改成 `(3,1)`

```python
x.unsqueeze(1)   # (3,) → (3,1)，然后 (3,4)+(3,1) 就能算 → (3,4)
```

**记住一句话：报错 `The size of tensor a (4) must match the size of tensor b (3)`，就是在骂你最后一维对不上。**

### ⑦ 类型与设备转换

```python
x.float()                # 转 float32（模型最爱）
x.to(torch.float32)      # 同一个效果，写法更通用
x.cuda()                 # 搬到 GPU
x.to("cuda")             # 同上
x.cpu()                  # 搬回 CPU（打印前要搬回来）
```

**GPU 张量不能直接 `print` 得很舒服，也不能和 CPU 张量运算：**

```python
a = torch.randn(3).cuda()
b = torch.randn(3)              # 在 CPU
a + b                           # ❌ RuntimeError: Expected all tensors to be on the same device
```

**这是新手第二大坑：`Expected all tensors to be on the same device`。**

---

## 第 3 节 · autograd：梯度是怎么自动算出来的

### 1. 三个开关

```python
w = torch.tensor(0.0, requires_grad=True)   # 打开：我要对它求导
```

**`requires_grad=True` 到底做了什么：**

> 它告诉 torch：**"这个张量参与的每一步运算，都记下来"**，记下来的东西叫**计算图**。

```python
w = torch.tensor(2.0, requires_grad=True)
y = w * 3 + 1          # 记录：Mul → Add
loss = y ** 2          # 记录：Pow
print(loss.grad_fn)    # <PowBackward0 object>   ← 说明图记住了
```

**`grad_fn` 就是"这个张量是怎么算出来的"。** 一路往回看 `grad_fn.next_functions`，就能走到最开始的 `w`——这就是**链式法则的载体**。

### 2. 叶子张量（leaf）与非叶子

| 类型 | 怎么产生 | `.grad` | `.grad_fn` |
|---|---|---|---|
| **叶子**（leaf） | 你自己创建的、`requires_grad=True` 的张量 | ✅ 有值 | `None` |
| **非叶子** | 由运算产生的中间结果 | ❌ **是 None** | 有值 |

```python
w = torch.tensor(2.0, requires_grad=True)   # 叶子
u = w * 3                                   # 非叶子
u.backward()
print(w.grad)    # tensor(3.)   ← 有
print(u.grad)    # None（还会警告 "UserWarning: The .grad attribute of a Tensor ... is being accessed"）
```

**为什么中间结果不存梯度**：几百万个中间变量都存梯度会爆内存。**torch 默认只保留"你要优化的那些参数"的梯度。**

### 3. backward() 做了什么

```python
w = torch.tensor(2.0, requires_grad=True)
loss = 3 * w ** 2          # 手算：dloss/dw = 6w = 12

loss.backward()            # 沿计算图反向走一遍，把梯度写到所有叶子上
print(w.grad)              # tensor(12.)
```

**`backward()` 默认只能对"标量"调用**（一个数）。对一个向量调用会报错：

```python
v = torch.randn(3, requires_grad=True)
out = v * 2
out.backward()    # ❌ RuntimeError: grad can be implicitly created only for scalar outputs
```

**因为"梯度"本身要先说清"对哪个输出求"**。多个输出时，要么 `out.sum().backward()`（先聚合），要么给 `backward(gradient=...)` 指定权重。

> **实操结论：`backward()` 前先 `loss = loss.mean()` 或 `loss.sum()`，让它是标量。** 你的 day1 里 `np.mean` 干的就是这个事。

### 4. `.grad` 是累加的（今天最重要的一节）

**这是新手第三大坑，也是面试常问。**

```python
w = torch.tensor(0.0, requires_grad=True)
x, y_true = 2.0, 3.0

for i in range(3):
    pred = w * x
    loss = (pred - y_true) ** 2
    loss.backward()
    print(f"第{i+1}次 backward 后 w.grad = {w.grad.item()}")
```

**输出（亲测）：**

```
第1次 backward 后 w.grad = -12.0
第2次 backward 后 w.grad = -24.0
第3次 backward 后 w.grad = -36.0
```

**为什么翻倍？**

手算：`pred = w*x = 0`，`loss = (0-3)^2 = 9`
`dloss/dw = 2*(pred-y)*x = 2*(-3)*2 = -12`

**但 `.grad` 没有清零，第三次是从 -12 往上加 -12 = -24，再加 -12 = -36。** 而正确答案永远是 -12。

**后果**：参数更新变成 `w = w - lr*(-36)`，等于**学习率被放大了 3 倍**。跑 100 轮就变成放大 100 倍——**直接训飞**。

### 5. 所以每个训练循环里必须有 `zero_grad()`

```python
for x, y in dataloader:
    pred = model(x)
    loss = ((pred - y) ** 2).mean()
    optimizer.zero_grad()      # ← 必须先清零
    loss.backward()            #    再反向
    optimizer.step()           #    再更新
```

**注意顺序**：`zero_grad()` 放在 `backward()` **之前**就行，放在循环开头最不容易错。

> 为什么很多人写在 `backward()` 后面？因为那样也行——**它清的是"上一次留下的"垃圾，不是这次的**。但写在前面（循环开头）更符合直觉，也更安全。

### 6. `detach()` 和 `no_grad()`

**两个"关掉自动求导"的工具，用途不同：**

| 工具 | 作用 | 用在哪 |
|---|---|---|
| `x.detach()` | 返回一个新张量，**从计算图上摘下来** | 想把张量存起来做统计/画图时 |
| `torch.no_grad()` | 这一段代码**不记图** | **推理、评估、验证集** |

```python
# 推理时（重要）
with torch.no_grad():
    pred = model(x)

# 或者更彻底
model.eval()
with torch.no_grad():
    ...
```

**为什么推理必须关**：

1. 不记图 → **省显存**（大模型能差 30%~50%）
2. 不记图 → **速度快 10%~30%**
3. 忘记关 → 显存慢慢涨，最后 `CUDA out of memory`

**还有一条：`.item()` 也能"取出来当普通数"，这在记录 loss 时必须用**

```python
loss_history.append(loss.item())    # ✅ 只存一个 float
loss_history.append(loss)           # ❌ 把整个计算图挂进列表，显存爆炸
```

**这就是 day1 笔记里提到的那个坑，今天正式解释原因了。**

---

## 第 4 节 · 用 autograd 复现 day1 的梯度下降

day1 你手写了这段：

```python
grad_w = np.mean(2 * (pred - y) * x)
grad_b = np.mean(2 * (pred - y))
```

**用 torch 就变成了：**

```python
import torch

torch.manual_seed(42)

# 造数据（和 day1 一样：真值 w=2, b=3，加噪声）
x = torch.linspace(-1, 1, 100)
y = 2 * x + 3 + 0.3 * torch.randn(100)

# 参数：唯一区别就是多了 requires_grad=True
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

lr = 0.1
loss_history = []

for epoch in range(200):
    pred = w * x + b
    loss = ((pred - y) ** 2).mean()

    loss_history.append(loss.item())    # 注意 .item()

    # 下面三行替代了原来的"手推梯度 + 手写更新"
    loss.backward()                      # 自动算出 w.grad 和 b.grad
    with torch.no_grad():                # 更新时不能记图
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad = None                        # 清零（等价于 zero_grad）
    b.grad = None
```

**三个必须理解的点：**

**① `loss.backward()` 替代了你的手推公式**

`w.grad` 里存的值，应该和手算的 `mean(2*(pred-y)*x)` **完全相等**。这就是任务卡里要你验的第一件事。

**② 更新参数必须在 `torch.no_grad()` 里**

否则 `w -= lr*w.grad` 这一步也会被记进计算图，图越滚越大。用 `optimizer.step()` 时它内部已经帮你关了，手写更新就得自己包。

**③ 梯度清零的两种写法**

```python
w.grad = None          # 置空（更快，推荐）
w.grad.zero_()         # 填 0
```

**用 optimizer 时就是一行 `optimizer.zero_grad()`。**

### 数值会和 day1 不一样，这是正常的

**torch 的随机数生成器和 numpy 不一样**，所以同样的 seed 也不会得到同一组噪声。

| 对比项 | day1（numpy） | day3（torch） |
|---|---|---|
| 数据 | `np.random.seed(42)` | `torch.manual_seed(42)` |
| 初始 loss | 10.2759 | 会不同（大约同一量级） |
| 最终 w / b | 2.021 / 2.969 | 接近但不相同 |

**不要追求数字相等，要看"结构是否一致"：loss 从 10 左右降到 0.1 以下、w 收敛到 2 附近、b 收敛到 3 附近。**

---

## 第 5 节 · 五个必然踩到的报错

### ① `RuntimeError: a leaf Variable that requires grad is being used in an in-place operation`

```python
w = torch.tensor(1.0, requires_grad=True)
w += 3        # ❌ 直接原地改
```

**原因**：torch 需要"改之前的值"才能算梯度，你原地改了它就没有旧值了。

**解决**：

```python
with torch.no_grad():
    w -= lr * w.grad          # ✅ 包在 no_grad 里
```

### ② `.grad` 是 `None`

```python
y = x * 2
y.backward()
print(x.grad)      # None
```

**三个常见原因：**

| 原因 | 检查 |
|---|---|
| `x.requires_grad` 没设成 True | `print(x.requires_grad)` |
| 对非叶子张量求梯度 | 只有叶子有 `.grad` |
| 张量是整数类型 | `x.dtype` 是 int 就求不了导 |

### ③ `grad can be implicitly created only for scalar outputs`

**原因**：对向量 `backward()` 了。**解决**：`out.sum().backward()` 或 `loss.mean().backward()`。

### ④ `Expected all tensors to be on the same device`

**原因**：一个张量在 GPU，一个在 CPU。**解决**：统一 `.to(device)`，模型和数据用同一个 device 变量。

### ⑤ `UserWarning: The .grad attribute of a Tensor that is not a leaf ... is being accessed`

**原因**：你访问了中间结果的 `.grad`（一定是 None）。**不是错误，是警告**，改成访问叶子张量即可。

---

## 第 6 节 · 自测题（能口述才算过）

1. Tensor 和 numpy 数组的三个区别是什么？
2. `torch.tensor([1,2,3])` 和 `torch.tensor([1.,2.,3.])` 有什么不同？哪个能求导？
3. `required_grad=True` 到底让 torch 做了什么？
4. 什么是叶子张量？为什么中间结果的 `.grad` 是 None？
5. `backward()` 为什么默认只能对单个数调用？
6. `.grad` 为什么会累加？不清零的后果是什么（用数字说）？
7. `detach()` 和 `no_grad()` 分别用在哪？
8. 为什么记录 loss 必须写 `.item()`？
9. `(3,4) + (3,)` 为什么报错？怎么改？
10. 用 torch 训练时，`loss.backward()` 替代了 day1 里哪两行代码？

---

## 第 7 节 · 今天的产出（收工前检查）

- [ ] `day3_tensor_autograd.py` 里五个任务都跑通
- [ ] 亲眼见过"不清零导致梯度翻倍"的实验（任务 4）
- [ ] 用 torch 跑通 day1 那个训练循环，loss 曲线画出来
- [ ] 能手算一次 `w.grad` 并和 torch 的结果对得上

**明天（day4）会做 MLP + MNIST**，今天这套东西（Tensor 操作、backward、zero_grad、no_grad）是明天的全部前置。
