# PyTorch 函数速查表

> 建立于 2026-11-21（day3）
> 说明：所有 `→` 右边都是**实测输出**（`D:\Anaconda\anaconda3\python.exe`，torch 2.4.1+cu118，RTX 3050）
> 用法：先看目录找类，再查表。**不要背，会用就行。**

---

## 目录

- [0. 三秒自查法（最重要）](#0-三秒自查法最重要)
- [1. 创建张量](#1-创建张量)
- [2. 查询属性](#2-查询属性)
- [3. 形状变换](#3-形状变换)
- [4. 拼接与切分](#4-拼接与切分)
- [5. 索引与切片](#5-索引与切片)
- [6. 广播](#6-广播)
- [7. 数学运算](#7-数学运算)
- [8. 统计归约（dim 是关键）](#8-统计归约dim-是关键)
- [9. 类型转换](#9-类型转换)
- [10. 设备与 CPU/GPU](#10-设备与-cpugpu)
- [11. 与 numpy / Python 互转](#11-与-numpy--python-互转)
- [12. autograd（自动求导）](#12-autograd自动求导)
- [13. 常用报错对照表](#13-常用报错对照表)
- [14. 还没学、但迟早要用的](#14-还没学但迟早要用的)

---

## 0. 三秒自查法（最重要）

**任何张量出问题，先打这一行：**

```python
print(t.shape, t.dtype, t.device)
```

**95% 的 torch 报错，答案都在这三个东西里。** 你这两天所有 bug 都出在 shape 或 dtype。

---

## 1. 创建张量

### 1.1 从数据创建

| 写法 | shape | dtype |
|---|---|---|
| `torch.tensor([1,2,3])` | `(3,)` | **int64** |
| `torch.tensor([1.,2.,3.])` | `(3,)` | **float32** |
| `torch.tensor(3.5)` | `()` 标量 | float32 |
| `torch.tensor([[1,2],[3,4]])` | `(2,2)` | int64 |
| `torch.Tensor([1,2,3])`（**大写**） | `(3,)` | float32（强转） |

**记住：`[1,2,3]` → int64 不能求导；`[1.,2.,3.]` → float32 能求导。一个点的差别。**
大写 `torch.Tensor` 是老式写法，**别用**。

### 1.2 常量填充

| 写法 | shape | dtype |
|---|---|---|
| `torch.zeros(2,3)` | `(2,3)` | float32 |
| `torch.ones(2,3)` | `(2,3)` | float32 |
| `torch.full((2,3), 7)` | `(2,3)` | **int64** ⚠️ |
| `torch.full((2,3), 7.0)` | `(2,3)` | float32 ✅ |
| `torch.empty(2,3)` | `(2,3)` | float32（**值是垃圾**） |
| `torch.eye(3)` | `(3,3)` | float32 单位矩阵 |
| `torch.zeros_like(x)` | 同 x | 同 x |
| `torch.ones_like(x)` | 同 x | 同 x |

**`full` 的 dtype 跟着填充值走**：填 `7` 得 int64，填 `7.0` 得 float32。这是新手第一坑。

**`empty` 不会清零**，只是分配内存，速度快但值不可信。除了明确要"先占位再全赋值"，一律用 `zeros`。

### 1.3 序列

| 写法 | 结果 | dtype | 含终点？ |
|---|---|---|---|
| `torch.arange(12)` | 0..11，12 个 | int64 | ❌ |
| `torch.arange(0, 1, 0.1)` | 0..0.9，10 个 | float32 | ❌ |
| `torch.linspace(-1, 1, 5)` | `[-1,-0.5,0,0.5,1]` | float32 | ✅ |

**两个核心区别：**

1. **`arange` 不含终点，`linspace` 含终点**
2. **浮点步长用 `arange` 会累积误差**——实测打出来是 `0.10000000149011612`、`0.20000000298023224`
   → **要精确等分一律用 `linspace(起点, 终点, 个数)`**

### 1.4 随机

| 写法 | 分布 | dtype |
|---|---|---|
| `torch.rand(2,3)` | 均匀 `[0,1)` | float32 |
| `torch.randn(2,3)` | 标准正态（均值0、标准差1） | float32 |
| `torch.randint(0, 10, (2,3))` | 整数均匀 `[0,10)` | int64 |
| `torch.normal(0.0, 1.0, (2,3))` | 自定义均值/标准差 | float32 |
| `torch.randperm(5)` | 0~4 随机打乱 | int64 |

**`rand` 不会出负数，`randn` 会。造噪声必须用 `randn`。**

**可复现：前面加 `torch.manual_seed(42)`。**

### 1.5 三个通用参数（所有创建函数都支持）

```python
torch.tensor([1,2], dtype=torch.float64)     # 指定类型
torch.zeros(2,3, device="cuda")              # 直接建在显存里
torch.tensor([1.,2.], requires_grad=True)    # 建好就能求导
```

**`requires_grad=True` 只能给浮点张量。**

---

## 2. 查询属性

| 属性/方法 | 例子 → 输出 |
|---|---|
| `t.shape` / `t.size()` | `torch.Size([2, 3])` |
| `t.dtype` | `torch.float32` |
| `t.device` | `cpu` / `cuda:0` |
| `t.dim()` | 维度个数 → `2` |
| `t.numel()` | 元素总数 → `6` |
| `t.is_leaf` | 是不是叶子节点 |
| `t.requires_grad` | 要不要算梯度 |
| `t.grad_fn` | 被哪个运算造出来的 |
| `t.is_contiguous()` | 内存是否连续（影响能不能 `view`） |

**`shape` 是属性（不加括号），`size()` 是方法（加括号），两个都行。**

---

## 3. 形状变换

### 3.1 改形状

| 写法 | `(12,)` 的结果 |
|---|---|
| `x.reshape(3,4)` | `(3,4)` |
| `x.view(3,4)` | `(3,4)` |
| `x.reshape(-1, 4)` | `(3,4)`，**-1 表示自动算** |
| `x.flatten()` | `(12,)`，拉平 |
| `x.reshape(4,2)` | **报错**：`shape '[4, 2]' is invalid for input of size 6` |

**`reshape` 和 `view` 的区别（唯一要记的）：**

- `view` **要求内存连续**，不连续就报错
- `reshape` **不要求**，不连续时会自动复制一份

实测：

```python
d = torch.randn(2,3).permute(1,0)   # permute 之后不连续
d.view(6)     # ❌ view size is not compatible with input tensor's size and stride
d.reshape(6)  # ✅ 可以
```

**结论：日常一律用 `reshape`，省心。**

### 3.2 加减维度

| 写法 | 原 `(3,5)` 的结果 | 用途 |
|---|---|---|
| `a.unsqueeze(0)` | `(1,3,5)` | 加在最前（造 batch 维） |
| `a.unsqueeze(1)` | `(3,1,5)` | 加在中间 |
| `a.unsqueeze(-1)` | `(3,5,1)` | 加在最后 |
| `b.squeeze()` | 去掉**所有**大小为 1 的维 | |
| `b.squeeze(0)` | 只去第 0 维 | |

实测：`(1,3,1).squeeze()` → `(3,)`；`(1,3,1).squeeze(0)` → `(3,1)`

**`unsqueeze` 和 `squeeze` 是一对：加 1 维、去 1 维。** 单张图片 `(3,224,224)` 变成 batch `(1,3,224,224)` 靠 `unsqueeze(0)`。

### 3.3 换轴序

| 写法 | 原 `(3,5)` / `(2,3,4)` | 说明 |
|---|---|---|
| `a.T` | `(5,3)` | 转置，**只适用于 ≤2 维** |
| `a.transpose(0,1)` | `(5,3)` | 交换两个指定维 |
| `c.permute(2,0,1)` | `(4,2,3)` | 多维任意换序 |

**图像 HWC↔CHW 就是这个：**

```python
img_hwc = torch.randn(224, 224, 3)
img_chw = img_hwc.permute(2, 0, 1)   # → (3, 224, 224)
```

**`permute` 之后张量不再连续**，紧接着要 `view` 会报错，用 `reshape` 或者先 `.contiguous()`。

---

## 4. 拼接与切分

### 4.1 拼接

`e1`、`e2` 都是 `(2,3)`：

| 写法 | 结果 | 区别 |
|---|---|---|
| `torch.cat([e1,e2], dim=0)` | `(4,3)` | **接长**（在已有维上接） |
| `torch.cat([e1,e2], dim=1)` | `(2,6)` | 横向接长 |
| `torch.stack([e1,e2], dim=0)` | `(2,2,3)` | **多套一层**（新建维度） |
| `torch.stack([e1,e2], dim=1)` | `(2,2,3)` | 插在第 1 维 |

**`cat` vs `stack` 一句话：**

- `cat` = **接长**，维度数不变，被接的那一维变大
- `stack` = **多套一层**，维度数 +1

**记忆方法：`stack` 就是"堆一叠"**，三张 `(3,224,224)` 的图片 stack 起来就是 `(3,3,224,224)`。

### 4.2 切分

`f` 是 `(6,4)`：

| 写法 | 结果 |
|---|---|
| `torch.split(f, 2, dim=0)` | 3 段，每段 `(2,4)` |
| `torch.chunk(f, 3, dim=0)` | 3 段，每段 `(2,4)` |

**`split` 按"每段多大"分，`chunk` 按"分几段"分。**

### 4.3 复制

| 写法 | 说明 |
|---|---|
| `x.clone()` | **真复制**（实测 `is` 比较为 False），新内存 |
| `x.detach()` | 共享内存，但**断开计算图** |
| `x.to(device)` | 搬设备，可能复制 |

**`y = x` 只是起个别名，改 y 会改 x。要独立就 `clone()`。** 这是 Python 通用坑，可能出问题，在列表 append 时尤其如此。

---

## 5. 索引与切片

`g = torch.arange(12).reshape(3,4).float()`：

```python
g[0]        → [0, 1, 2, 3]              第 0 行
g[:, 0]     → [0, 4, 8]                 第 0 列
g[1:3, :2]  → [[4,5],[8,9]]             切片
g[g > 5]    → 6 个数                     布尔索引，形状会变
g[[0, 2]]   → [[0,1,2,3],[8,9,10,11]]   花式索引，挑指定几行
```

| 类型 | 写法 | 特点 |
|---|---|---|
| 整数索引 | `g[0]` | **降维**：`(3,4)` → `(4,)` |
| 切片 | `g[0:1]` | **保维**：`(3,4)` → `(1,4)` |
| 布尔索引 | `g[g>5]` | 形状会变，返回一维 |
| 花式索引 | `g[[0,2]]` | 按索引列表挑 |

**降维和保维的区别非常容易踩**：想让 `g[0]` 保持 `(1,4)`，就写 `g[0:1]` 或 `g[0].unsqueeze(0)`。

**赋值：**

```python
g[0, 0] = 99          # 单个位置
g[g > 5] = 0          # 条件赋值，mask 的经典用法
```

**越界报错原文：**

```
IndexError: index 5 is out of bounds for dimension 0 with size 3
```

---

## 6. 广播

### 6.1 规则（三步）

1. **右对齐**，缺的维度在左边补 1
2. 逐维比：**相等 → 过；一个是 1 → 铺开；都不是 1 且不等 → 报错**
3. 结果 shape 取每维较大的那个

**记忆：`1` 是万能牌，`2` 不能变成 `3`。**

### 6.2 实测三例

`a` 是 `(2,3)` = `[[1,2,3],[4,5,6]]`：

| 表达式 | 展开过程 | 结果 |
|---|---|---|
| `a + torch.tensor([10,20,30])` | `b` 复印成 2 行 | `[[11,22,33],[14,25,36]]` |
| `a + torch.tensor([[100],[200]])` | `c` 横向复印 3 份 | `[[101,102,103],[204,205,206]]` |
| `a + 5` | 标量复印 6 份 | `[[6,7,8],[9,10,11]]` |

用 `expand` 能**看见复印件**：

```python
torch.tensor([10,20,30]).expand(2,3)   → [[10,20,30],[10,20,30]]
torch.tensor([[100],[200]]).expand(2,3) → [[100,100,100],[200,200,200]]
```

### 6.3 为什么会报错

`(2,3) + (2,)` 报错：

```
The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1
```

**因为右对齐后，`(2,)` 的 2 对上的是最后一维 3，不是第一维 2。**
要加 `(2,)` 必须手动 `reshape(2, 1)`。

**图像归一化的经典写法：**

```python
img  = torch.randn(3, 224, 224)
mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)   # 必须是 (3,1,1)
img - mean   → (3,224,224)   # 每通道减自己的均值
```

**写成 `(3,)` 就对不上最后一维 224，直接报错。新手必踩。**

---

## 7. 数学运算

### 7.1 逐元素运算

```python
a + b        a - b        a * b        a / b        a ** 2
torch.add(a, b)   torch.sub(a, b)   torch.mul(a, b)
```

**注意 `*` 是逐元素乘，不是矩阵乘！**

### 7.2 矩阵乘（三种写法都是同一个东西）

```python
a @ b                 # 最简洁，推荐
torch.matmul(a, b)    # 通用
torch.mm(a, b)        # 只用于 2 维
torch.bmm(a, b)       # 只用于 3 维 batch
```

实测 `h(2,3) @ ones(3,2)` → `[[6,6],[15,15]]`

**形状必须满足 `(m,n) @ (n,p)`。报错原文：**

```
mat1 and mat2 shapes cannot be multiplied (2x3 and 2x3)
```

**训练里最常见的一次矩阵乘**：`x(100,1) @ w(1,1)` 或者全连接层 `x(batch, in) @ W(in, out)`。

### 7.3 常用函数

| 函数 | 作用 |
|---|---|
| `torch.abs(x)` / `x.abs()` | 绝对值 |
| `torch.exp(x)` | e 的幂 |
| `torch.log(x)` | 自然对数 |
| `torch.sqrt(x)` | 开方 |
| `torch.pow(x, 2)` | 幂 |
| `torch.clamp(x, 2, 5)` | 截断到 `[2,5]`，实测 `[[1,2,3],[4,5,6]]` → `[[2,2,3],[4,5,5]]` |
| `torch.relu(x)` / `x.relu()` | ReLU |
| `torch.sigmoid(x)` | sigmoid |
| `torch.tanh(x)` | tanh |
| `torch.softmax(x, dim=1)` | softmax |

**`clamp` 就是梯度裁剪的基础，以后会天天见。**

### 7.4 原地操作（带下划线）

```python
x.add_(1)      # 等价于 x = x + 1，但直接改 x
x.mul_(2)
x.zero_()
```

**带 `_` 结尾的都是原地改。对 `requires_grad=True` 的叶子张量用原地操作会报错**（框架不允许你偷偷改参数），实际训练里更新参数要包在 `with torch.no_grad():` 里。

### 7.5 类型提升（唯一例外的"宽容"）

实测：`torch.zeros(3) + torch.zeros(3, dtype=torch.int64)` → **不报错**，自动提升成 float32。

**但矩阵乘不宽容：**

```
RuntimeError: expected m1 and m2 to have the same dtype, but got: __int64 != float
```

**所以：加减小事会自动提升，矩阵乘必须 dtype 一致。**

---

## 8. 统计归约（dim 是关键）

`h = [[1,2,3],[4,5,6]]`：

| 写法 | 输出 | 说明 |
|---|---|---|
| `h.sum()` | `21.0` | 全部加起来，返回标量 |
| `h.sum(dim=0)` | `[5,7,9]` | **沿行方向压**，得到每列的和 |
| `h.sum(dim=1)` | `[6,15]` | **沿列方向压**，得到每行的和 |
| `h.mean(dim=0)` | `[2.5,3.5,4.5]` | 每列的均值 |
| `h.max()` | `6.0` | 全局最大值 |
| `h.max(dim=1)` | `values=[3,6], indices=[2,2]` | **返回两个东西** |
| `h.argmax(dim=1)` | `[2,2]` | 只返回位置 |
| `h.min(dim=0)` | 同 `max` | |

### `dim` 到底是什么意思（最容易糊的地方）

**`dim=k` 的意思是"把第 k 维压掉"。**

```
h 是 (2,3)：
dim=0 → 压掉第 0 维（2 那一维）→ 结果 (3,)   ← 跨行求，得到每列一个值
dim=1 → 压掉第 1 维（3 那一维）→ 结果 (2,)   ← 跨列求，得到每行一个值
```

**口语记忆：`dim=0` 竖着压（每列一个数），`dim=1` 横着压（每行一个数）。**

### `keepdim=True` 为什么要用

```python
h.mean(dim=1)                 → (2,)      一维
h.mean(dim=1, keepdim=True)   → (2,1)     保留被压的那一维
```

**为了继续广播！** 例如 `x(2,3) - h.mean(dim=1)` 会报错（`(2,3)` 对 `(2,)`），而 `x - h.mean(dim=1, keepdim=True)` 正好 `(2,3)` 对 `(2,1)`，能广播。

**规则：归约完还要参与运算的，一律加 `keepdim=True`。**

### 三合一：argmax

```python
pred.argmax(dim=1)   # 分类任务里，从 logits 取预测类别
```

**这是分类任务最常用的一行代码。**

---

## 9. 类型转换

| 写法 | 例子结果 |
|---|---|
| `x.float()` | → float32 |
| `x.double()` | → float64 |
| `x.int()` | → int32 |
| `x.long()` | → int64（**索引必须用这个**） |
| `x.bool()` | → 布尔 |
| `x.to(torch.float64)` | 通用写法 |
| `x.type(torch.float32)` | 老写法，别用 |

实测 `torch.tensor([1.7, -2.3]).long()` → `[1, -2]`（**向零截断，不是四舍五入**）

**要四舍五入用 `torch.round(x).long()`。**

**索引、`CrossEntropyLoss` 的标签、`embedding` 的下标，都必须是 `long`（int64）。** 传 float 进去会报类型错。

---

## 10. 设备与 CPU/GPU

| 写法 | 说明 |
|---|---|
| `torch.cuda.is_available()` | 实测 **True**（你的 3050 可用） |
| `x.device` | `cpu` / `cuda:0` |
| `x.to("cuda")` | 搬到 GPU，实测 `.device` → `cuda:0` |
| `x.cuda()` | 同上，简写 |
| `x.cpu()` | 搬回 CPU |
| `x.to("cpu", dtype=torch.float64)` | 设备+类型一次搞定 |

**铁律：参与运算的张量必须在同一设备。** CPU 张量和 GPU 张量相加会报错。

**查看显存：**

```python
torch.cuda.memory_allocated() / 1024**2   # 当前占用 MB
```

**你的 3050 只有 4GB 显存**，以后跑 YOLO 要小心 batch size，先从 `batch=4` 起。

---

## 11. 与 numpy / Python 互转

| 写法 | 关系 |
|---|---|
| `torch.tensor(np_arr)` | **复制**（改 np 不影响，实测仍 1.0） |
| `torch.from_numpy(np_arr)` | **共享内存**（改 np 跟着变，实测变 99.0） |
| `x.numpy()` | 转回 numpy |
| `x.tolist()` | 转成 Python 列表 |
| `x.item()` | 转成 Python 标量（**只能是单元素**） |

**`x.numpy()` 的两个限制：**

1. 有梯度的张量不能直接转：
   ```
   RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
   ```
   → 正确写法 `x.detach().cpu().numpy()`（**这行以后会写几百遍**）
2. 在 GPU 上的张量要先 `.cpu()`

**多元素 `.item()` 报错：**

```
a Tensor with 3 elements cannot be converted to Scalar
```

---

## 12. autograd（自动求导）

### 12.1 三要素

| 概念 | 代码 | 说明 |
|---|---|---|
| 谁要求导 | `requires_grad=True` | 只对浮点张量有效 |
| 怎么算出来的 | `t.grad_fn` | 计算图节点，如 `<PowBackward0>` |
| 触发反传 | `loss.backward()` | 从 loss 往回走一遍 |

**整数张量设了会报错：**

```
RuntimeError: Only Tensors of floating point and complex dtype can require gradients
```

### 12.2 叶子 vs 非叶子

| 类型 | 定义 | `.grad` |
|---|---|---|
| 叶子 | 自己建的张量 | ✅ 会填 |
| 非叶子 | 运算算出来的中间结果 | ❌ None |

**要保留中间结果梯度：`u.retain_grad()`**（实测 `u.grad = 1.0`）。

### 12.3 四条必守规则

```python
# 1. 每轮必须清零（backward 是累加）
w.grad = None                  # 最快
# w.grad.zero_()               # 等价
# optimizer.zero_grad()        # 以后用优化器时用这个

# 2. 更新参数时不能记图
with torch.no_grad():
    w -= lr * w.grad

# 3. 同一张图不能 backward 两次
#    Trying to backward through the graph a second time ...

# 4. 存 loss 必须先 item()
loss_history.append(loss.item())
```

**关于第 1 条，实测梯度累加：**

```
第1次 backward 后：w.grad = -12.0000
第2次：-24.0000
第3次：-36.0000      ← 是 += 不是 =
```

### 12.4 多路径梯度相加

```python
w = torch.tensor(2.0, requires_grad=True)
z = w * w + w * 3
z.backward()
print(w.grad)   # tensor(7.)  = 2*w + 3
```

**`w` 出现了两次，两条路径的梯度加起来了。** 手算 `dz/dw = 2w + 3 = 7` ✅

### 12.5 断图的两种方式

| 写法 | 效果 |
|---|---|
| `x.detach()` | 返回断图副本 |
| `with torch.no_grad():` | 块内所有运算都不记图 |

**断了之后 backward 报错：**

```
element 0 of tensors does not require grad and does not have a grad_fn
```

### 12.6 梯度核对的固定模板（day3 验证过的）

```python
if epoch == 0:
    with torch.no_grad():
        grad_w_hand = (2 * (pred - y) * x).mean().item()
    print(f"autograd={w.grad.item():.8f}  手算={grad_w_hand:.8f}")
```

实测差值：w **0.00e+00**、b **4.77e-07**（float32 精度极限，**≤1e-6 就算通过**）。

**判据：loss 是标量，梯度也必须是标量。算出来是向量，说明漏了 reduce（`.mean()` / `.sum()`）。**

---

## 13. 常用报错对照表

| 报错原文（关键部分） | 原因 | 怎么改 |
|---|---|---|
| `mat1 and mat2 shapes cannot be multiplied (2x3 and 2x3)` | 矩阵乘形状不满足 `(m,n)@(n,p)` | 打印 shape，检查内维是否相等 |
| `expected m1 and m2 to have the same dtype, but got: __int64 != float` | 矩阵乘两边 dtype 不一致 | `x.float()` |
| `The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 1` | 广播对不上 | 右对齐检查，或 `unsqueeze`/`reshape` |
| `shape '[4, 2]' is invalid for input of size 6` | reshape 元素数不匹配 | 元素总数要能整除 |
| `view size is not compatible with input tensor's size and stride` | permute 之后不连续 | 用 `reshape` 或 `.contiguous()` |
| `Only Tensors of floating point and complex dtype can require gradients` | 整数张量设了 requires_grad | 加小数点或用 `.float()` |
| `element 0 of tensors does not require grad and does not have a grad_fn` | 在 detach/no_grad 的张量上 backward | 检查有没有不小心 detach |
| `Trying to backward through the graph a second time` | 同一张图 backward 两次 | 重新前向一遍 |
| `a Tensor with 3 elements cannot be converted to Scalar` | 多元素张量 `.item()` | 用 `.tolist()` 或先 `.sum()` |
| `Can't call numpy() on Tensor that requires grad` | 有梯度转 numpy | `x.detach().cpu().numpy()` |
| `IndexError: index 5 is out of bounds for dimension 0 with size 3` | 索引越界 | 检查 size |

**遇到不认识的报错，先做两件事：** ① `print(shape, dtype)` ② 直接搜报错原文（九成有人踩过）。

---

## 14. 还没学、但迟早要用的

先知道名字，不用现在学：

| 类别 | 常用 | 什么时候 |
|---|---|---|
| 模块 | `nn.Module`、`nn.Linear`、`nn.Conv2d` | day4~day5 |
| 参数 | `nn.Parameter`（就是带标签的可求导张量） | day4 |
| 优化器 | `torch.optim.SGD`、`Adam`、`optimizer.step()` | day4 |
| 损失 | `nn.MSELoss`、`nn.CrossEntropyLoss` | day4 |
| 数据集 | `Dataset`、`DataLoader` | day5 |
| 保存加载 | `torch.save`、`torch.load`、`state_dict` | day6 |
| 预训练 | `torchvision.models`、`torch.hub` | day8 之后 |
| 部署 | `torch.onnx.export`、`torch.jit.trace` | 第 10~11 周（**你的主攻方向**） |

**部署工程岗面试高频就俩：`torch.onnx.export` 和 `torch.jit.trace`。** 记在心上，后面专门练。

---

## 附：一张最常用的"模板代码"

以后每天写脚本都从这段开始改：

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 数据
x = torch.linspace(-1, 1, 100).to(device)
y = 2 * x + 3 + 0.3 * torch.randn(100, device=device)

# 参数
w = torch.tensor(0.0, requires_grad=True, device=device)
b = torch.tensor(0.0, requires_grad=True, device=device)
lr = 0.1

loss_history = []
for epoch in range(200):
    pred = w * x + b                    # 前向（广播）
    loss = ((pred - y) ** 2).mean()

    loss_history.append(loss.item())    # 先 item
    loss.backward()                     # 再反传

    with torch.no_grad():               # 更新不记图
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad = None                       # 清零
    b.grad = None

    if epoch % 10 == 0 or epoch == 199:
        print(f"epoch {epoch:3d}  loss={loss.item():.6f}  w={w.item():.4f}  b={b.item():.4f}")
```

**这段顺序记住：前向 → 算 loss → item → backward → no_grad 更新 → 清零。** day1 手推的、day3 torch 版的，都是这六步。以后换成 `optimizer` 只是把中间两行换掉，骨架不变。
