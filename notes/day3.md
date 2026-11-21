# day3 · Tensor 与 autograd

> 日期：2026-11-21　｜　阶段3：深度学习基础 + PyTorch
> 代码：`D:\cv_study\stage3_pytorch\day3.py`
> 用时：约 3 小时（含解释器环境排查）

---

## 一、今天到底做了什么

一条主线：**把 day1 手推梯度的活儿，交给 torch 自动算，然后证明它算的和手推的一模一样。**

1. 认识 `torch.tensor`——它和 numpy 数组差在哪儿（`dtype`、`device`、`requires_grad`）
2. 把张量的形状操作过一遍（reshape / unsqueeze / permute / cat / stack / 索引）
3. 搞懂 autograd 的三样东西：`requires_grad`、`grad_fn`、`backward`
4. **用 torch 复现 day1 的梯度下降**，200 轮跑到底
5. **梯度核对**：手推公式算出的梯度和 `w.grad` 对比 → **差值 0.00e+00**

**一句话结论：`loss.backward()` 就是你 day1 手写的那两行求导公式，只不过它是按链式法则一层层自动走出来的。**

---

## 二、torch.tensor 是什么

**一句话：numpy 数组 + 三个额外能力。**

| 能力 | 对应属性 | 有什么用 |
|---|---|---|
| 能上 GPU | `.device` / `.to("cuda")` | 同样的矩阵乘，GPU 快 10~100 倍 |
| 能自动求导 | `.requires_grad` / `.grad_fn` | 不用手推公式 |
| 能做自动微分图 | `.backward()` | 反向传播 |

### dtype 决定了能不能求导

```python
a = torch.tensor([1, 2, 3])     # torch.int64，整数不能求导
b = torch.tensor([1., 2., 3.])  # torch.float32，可以求导
```

**`[1,2,3]` 和 `[1.,2.,3.]` 差一个点，类型就从 int64 变成 float32。** 整数没法做导数（导数要小数），所以 `requires_grad=True` 只对浮点张量有效。

### 两个容易混的写法

| 写法 | 结果 |
|---|---|
| `torch.tensor([1,2,3])` | 保留原类型 → `int64` |
| `torch.tensor([1.,2.,3.])` | `float32` |
| `torch.Tensor([1,2,3])`（**大写 T**） | **强制转成 float32** |

大写 `torch.Tensor` 是老式的构造函数，不管传什么都变 float32。**新代码统一用小写 `torch.tensor`，行为更可控。**

### 标量张量

```python
torch.tensor(3.5)   # shape 是 torch.Size([])，空的
```

**shape 为空（`[]`）和 shape 是 `[1]` 不是一回事。** 空的才叫标量，才能 `.item()`。

### tensor 和 numpy 的两种关系（关键）

| 写法 | 是否共享内存 |
|---|---|
| `torch.tensor(np_array)` | **复制**，改 numpy 不影响 tensor |
| `torch.from_numpy(np_array)` | **共享**，改 numpy 后 tensor 跟着变 |

实测：用 `from_numpy` 建好后，把 numpy 里的值改成 99.0，tensor 读出来也是 99.0。**这个特性很危险也很方便**——省内存，但会踩"数据被别处偷偷改了"的坑。

---

## 三、形状操作（今天过了一遍）

| 操作 | 代码 | 作用 |
|---|---|---|
| 变形 | `x.reshape(3, 4)` | 改形状，元素总数不变 |
| 加一维 | `x.unsqueeze(0)` | `(3,)` → `(1,3)`，加在最前面 |
| 换轴序 | `x.permute(1,2,0)` | `(3,224,224)` → `(224,224,3)`，图像 HWC↔CHW 靠它 |
| 拼接 | `torch.cat([a,b], dim=0)` | 在已有维度上接，`(2,3)+(2,3)` → `(4,3)` |
| 堆叠 | `torch.stack([a,b], dim=0)` | 新建一个维度，`(2,3)+(2,3)` → `(2,2,3)` |

**`cat` 和 `stack` 的区别就一句**：`cat` 是"接长"，`stack` 是"多套一层"。

### 索引四种（前三种 day1 numpy 就学过，第四种是新增）

```python
x[0]          # 第 0 行
x[:, 0]       # 第 0 列
x[1:3, :2]    # 切片
x[x > 0]      # 布尔索引：把所有正数挑出来，形状会变
x[[0,2,3]]    # 花式索引：一次挑指定几行
```

---

## 四、广播规则（从右往左比）

**规则一句话：从最右边的维度开始，两个维度要么相等，要么其中一个是 1，否则报错。**

| 形状组合 | 结果 | 为什么 |
|---|---|---|
| `(3,4) + (4,)` | `(3,4)` | `(4,)` 自动补成 `(1,4)` 再铺成 `(3,4)` ✅ |
| `(3,1) + (1,4)` | `(3,4)` | 各自在缺的那一维上铺开 ✅ |
| `(3,4) + (3,)` | **报错** | 右边对齐是 `4` 对 `3`，既不相等也没有 1 ❌ |

报错原文（记住长什么样）：

```
The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 1
```

**翻译：在第 1 维上，a 是 4、b 是 3，既不等也没有 1，无法广播。**

**将来最常见的一次广播**：图像 `(3,224,224)` 减均值 `(3,1,1)`，就是靠这条规则自动铺开。

> 任务卡的任务 2（广播手算）还没做，待补。补做方式：先在注释里手推形状，再跑代码验证，顺序不能反。

---

## 五、autograd 的三个开关

### 1. requires_grad：谁来求导

```python
w = torch.tensor(0.0, requires_grad=True)
```

**只有标了 `requires_grad=True` 的张量，才会被记录进计算图、才会有 `.grad`。** 数据（x、y）标 False，参数（w、b）标 True——这是铁律。

### 2. grad_fn：这张量是怎么算出来的

```python
w = torch.tensor(0.0, requires_grad=True)
y = w * 3 + 1
loss = y ** 2
print(loss.grad_fn)   # <PowBackward0 object at ...>
```

**`grad_fn` 就是"我是被哪个运算造出来的"，它连着上一个节点。** 顺藤摸瓜能扒出整条链：

```
loss  PowBackward0
  ↑
y     AddBackward0
  ↑
      MulBackward0
  ↑
w     AccumulateGrad   ← 链条终点，梯度最后落到这里
```

**这条链就是"计算图"。** `backward()` 就是沿这条链从后往前走一遍。

### 3. 叶子 / 非叶子（重要）

| 概念 | 定义 | `.grad` 会不会被填 |
|---|---|---|
| 叶子（leaf） | 用户自己建的张量，`is_leaf=True` | ✅ 会 |
| 非叶子 | 运算算出来的中间结果 | ❌ 不会，默认是 None |

实测：

```python
w = torch.tensor(2.0, requires_grad=True)
u = w * 3
u.backward()
print(w.grad)   # tensor(3.)   ← 叶子有
print(u.grad)   # None         ← 非叶子没有，还带一条 UserWarning
```

**中间结果的梯度默认被丢掉了（省内存）。** 想看的话加一句 `u.retain_grad()`：

```python
u = w * 3
u.retain_grad()     # 强制保留
u.backward()
print(u.grad)       # tensor(1.)
```

**这个技巧以后诊断"第 3 层梯度是不是传丢了"非常有用。**

### 4. 两个"断图"的方法

| 方法 | 效果 |
|---|---|
| `.detach()` | 返回一个不记图的副本，`grad_fn` 变 None |
| `with torch.no_grad():` | 这个块里的所有运算都不记图 |

**断了图之后再 backward 会报错：**

```
element 0 of tensors does not require grad and does not have a grad_fn
```

---

## 六、backward 与链式法则（今天的核心）

### 1. 我自己的一句话总结（已验证正确）

> **backward 就是反向传播的过程，遵循链式法则。**

补三点让它更完整：

1. **`grad_fn` 链是物理载体**——链式法则不是抽象概念，它真的是一串对象连起来的
2. **每个节点只管自己的局部导数**——Pow 只知道 `d(y^2)/dy = 2y`，Mul 只知道 `d(3w)/dw = 3`，谁也不知道全局
3. **多条路径时梯度相加**——这是最容易漏的一点（见下）

### 2. 多路径必须相加

```python
w = torch.tensor(2.0, requires_grad=True)
z = w * w + w * 3
z.backward()
print(w.grad)   # tensor(7.)
```

手算：`z = w^2 + 3w`，`dz/dw = 2w + 3 = 4 + 3 = 7` ✅

**注意 `w*w` 里的 w 出现了两次**，两条路径上的梯度 `2w` 和 `3` 是**相加**的。框架自动做了这件事，但心里要有数。

### 3. 梯度会累加，不会自动清零（今天踩过的坑）

```python
w = torch.tensor(0.0, requires_grad=True)
x, y_true = 2.0, 3.0
for i in range(3):
    pred = w * x
    loss = (pred - y_true) ** 2
    loss.backward()          # 注意：没有清零
    print(f"第{i+1}次 backward 后，w.grad = {w.grad.item()}")
```

实测输出：

```
第1次：-12.0000
第2次：-24.0000
第3次：-36.0000
```

**`backward()` 是 `+=`，不是 `=`。** 每轮都在上一次的基础上继续加。

**这就是为什么训练循环里必须清零**，三种等价写法：

```python
w.grad = None                          # 最快，置空
# w.grad.zero_()                       # 原地清零
# optimizer.zero_grad()                # 以后用优化器时用这个
```

### 4. 同一张图不能 backward 两次

```python
loss.backward()
loss.backward()   # 报错
```

报错原文：

```
Trying to backward through the graph a second time ... the buffers have already been freed
```

**原因：backward 走完一遍，中间结果就被释放了（省内存）。** 所以训练循环里每轮都要重新前向一遍，图才会重新建。

---

## 七、`.item()` 是干什么的

**把单元素张量变成 Python 的 float。**

```python
loss = tensor(0.0866)
loss.item()    # 0.0866，纯 float
float(loss)    # 等价
```

**为什么必须转？** 因为 `loss` 是挂在计算图上的张量，如果直接把它 append 进列表，**整张计算图会被一起留住**，跑几百轮内存就爆了。

```python
loss_history.append(loss.item())   # ✅ 只存数字
loss_history.append(loss)          # ❌ 存了整个图
```

### 两个报错要知道

| 情况 | 报错 |
|---|---|
| 多元素张量 `.item()` | `a Tensor with 3 elements cannot be converted to Scalar` |
| 顺序搞反（先 item 再 backward） | `'float' object has no attribute 'backward'` |

**顺序必须是：`loss.backward()` → 之后才 `loss.item()`。**

---

## 八、用 torch 复现 day1（实测结果）

```python
torch.manual_seed(42)
x = torch.linspace(-1, 1, 100)
y = 2 * x + 3 + 0.3 * torch.randn(100)

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
lr = 0.1

for epoch in range(200):
    pred = w * x + b
    loss = ((pred - y) ** 2).mean()
    loss_history.append(loss.item())
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad = None
    b.grad = None
```

### 三行代码替代了 day1 的整套手推

| day1 怎么干 | day3 怎么干 |
|---|---|
| 手写 `dw = 2*mean((pred-y)*x)` | `loss.backward()` |
| 手写 `db = 2*mean(pred-y)` | `loss.backward()` |
| `w = w - lr*dw` | `w -= lr*w.grad` |

### 实测数字

| 项 | 结果 |
|---|---|
| 初始 loss | **10.5428** |
| 最终 loss（200 轮） | **0.086661** |
| 最终 w | **1.9912**（真值 2.0） |
| 最终 b | **3.0179**（真值 3.0） |

**loss 0.0866 ≈ 噪声方差 `0.3^2 = 0.09`** —— 正好贴到"地板"，再降不下去了，因为剩下的误差全是噪声，不是模型能学的部分。

---

## 九、梯度核对（今天最有价值的一步）

### 1. 怎么核对

```python
if epoch == 0:
    with torch.no_grad():
        grad_w_hand = (2 * (pred - y) * x).mean().item()
        grad_b_hand = (2 * (pred - y)).mean().item()
    print(f"【核对】w: autograd={w.grad.item():.8f}  手算={grad_w_hand:.8f}")
```

### 2. 实测结果

| 参数 | autograd 算的 | 手推公式算的 | 差值 |
|---|---|---|---|
| w.grad | **-1.35425520** | **-1.35425520** | **0.00e+00** |
| b.grad | **-6.03586006** | **-6.03585958** | **4.77e-07** |

**w 完全相等，b 差在小数点后第 7 位。**

**这证明：`backward()` 算的就是你 day1 手推的那条公式，一点没走样。**

### 3. 为什么 b 的差值不是 0

**4.77e-07 不是错，是 float32 的精度极限。**

- float32 只有约 7 位有效十进制数字
- autograd 内部做 `mean` 的除法顺序，和手算的乘法/加法顺序不完全一样
- 浮点运算不满足结合律 → 末位会差一点

**判据：差值 ≤ 1e-6 就算对上。**

### 4. 我自己踩的坑：漏了 `.mean()`

第一版我写的是：

```python
grad_w_hand = 2 * x * (w * x + b - y)
```

**结果是 shape `[100]` 的向量，不是 1 个数，每打印一行 1144 个字符，200 轮直接刷屏。**

**错在哪：**

- `loss = ((pred-y)**2).mean()` 里的 `.mean()` 已经把 100 个样本**求和再除以 100**了
- 反向传到 x 时，那 100 个样本各自的梯度也要**加起来再除以 100**
- 我算的是"每个样本各自的梯度"，漏了最后"求和平均"

**更直观的验证**：把那 100 个数加起来除以 100，正好等于 autograd 的 -1.35425520。

**记住这条：loss 是标量，梯度也一定是标量。算出来是向量，说明漏了 reduce。**

---

## 十、今天踩的坑

| 坑 | 原因 / 解决 |
|---|---|
| 手推梯度算出 100 个数 | 漏 `.mean()`，见第九节 |
| 输出 221 行刷屏 | 每轮都 `print` → 改成 `if epoch % 10 == 0 or epoch == 199` |
| `{0}` 打出来永远不变 | `f"第{0}次"` 里 `{0}` 是字面常量，不是变量 |
| `u.grad = None` + UserWarning | 非叶子张量不保留梯度，要 `u.retain_grad()` |
| `backward` 第二次报错 | 图已释放，必须重新前向 |
| 梯度越跑越大 | 忘记 `w.grad = None`，`backward` 是累加 |
| `loss.item()` 报错 | 多元素张量不能转标量 |
| 更新时忘记 `no_grad` | 计算图会滚雪球，必须包起来 |
| 在 3 附近的 b 完全不动 | float32 在这个量级的最小增量约 `2.4e-7`，更新量比它小就"看起来没变" |

### 输出行数从 221 降到 25

**算对了但看不见关键信息，等于没算。** 打印节奏：每 10 轮一行 + 核对只打一次 + 末尾汇总一次。

---

## 十一、自测题（能口述才算过）

1. `torch.tensor([1,2,3])` 和 `torch.tensor([1.,2.,3.])` 有什么区别？哪个能求导？
2. `torch.tensor(np)` 和 `torch.from_numpy(np)` 差在哪？
3. `(3,4)` 加 `(4,)` 能得到什么形状？`(3,4)` 加 `(3,)` 为什么报错？
4. `grad_fn` 是干什么的？顺着它能扒出什么？
5. 叶子和非叶子的区别？为什么 `u.grad` 是 None？
6. `backward` 为什么是链式法则？每个节点知道自己要算全局导数吗？
7. `w*w + w*3` 对 w 求导是多少？为什么两条路径要相加？
8. `backward()` 连调两次为什么报错？
9. 训练循环里为什么必须 `w.grad = None`？
10. `.item()` 不加会有什么后果？
11. 手推梯度和 `w.grad` 对比，差值多大算通过？为什么 b 不是精确 0？

### 我自己答错的一题（已纠正）

| 题 | 我的答案 | 正确说法 |
|---|---|---|
| 梯度核对 | 写成 `2*x*(pred-y)`，出来 100 个数 | 必须 `.mean()`：loss 用了 mean，梯度也要 mean；**loss 是标量，梯度也必须是标量** |

---

## 十二、明日（day4）

按 13 周计划继续：**nn.Module 与优化器**——把今天的 `w -= lr*w.grad` 换成 `optimizer.step()`，把参数装进 `nn.Module`，开始写"像样的"模型代码。

---

## 附：今日命令

```powershell
cd D:\cv_study
git add -A
git commit -m "day3: Tensor 与 autograd + 梯度核对"
git push
```
