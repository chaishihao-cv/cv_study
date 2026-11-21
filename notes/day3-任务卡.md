# day3 任务卡 · Tensor 与 autograd

> 日期：2026-11-21　｜　阶段3：深度学习基础 + PyTorch
> 讲义：`D:\清美教育课件\计算机视觉3阶段学习\3阶段\day3-详细讲义.md`
> 脚本：`D:\cv_study\stage3_pytorch\day3_tensor_autograd.py`
> 解释器：**必须是 `D:\Anaconda\anaconda3\python.exe`**（先跑 `D:\cv_study\check_env.py` 确认）

---

## 开始前必做（30 秒）

在 VS Code 里打开 `D:\cv_study\check_env.py` 运行一次，确认最后一行是：

```
✅ 环境正常，可以开始练习
```

**如果不是（比如显示 `D:\python 软件\python.exe` 或 torch ❌），先切换解释器，否则今天的代码一行都跑不了。**

---

## 任务 1 · 张量基本操作（热身，15 分钟）

在 `day3_tensor_autograd.py` 里写：

1. 创建三个张量：`torch.arange(12)`、`torch.zeros(3,4)`、`torch.randn(2,3)`
2. 分别打印它们的 `shape`、`dtype`、`device`、`requires_grad`
3. 把 `arange(12)` 变形为 `(3,4)`，再变形成 `(4,3)`，再展平成 `(12,)`
4. 用 `-1` 写一次：`reshape(3, -1)`

**验收**：能说出 `torch.Size([3, 4])` 里的两个数字分别代表什么。

---

## 任务 2 · 广播：先手算，再验证（20 分钟）

**先在纸上算出结果形状，再写代码验证。** 顺序不能反。

| 表达式 | 你算出形状 | 代码结果 |
|---|---|---|
| `(3,4) + (4,)` | ？ | ？ |
| `(3,1) + (1,4)` | ？ | ？ |
| `(3,4) + (3,)` | ？ | ❌ 报错 |
| `(3,4) + (3,1)` | ？ | ？ |

第三行会报错，**把报错信息完整抄下来**（讲义第 2 节⑥有对照）。

然后想清楚：`(3,4)` 要和"每一行的第 i 个元素"相加（即每个元素加同一个常量），应该用哪个形状？

**验收**：能解释为什么 `(3,4)+(3,)` 不行而 `(3,4)+(3,1)` 行。

---

## 任务 3 · requires_grad 观察实验（20 分钟）

按这个顺序做，每一步都打印出来看：

```python
w = torch.tensor(2.0, requires_grad=True)
u = w * 3
loss = u ** 2

print("w.grad_fn       :", w.grad_fn)          # 期望 None（叶子）
print("u.grad_fn       :", u.grad_fn)          # 期望 MulBackward0
print("loss.grad_fn    :", loss.grad_fn)       # 期望 PowBackward0
print("w 是叶子吗     :", w.is_leaf)
```

然后：

1. `loss.backward()`，打印 `w.grad`，**手算核对**：
   - `loss = (3w)^2 = 9w^2`，`dloss/dw = 18w`，w=2 → **应该是 36**
2. 打印 `u.grad`，看它是 `None` 并观察那条 `UserWarning`

**验收**：能解释"为什么 `w.grad` 有值而 `u.grad` 是 None"。

---

## 任务 4 · 梯度累加实验（今天最重要，20 分钟）

**这段代码要完整跑一遍，并记录输出：**

```python
w = torch.tensor(0.0, requires_grad=True)
x, y_true = 2.0, 3.0

for i in range(3):
    pred = w * x
    loss = (pred - y_true) ** 2
    loss.backward()
    print(f"第{i+1}次  w.grad = {w.grad.item():.4f}")
```

**预期输出（讲义第 3 节④亲测）：**

```
第1次  w.grad = -12.0000
第2次  w.grad = -24.0000
第3次  w.grad = -36.0000
```

**然后回答问题（写在脚本注释里）：**

1. 正确答案应该永远是 -12，为什么变成了 -24、-36？
2. 如果每轮都用这个放大后的梯度更新参数，等价于学习率变大了几倍？
3. 在循环里加一句 `w.grad = None`（放在哪里？），输出会变成什么？

**验收**：能用自己的话说清"为什么必须清零"，并且说得出"不清零 = 学习率翻倍"。

---

## 任务 5 · 用 autograd 复现 day1 的梯度下降（30 分钟）

把 day1 的训练循环改写成长义第 4 节那段代码，要求：

- [ ] 数据：`torch.linspace(-1,1,100)`、真值 `y=2x+3`、噪声 `0.3*randn`
- [ ] 参数 `w`、`b` 都要 `requires_grad=True`
- [ ] 训练 200 轮，`lr=0.1`
- [ ] 记录 `loss.item()`
- [ ] **验证 `w.grad` 和你手算的 `mean(2*(pred-y)*x)` 一致**（打印两个值，差应该是 0 或极小）
- [ ] 画 loss 曲线（用 day2 学的 `plot`）

**验收**：

| 检查项 | 期望 |
|---|---|
| 初始 loss | 与 day1 同量级（约 10 左右） |
| 最终 loss | 降到 0.1 以下 |
| 最终 w | 接近 2 |
| 最终 b | 接近 3 |
| grad 核对 | 手算与 torch 之差 ≈ 0 |

> **数字不会和 day1 完全相同**（torch 和 numpy 的随机数生成器不同），看结构不要看数值。

---

## 任务 6 · 思考题（不写代码，5 分钟口述）

1. `no_grad()` 应该用在训练的哪一步？为什么？
2. 记录 loss 时为什么必须 `.item()`？不写会怎样？
3. `detach()` 和 `no_grad()` 有什么区别？
4. 如果模型在 GPU 上但数据在 CPU 上，会报什么错？怎么修？

---

## 自测题（8 道，能口述才算过）

1. Tensor 比 numpy 多了哪三个能力？
2. `[1,2,3]` 和 `[1.,2.,3.]` 有什么区别？哪个能求导？
3. `requires_grad=True` 让 torch 做了什么？
4. 叶子张量和非叶子张量，哪个有 `.grad`？
5. `backward()` 为什么只能对单个数调用？怎么处理向量输出？
6. 不清零 `.grad`，跑 100 轮后学习率相当于放大了多少倍？
7. `view` 和 `reshape` 的区别？
8. `(3,4)+(3,)` 为什么报错？

---

## 收工

- [ ] 五个任务都跑通
- [ ] 自测题能口述
- [ ] 更新 day3 笔记（我帮你写）
- [ ] git 提交：

```powershell
cd D:\cv_study
git add -A
git commit -m "day3: Tensor 操作 + autograd 梯度累加实验 + torch 复现梯度下降"
git push
```
