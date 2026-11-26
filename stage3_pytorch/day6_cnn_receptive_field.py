# day6 感受野计算与实测
# 思路：堆 L 层卷积 → 取输出中心那一个点往回求导 → 梯度非零的范围就是感受野
# 思考题：池化层有没有可学习参数？感受野代表什么？

import torch
import torch.nn as nn

TOL = 0.0            # 感受野内的梯度理论上必不为 0，所以阈值取 0 最稳


def build_stack(L, k=3, s=1, p=1):
    """造 L 个相同的卷积层，串成一条流水线"""
    return nn.Sequential(*[nn.Conv2d(1, 1, k, stride=s, padding=p, bias=False)
                           for _ in range(L)])


def measure_rf(net, H=64, verbose=True):
    """梯度法实测感受野：返回高度方向的跨度（整数）"""
    x = torch.randn(1, 1, H, H, requires_grad=True)
    out = net(x)

    r = c = out.shape[-1] // 2          # 输出正中心，避开边界截断

    # 空1：那一张量上做反向
    out[0, 0, r, c].backward()

    # 空2：取输入梯度，取绝对值，大于阈值的就算"被算到过"
    # 注意：不要用 1e-12 这类固定阈值。层数一多，梯度按指数衰减，
    # 最外圈可能比中心小好几个数量级；用固定阈值会把最外圈误判成 0，
    # 实测就会少 1 像素（比如 20 层算成 40 而不是 41）。
    mask = x.grad[0, 0].abs() > TOL

    idx = mask.nonzero()                # 非零位置，形状 (N, 2)：[行, 列]
    h_lo, h_hi = idx[:, 0].min().item(), idx[:, 0].max().item()
    w_lo, w_hi = idx[:, 1].min().item(), idx[:, 1].max().item()

    # 空3：包含两端，所以要 +1
    rf_h = h_hi - h_lo + 1
    rf_w = w_hi - w_lo + 1

    # 输出边长是偶数时，out.shape[-1] // 2 并不是真正的几何中心，
    # 会向左或向右偏一格，导致两个方向测出的跨度差 2。取较大值才是真实感受野。
    rf = max(rf_h, rf_w)

    if verbose:
        print(f"    非零像素 {mask.sum().item():4d}/{H * H}"
              f" | 包围盒 {rf_h}x{rf_w}"
              f" | 行范围 [{h_lo:2d}, {h_hi:2d}]"
              f" | 输入 {H}")

    if rf_h >= H - 1:                   # 贴着边界，说明结果已被截断
        print(f"    [!] 输入只有 {H}，感受野已接近边界，实测值可能被截断")

    return rf


print("=" * 66)
print("任务 1 核对：手算 vs 实测")
print("=" * 66)

conv3 = lambda: nn.Conv2d(1, 1, 3, padding=1, bias=False)
pool2 = lambda: nn.MaxPool2d(2, 2)

cases = [
    ("A 3 个 conv3(s1,p1)",            7,  build_stack(3)),
    ("B 5 个 conv3(s1,p1)",            11, build_stack(5)),
    ("C 3 个 conv3(s2,p1)",            15, build_stack(3, s=2)),
    ("D conv3-conv3-pool2-conv3",      10, nn.Sequential(conv3(), conv3(), pool2(), conv3())),
    ("E conv3x2-pool2-conv3x2-pool2-conv3", 24,
     nn.Sequential(conv3(), conv3(), pool2(), conv3(), conv3(), pool2(), conv3())),
]

for name, manual, net in cases:
    print(f"\n{name} | 手算 {manual}")
    actual = measure_rf(net)
    flag = "OK" if actual == manual else "WRONG"
    print(f"    手算 {manual:3d} | 实测 {actual:3d} | {flag}")

print()
print("=" * 66)
print("任务 2 主表：堆叠层数 vs 感受野（p=1, s=1）")
print("=" * 66)
for L in [1, 2, 3, 4, 5, 10, 20]:
    net = build_stack(L)
    print(f"\n层数 {L:2d} | 公式 1 + 2*{L} = {1 + 2 * L}")
    actual = measure_rf(net)
    flag = "OK" if actual == 1 + 2 * L else "WRONG"
    print(f"    实测 {actual:3d} | 公式 {1 + 2 * L:3d} | {flag}")

print()
print("=" * 66)
print("对照：同样的层数，换成 s=2（看感受野涨得多快，代价是什么）")
print("=" * 66)
for L in [1, 2, 3]:
    net = build_stack(L, s=2)
    print(f"\nL={L} s=2")
    actual = measure_rf(net)
    print(f"    s=1 时的 RF = {1 + 2 * L} | s=2 时的 RF = {actual}")
