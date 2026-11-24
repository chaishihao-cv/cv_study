# day5 卷积尺寸计算 + 单层 Conv2d 测试
# 任务：手算 Conv2d 输出尺寸，并与实际 shape 核对
# 思考题：padding=same 与 valid 区别？stride 变大特征图怎么变？
# 公式：out = (in + 2*padding - kernel) // stride + 1

import torch
import torch.nn as nn

# TODO：在这里写代码
# x = torch.tensor([[1.,2.,3.,4.,5.],
#                   [6.,7.,8.,9.,10.],
#                   [11.,12.,13.,14.,15.],
#                   [16.,17.,18.,19.,20.],
#                   [21.,22.,23.,24.,25.]]).view(1,1,5,5)

# k = torch.tensor([[1.,0.,-1.],
#                   [1.,0.,-1.],
#                   [1.,0.,-1.]]).view(1,1,3,3)

# conv = nn.Conv2d(1,1,3, bias=False)
# conv.weight.data = k

# y = conv(x)
# print(y)
# print(y.squeeze())


print('\n--- 任务2: 输出尺寸 ---')

cfgs = [(5,3,0,1), (5,3,1,1), (5,3,2,1), (28,3,1,2), (224,7,3,2), (11,3,0,4)]
h = [(28,3,1,1),(28,3,1,2)]

# for H,k,p,s in cfgs:
#     manual = (H+2*p-k)//s+1
#     m = nn.Conv2d(1,4,k,padding = p, stride = s)
#     actual = m(torch.randn(1,1,H,H)).shape[2]
#     flag = 'OK' if manual == actual else 'WRONG'
#     print(f'in={H:3d} k={k} p={p} s={s} | 手算={manual:3d} 实际={actual:3d} | {flag}')

for H,k,p,s in h:
    manual = (H+2*p-k)//s+1
    m = nn.Conv2d(1,4,k,padding = p, stride = s)
    actual = m(torch.randn(1,1,H,H)).shape[2]
    flag = 'OK' if manual == actual else 'WRONG'
    print(f'in={H:3d} k={k} p={p} s={s} | 手算={manual:3d} 实际={actual:3d} | {flag}')

print('\n--- 任务3:参数量 ---')

cases = [(1,16,3,1), (1,64,3,1), (3,64,3,1), (16,32,3,1), (3,64,7,2)]

for ci, co, k, s in cases:
    manual = k*k*ci*co+co
    m = nn.Conv2d(ci,co,k,stride=s)
    actual = sum(p.numel() for p in m.parameters())
    flag = 'OK' if manual == actual else 'WRONG'
    print(f'ci={ci} co={co} k={k} s={s} | 手算={manual:3d} 实际={actual:3d} | {flag}')

print('\n--- 任务4: FLOPs ---')
out_h = (224 + 2*1 - 3)//1 + 1
macs = 64 * out_h * 224 * 9 * 3
print(f'out={out_h}x{out_h}  MACs={macs:,} = {macs/1e6:.2f} M')
print(f'FLOPs = 2*MACs = {2*macs/1e6:.2f} M')
print(f'参数量 = {3*3*3*64+64}')

import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("设备:", dev)

# ---------- 数据（MNIST 已在本地，download=False）----------
train_set = datasets.MNIST('D:/cv_study/data', train=True,  download=False, transform=transforms.ToTensor())
test_set  = datasets.MNIST('D:/cv_study/data', train=False, download=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_set, batch_size=64,  shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False, num_workers=0)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = nn.Conv2d(1,16,3,padding=1)
        self.f2 = nn.Conv2d(16,32,3,padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32*7*7,10)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.pool(self.relu(self.f1(x)))
        x = self.pool(self.relu(self.f2(x)))
        return self.fc(x.flatten(start_dim=1))

model = CNN().to(dev)
print(model)


# ---------- 参数量：手算 + 实测 ----------
print("\n--- 参数量 ---")
print("conv1: 3*3*1*16 + 16  =", 3*3*1*16 + 16)
print("conv2: 3*3*16*32 + 32 =", 3*3*16*32 + 32)
print("fc:    32*7*7*10 + 10 =", 32*7*7*10 + 10)
print("总参数:", sum(p.numel() for p in model.parameters()))

# ---------- shape 探查（不改 forward，单独跑一次）----------
print("\n--- 每层 shape ---")
x, y = next(iter(train_loader))
x = x.to(dev)
with torch.no_grad():
    print("输入       :", tuple(x.shape))
    h = model.relu(model.f1(x));         print("conv1+relu :", tuple(h.shape))
    h = model.pool(h);                   print("pool1      :", tuple(h.shape))
    h = model.relu(model.f2(h));         print("conv2+relu :", tuple(h.shape))
    h = model.pool(h);                   print("pool2      :", tuple(h.shape))
    h = h.flatten(start_dim=1);          print("flatten    :", tuple(h.shape))
    h = model.fc(h);                     print("fc         :", tuple(h.shape))

# ---------- 训练 ----------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def evaluate():
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(dev), y.to(dev)
            correct += (model(x).argmax(dim=1) == y).sum().item()
    return correct / len(test_set)

epochs = 3
loss_history, acc_history = [], []

for epoch in range(epochs):
    model.train()
    total, n = 0.0, 0                      # ← 每轮开头清零
    t0 = time.time()

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(dev), y.to(dev)

        out  = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item() * x.size(0)
        n     += x.size(0)

        if batch_idx % 200 == 0:
            print(f"  batch {batch_idx:4d}  loss={loss.item():.4f}")

    avg = total / n
    acc = evaluate()
    loss_history.append(avg)
    acc_history.append(acc)
    print(f"epoch{epoch+1}  loss={avg:.4f}  acc={acc:.4f}  用时={time.time()-t0:.1f}s")

print("\nloss 曲线数据:", [round(v, 4) for v in loss_history])
print("acc  曲线数据:", [round(v, 4) for v in acc_history])