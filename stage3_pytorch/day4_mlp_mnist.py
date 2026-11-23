# day4 MLP 手写数字识别（MNIST 完整训练脚本）
# 任务：搭建 MLP 训练 MNIST；打印每层输出 shape；画 loss 曲线
# 思考题：反向传播更新的是什么？为什么推理时不需要计算梯度？

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# TODO：在这里写代码
from torchvision import datasets, transforms

train_set = datasets.MNIST('D:/cv_study/data', train=True, download=False,transform=transforms.ToTensor())
test_set = datasets.MNIST('D:/cv_study/data', train=False, download=False,transform=transforms.ToTensor())

# print(len(train_set), len(test_set))
# img,label = train_set[0]
# print(img.shape, label)

# model = nn.Sequential(
#     nn.Flatten(),
#     nn.Linear(784,128),
#     nn.ReLU(),
#     nn.Linear(128,10),
# )
# print(model)

# total = sum(p.numel() for p in model.parameters())
# print('总参数：', total)

# x = torch.randn(32,1,28,28)
# print('输入：', x.shape,'输出：', model(x).shape)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784,128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128,10)

    def forward(self, x):
        #print("原始输入 x shape:", x.shape)
        x = x.flatten(start_dim=1)
       # print("flatten 之后:", x.shape)
        x = self.fc1(x)
       # print("fc1 输出:", x.shape)
        x = self.relu(x)
      #  print("relu 输出:", x.shape)
        x = self.fc2(x)
       # print("fc2 输出（最终输出）:", x.shape)
        return x

# model = MLP()
# x = torch.randn(32, 1, 28, 28)
# out = model(x)
# print(model)
# print(out.shape)
    
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MLP().to(dev)
print('总参数：', sum(p.numel() for p in model.parameters()))
print('输出形状：',model(torch.randn(4,1,28,28).to(dev)).shape)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def evaluate():
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(dev)
            y = y.to(dev)
            pred = torch.argmax(model(x), dim=1)
            correct += (pred == y).sum().item()
    return correct / len(test_set)



epochs = 5
loss_history = []
acc_history = []
for epoch in range(epochs):
    model.train()
    total, n= 0.0, 0 
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(dev)
        y = y.to(dev)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)

    
        if batch_idx % 200 == 0:
            print(f'batch{batch_idx} loss={loss.item():.4f}')
    avg = total / n
    loss_history.append(avg)
    acc_history.append(evaluate())
    print(f'epoch{epoch+1} loss={avg:.4f} acc={evaluate():.4f}')



plt.figure(figsize=(10, 4))
plt.subplot(1,2,1); plt.plot(loss_history, marker="o")
plt.title("Train loss"); plt.xlabel("epoch"); plt.grid(alpha=0.3)
plt.subplot(1,2,2); plt.plot([a*100 for a in acc_history], marker="o")
plt.title("Test acc (%)"); plt.xlabel("epoch"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()

path = 'D:/cv_study/stage3_pytorch/day4_mlp_mnist.pth'
torch.save(model.state_dict(), path)
print('模型已保存到', path)






