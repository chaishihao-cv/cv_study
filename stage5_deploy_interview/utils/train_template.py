# 通用训练模板（分类/检测/分割反复复用，重点吃透）

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 数据集
train_loader = None  # DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = None    # DataLoader(val_dataset, batch_size=8, shuffle=False)

# 2. 模型 / 损失 / 优化器
model = None
model = model.to(device)
criterion = None
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 30
for epoch in range(epochs):
    # -------- train --------
    model.train()
    train_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        pred = model(imgs)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # -------- val --------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            pred = model(imgs)
            val_loss += criterion(pred, labels).item()
    val_loss /= len(val_loader)

    print(f"Epoch:{epoch} train_loss:{train_loss:.4f} val_loss:{val_loss:.4f}")

# torch.save(model.state_dict(), "best.pth")
