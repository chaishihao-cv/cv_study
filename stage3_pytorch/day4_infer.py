import torch, torch.nn as nn
from torchvision import datasets, transforms

dev = "cuda" if torch.cuda.is_available() else "cpu"

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.fc2(self.relu(self.fc1(x)))

path = "D:/cv_study/stage3_pytorch/day4_mlp_mnist.pth"

model = MLP().to(dev)
model.load_state_dict(torch.load(path, weights_only=True))
model.eval()

test_set = datasets.MNIST("D:/cv_study/data", train=False, download=False,
                          transform=transforms.ToTensor())

img, label = test_set[0]
with torch.no_grad():
    out = model(img.unsqueeze(0).to(dev))
    pred = torch.argmax(out, dim=1).item()

print(f"真值: {label}  预测: {pred}  置信度: {torch.softmax(out, dim=1).max().item()*100:.2f}%")
