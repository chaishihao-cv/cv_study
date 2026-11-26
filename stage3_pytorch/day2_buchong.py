import torch
# 一个故意全负的输入，看 ReLU 的导数是不是 0
x = torch.tensor([-2., -1., -0.5], requires_grad=True)
y = torch.relu(x).sum()
y.backward()
print(x.grad)      # 预期 [0., 0., 0.] —— 这就是"死区"
x2 = torch.tensor([-1.,1.,-0.5],
                  requires_grad=True)
torch.relu(x2).sum().backward()
print(x2.grad)
