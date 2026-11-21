import torch
import matplotlib.pyplot as plt

# a = torch.tensor([1,2,3])    # int64, 不能求导
# b = torch.tensor([1.,2.,3.])  # float32, 可以求导
# c = torch.tensor([1.,2.,3.], requires_grad=True)  

# print(a.dtype, a.requires_grad)   #torch.int64 False
# print(b.dtype, b.requires_grad)   #torch.float32 False
# print(c.dtype, c.requires_grad)   #torch.float32 True
# print(a.device)

# x = torch.arange(12)
# print(x)
# print(x.shape)
# x.reshape(3, 4)
# print(x.reshape(3, 4))
# print(x.reshape(3, 4).shape)


# x = torch.randn(3, 4) 
# print(x)
# y = torch.rand(3,4)
# print(y)
# x2 = x.unsqueeze(0).shape
# print(x2)

# x = torch.randn(3,224,224)
# print(x)
# print(x.shape)
# s = x.permute(1,2,0)
# print(s)
# print(s.shape)

# a = torch.randn(2,3)
# b = torch.randn(2,3)
# print(a)
# print(b)
# print(a.shape)
# print(b.shape)
# a_b = torch.cat([a,b], dim=0)
# print(a_b)
# print(a_b.shape)

# ab = torch.stack([a,b], dim=0)
# print(ab)
# print(ab.shape)

# x = torch.randn(4,5)
# print(x)
# print(x.shape)
# print(x[0])
# print(x[:,0])
# print(x[1:3,:2])
# print(x[x>0])
# print(x[[0,2,3]])

# w = torch.tensor(0.0, requires_grad=True)
# y = w * 3 +1
# loss = y ** 2
# print(loss.grad_fn)

# w = torch.tensor(2.0, requires_grad=True)
# u = w * 3
# u.backward()
# print(w.grad)
# print(u.grad)

# w = torch.tensor(0.0, requires_grad=True)
# x,y_true=2.0,3.0
# for i in range(3):
#     pred = w * x 
#     loss = (pred - y_true) ** 2
#     loss.backward()
#     print(f'第{i+1}次backward后，w.grad = {w.grad.item()}')


# torch.manual_seed(42)

# # 造数据（和 day1 一样：真值 w=2, b=3，加噪声）
# x = torch.linspace(-1, 1, 100)
# y = 2 * x + 3 + 0.3 * torch.randn(100)

# # 参数：唯一区别就是多了 requires_grad=True
# w = torch.tensor(0.0, requires_grad=True)
# b = torch.tensor(0.0, requires_grad=True)

# lr = 0.1
# loss_history = []

# for epoch in range(200):
#     pred = w * x + b
#     loss = ((pred - y) ** 2).mean()

#     loss_history.append(loss.item())
#     loss.backward()

#     # ===== 梯度核对：只在第 1 轮做一次 =====
#     if epoch == 0:
#         with torch.no_grad():
#             grad_w_hand = (2 * (pred - y) * x).mean().item()
#             grad_b_hand = (2 * (pred - y)).mean().item()
#         print(f"【核对】w: autograd={w.grad.item():.8f}  手算={grad_w_hand:.8f}  差={abs(w.grad.item()-grad_w_hand):.2e}")
#         print(f"【核对】b: autograd={b.grad.item():.8f}  手算={grad_b_hand:.8f}  差={abs(b.grad.item()-grad_b_hand):.2e}")

#     with torch.no_grad():
#         w -= lr * w.grad
#         b -= lr * b.grad
#     w.grad = None
#     b.grad = None

#     # ===== 每 10 轮打一次进度 =====
#     if epoch % 10 == 0 or epoch == 199:
#         print(f"epoch {epoch:3d}  w={w.item():.4f}  b={b.item():.4f}  loss={loss.item():.6f}")



# plt.plot(loss_history, label="torch autograd")
# plt.xlabel("epoch"); plt.ylabel("loss")
# plt.title("Day3: gradient descent with autograd")
# plt.grid(alpha=0.3); plt.legend()
# plt.show()

a = torch.randn(3,4)
print(a)
print(a.shape)
b = torch.tensor([5.,6.0,7.0], requires_grad=True)
print(b)
print(b.shape)
try:
    a_b = a + b
    print(a_b)
    print(a_b.shape)
except RuntimeError as err:
    print('a + b 会报错，因为 a 是 (3,4) 而 b 是 (3,1)')
c = torch.tensor([1.,2.,3.,4.])
print(c)
print(c.shape)
a_c = a + c
print(a_c)
print(a_c.shape)

d = torch.tensor(5)

d_a = d + a
print(d_a)
print(d_a.shape)

e = torch.tensor([[1,2,3,4]])
e_a = e + a
print(e_a)
print(e_a.shape)

f = torch.tensor([[10.], [20.], [30.]])   # (3,1)
print((a + f).shape)    # (3,4)
print((a + f))          # 每行加自己的那个数
