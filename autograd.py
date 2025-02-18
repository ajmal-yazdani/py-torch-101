import torch


# Binary Cross Entropy Loss for scalar
def binary_cross_entropy_loss(prediction, target):
    epsilon = 1e-8  # Small value to avoid log(0)
    prediction = torch.clamp(prediction, epsilon, 1 - epsilon)
    return -(target * torch.log(prediction) + (1 - target) * torch.log(1 - prediction))


x = torch.tensor(6.7)
y = torch.tensor(0.0)

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

z = w * x + b

y_pred = torch.sigmoid(z)

loss = binary_cross_entropy_loss(y_pred, y)
loss.backward()

print(w.grad)
print(b.grad)
