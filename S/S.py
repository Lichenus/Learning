import torch
import torch.nn as nn
import testData
import time

# 初始化回归模型，输入和输出的维数
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


input_dim = 1
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)



# 指定参数和损失函数
epochs = 1000
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

tic = time.time()
# 训练模型
for epoch in range(epochs):
    epoch += 1

    inputs = torch.from_numpy(testData.x_train)
    labels = torch.from_numpy(testData.y_train)

    # 每次迭代时梯度清零
    optimizer.zero_grad()
    # 前向传播
    outputs = model(inputs)
    # 计算损失
    loss = criterion(outputs, labels)
    # 反向传播
    loss.backward()
    # 更新权重
    optimizer.step()

    if epoch % 50 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))

toc = time.time()
print(str(1000*(toc-tic)) + "ms")