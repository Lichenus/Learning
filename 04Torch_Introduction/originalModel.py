import torch


def original_model(labels, feature_list, input_features):
    # 构建网络模型
    x = torch.tensor(input_features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.float)

    # 权重参数初始化
    weights = torch.randn((14, 128), dtype=torch.float, requires_grad=True)
    biases = torch.randn(128, dtype=torch.float, requires_grad=True)
    weights2 = torch.randn((128, 1), dtype=torch.float, requires_grad=True)
    biases2 = torch.randn(1, dtype=torch.float, requires_grad=True)

    learning_rate = 0.002
    losses = []

    for i in range(5000):
        # 计算隐层
        hidden = x.mm(weights) + biases
        # 加入激活函数
        hidden = torch.relu(hidden)
        # 预测结果
        predictions = hidden.mm(weights2) + biases2
        # 计算损失
        loss = torch.mean((predictions - y) ** 2)
        losses.append(loss.data.numpy())

        # 打印损失值
        if i % 500 == 0:
            print('epoch: '+str(i)+'   '+'loss: '+str(loss))
        # 返向传播计算
        loss.backward()

        # 更新参数
        weights.data.add_(- learning_rate * weights.grad.data)
        biases.data.add_(- learning_rate * biases.grad.data)
        weights2.data.add_(- learning_rate * weights2.grad.data)
        biases2.data.add_(- learning_rate * biases2.grad.data)

        # 每次迭代后清空
        weights.grad.data.zero_()
        biases.grad.data.zero_()
        weights2.grad.data.zero_()
        biases2.grad.data.zero_()

    return
