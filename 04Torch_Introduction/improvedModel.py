import numpy as np
import torch

use_gpu = torch.cuda.is_available()


def improved_model(labels, feature_list, input_features, input_size, hidden_size, output_size, batch_size):

    my_nn = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.Sigmoid(),
        torch.nn.Linear(hidden_size, output_size),
    )
    cost = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(my_nn.parameters(), lr=0.001)

    # 训练网络
    losses = []
    for i in range(5000):
        batch_loss = []
        # MINI-Batch方法来进行训练
        for start in range(0, len(input_features), batch_size):
            end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
            xx = torch.tensor(input_features[start:end], dtype=torch.float, requires_grad=True)
            yy = torch.tensor(labels[start:end], dtype=torch.float, requires_grad=True)
            prediction = my_nn(xx)
            loss = cost(prediction, yy)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            batch_loss.append(loss.data.numpy())

        # 打印损失
        if i % 500 == 0:
            losses.append(np.mean(batch_loss))
            print('epoch: ' + str(i) + '   ' + 'loss: ' + str(np.mean(batch_loss)))

    return
