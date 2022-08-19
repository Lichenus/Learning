import torch
import torch.nn.functional as F
from torch import nn
import dataInit
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def model(x):
    return x.mm(weights) + bias


loss_func = F.cross_entropy
bs = 64
xb = dataInit.x_train[0:bs]  # a mini-batch from x
yb = dataInit.y_train[0:bs]
weights = torch.randn([784, 10], dtype=torch.float,  requires_grad=True)
bs = 64
bias = torch.zeros(10, requires_grad=True)


class MnistNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x


net = MnistNN()

train_ds = TensorDataset(dataInit.x_train, dataInit.y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(dataInit.x_valid, dataInit.y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs*2)






