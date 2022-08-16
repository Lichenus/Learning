import numpy as np
import h5py
import matplotlib.pyplot as plt
import testCase
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
import lr_utils


np.random.seed(1)


def initialize_parameters(n_x, n_h, n_y):
    """
    参数：
        n_x - 输入层节点数量
        n_h - 隐藏层节点数量
        n_y - 输出层节点数量

    返回：
        parameter:
        W1 - 权重矩阵,维度为（n_h，n_x）
        b1 - 偏向量，维度为（n_h，1）
        W2 - 权重矩阵，维度为（n_y，n_h）
        b2 - 偏向量，维度为（n_y，1）
    """

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))