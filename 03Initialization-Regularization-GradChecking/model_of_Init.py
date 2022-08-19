import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils
import reg_utils
import gc_utils
import paraInit


def model(X, Y, learning_rate=0.02, num_iterations=15000, print_cost=True, initialization=" ", is_polt=True):
    """
    实现一个三层的神经网络：LINEAR-RELU -> LINEAR-RELU -> LINEAR-SIGMOID

    参数：
        X - 输入的数据，维度为(2, 要训练/测试的数量)
        Y - 标签，【0 | 1】，维度为(1, 对应的是输入的数据的标签)
        learning_rate - 学习速率
        num_iterations - 迭代的次数
        print_cost - 是否打印成本值，每迭代1000次打印一次
        initialization - 字符串类型，初始化的类型【"zeros" | "random" | "he"】
        is_polt - 是否绘制梯度下降的曲线图
    返回
        parameters - 学习后的参数
    """
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 10, 5, 1]

    # 选择初始化参数的类型
    if initialization == "zeros":
        parameters = paraInit.initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = paraInit.initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = paraInit.initialize_parameters_he(layers_dims)
    else:
        print("错误的初始化参数！程序退出")
        exit

    # 开始学习
    for i in range(0, num_iterations):
        # 前向传播
        a3, cache = init_utils.forward_propagation(X, parameters)

        # 计算成本
        cost = init_utils.compute_loss(a3, Y)

        # 反向传播
        grads = init_utils.backward_propagation(X, Y, cache)

        # 更新参数
        parameters = init_utils.update_parameters(parameters, grads, learning_rate)

        # 记录成本
        if i % 1000 == 0:
            costs.append(cost)
            # 打印成本
            if print_cost:
                print("第" + str(i) + "次迭代，成本值为：" + str(cost))

    # 学习完毕，绘制成本曲线
    if is_polt:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    # 返回学习完毕后的参数
    return parameters
