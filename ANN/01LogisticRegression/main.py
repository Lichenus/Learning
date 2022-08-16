import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

index = 80
# plt.imshow(train_set_x_orig[index])
# plt.show()
# print("y=" + str(train_set_y[:,index]) + ", it's a " + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") + "picture")

m_train = train_set_y.shape[1] #训练集里图片的数量。
m_test = test_set_y.shape[1] #测试集里图片的数量。
num_px = train_set_x_orig.shape[1] #训练、测试集里面的图片的宽度和高度（均为64x64）。
# print("训练集的数量: m_train = " + str(m_train))
# print("测试集的数量 : m_test = " + str(m_test))
# print("每张图片的宽/高 : num_px = " + str(num_px))
# print("每张图片的大小 : (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print("训练集_图片的维数 : " + str(train_set_x_orig.shape))
# print("训练集_标签的维数 : " + str(train_set_y.shape))
# print("测试集_图片的维数: " + str(test_set_x_orig.shape))
# print("测试集_标签的维数: " + str(test_set_y.shape))
# print()

#X_flatten = X.reshape(X.shape [0]，-1).T ＃X.T是X的转置
#将训练集的维度降低并转置。
train_set_x_flatten  = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
#将测试集的维度降低并转置。
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
# print("训练集降维最后的维数： " + str(train_set_x_flatten.shape))
# print("训练集_标签的维数 : " + str(train_set_y.shape))
# print("测试集降维之后的维数: " + str(test_set_x_flatten.shape))
# print("测试集_标签的维数 : " + str(test_set_y.shape))

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return (w, b)


def propagate(w, b, X, Y):
    m = X.shape[1]

    # 正向传播
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))

    # 反向传播
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    # cost = np.squeeze(cost)
    # assert (cost.shape == ())

    grads = {
        "dw": dw,
        "db": db
    }
    return (grads, cost)


def optimize(w, b, X, Y, num_iterations, learning_rate):
    global dw, db
    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 记录成本
        if i % 100 == 0:
            costs.append(cost)
        # 打印成本数据
        if(i % 500 == 0):
            print("迭代的次数：%i ， 误差值：%f" % (i, cost))

    params = {
        "w": w,
        "b": b}
    grads = {
        "dw": dw,
        "db": db}
    return (params, grads, costs)


def predict(w, b, X):
    m = X.shape[1]  # 图片的数量
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # 预测猫在图片中出现的概率
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        # 将概率a [0，i]转换为实际预测p [0，i]
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    # 使用断言
    assert (Y_prediction.shape == (1, m))
    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate):
    w, b = initialize_with_zeros(X_train.shape[0])

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)

    # 从字典“参数”中检索参数w和b
    w, b = parameters["w"], parameters["b"]

    # 预测测试/训练集的例子
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # 打印训练后的准确性
    print("训练集准确性：", format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100), "%")
    print("测试集准确性：", format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100), "%")

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediciton_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations}
    return d

print("====================测试model====================")

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 20000, learning_rate = 0.005)

#绘制图
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()
