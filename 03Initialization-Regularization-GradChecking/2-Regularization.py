# 2. 正则化模型：
# 	2.1：使用二范数对二分类模型正则化，尝试避免过拟合。
# 	2.2：使用随机删除节点的方法精简模型，同样是为了尝试避免过拟合。
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils   # 第一部分，初始化
import reg_utils    # 第二部分，正则化
import gc_utils     # 第三部分，梯度校验
import paraInit     # 参数初始化
import model_of_Reg        # 模型


plt.rcParams['figure.figsize'] = (9.0, 5.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
train_X, train_Y, test_X, test_Y = reg_utils.load_2D_dataset(is_plot=False)
plt.show()

parameters = model_of_Reg.model(train_X, train_Y, is_plot=True)
print("训练集:")
predictions_train = reg_utils.predict(train_X, train_Y, parameters)
print("测试集:")
predictions_test = reg_utils.predict(test_X, test_Y, parameters)

plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)
