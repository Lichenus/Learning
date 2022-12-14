import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn import preprocessing




def data_init(is_draw=True):
    features = pd.read_csv('temps.csv')
    # 处理时间数据
    # 分别得到年，月，日
    years = features['year']
    months = features['month']
    days = features['day']

    # datetime格式
    dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
             zip(years, months, days)]
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
    if is_draw:
        # 准备画图
        # 指定默认风格
        plt.style.use('fivethirtyeight')

        # 设置布局
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(9, 9))
        fig.autofmt_xdate(rotation=45)

        # 标签值
        ax1.plot(features['actual'])
        ax1.set_xlabel('');
        ax1.set_ylabel('Temperature');
        ax1.set_title('Max Temp')

        # 昨天
        ax2.plot(dates, features['temp_1'])
        ax2.set_xlabel('');
        ax2.set_ylabel('Temperature');
        ax2.set_title('Previous Max Temp')

        # 前天
        ax3.plot(dates, features['temp_2'])
        ax3.set_xlabel('Date');
        ax3.set_ylabel('Temperature');
        ax3.set_title('Two Days Prior Max Temp')

        # 我的逗逼朋友
        ax4.plot(dates, features['friend'])
        ax4.set_xlabel('Date');
        ax4.set_ylabel('Temperature');
        ax4.set_title('Friend Estimate')

        plt.tight_layout(pad=2)
        plt.show()
    # 独热编码
    features = pd.get_dummies(features)
    # 标签
    labels = np.array(features['actual'])

    # 在特征中去掉标签
    features = features.drop('actual', axis=1)

    # 名字单独保存一下，以备后患
    feature_list = list(features.columns)

    # 转换成合适的格式
    features = np.array(features)
    input_features = preprocessing.StandardScaler().fit_transform(features)

    return labels, feature_list, input_features





