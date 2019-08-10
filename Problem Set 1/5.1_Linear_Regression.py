import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd_train = pd.read_csv('data/quasar_train.csv')

y = pd_train.head(1).values.T  # 获取第一行为样本数据
x = np.vstack((np.ones(pd_train.columns.shape), pd_train.columns.values.astype(float))).T # 获取列名为特征，并增加x0

def normal_equation(x,y):
    theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    return theta
theta = normal_equation(x,y)

plt.plot(x[:,1],y,'b+',label="Raw data")
x = np.linspace(np.min(x[:,1]),np.max(x[:,1]),100)
plt.plot(x,theta[1]*x+theta[0],'r',linewidth=2,label="Regression line")
plt.legend()
plt.show()
