import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd_train = pd.read_csv('data/smooth_train.csv')
pd_train.drop(columns=[pd_train.columns.values[0]],inplace=True)
X = pd_train.columns.values.astype(float)
Y_train = pd_train.values
lyman_alpha = 1200

pd_test = pd.read_csv('data/smooth_test.csv')
pd_test.drop(columns=[pd_test.columns.values[0]],inplace=True)

Y_test = pd_test.values

def distant(y_1,y_2):  ##计算两个向量之间的距离
    return np.sum((y_1-y_2)**2)

def neighber(Y,y_i,k):  ## 返回向量yi的邻居节点的坐标，和其距离最大的坐标
    d_list = np.zeros((Y.shape[0]))
    for i in range(Y.shape[0]):
        d_list[i] = distant(Y[i,:],y_i)
    return np.argsort(d_list)[:k],np.argsort(d_list)[-1]

def ker(t):
    return max(1-t,0)


def funcion_regression(Y_train,Y_test): # Y_train 是用来当做样本训练的，Y_test是需要估计的左边的函数值。Y_train可以于Y_test相同
    Y_train_right = Y_train[:, X>=lyman_alpha+100]
    Y_train_left = Y_train[:, X < lyman_alpha]
    Y_test_right = Y_test[:,X>=lyman_alpha+100]
    Y_test_left = Y_test[:,X < lyman_alpha]

    m_test,n_test = Y_test_left.shape
    f_left_hat = np.zeros((m_test,n_test))

    for i in range(m_test):
        y_i = Y_test_right[i,:]
        neighbers,max_id = neighber(Y_train_right,y_i,3)
        h = distant(Y_train_right[max_id,:],y_i)

        for lyman in range(n_test):
            f_up = 0
            f_down = 0
            for j in neighbers:
                t = ker(distant(Y_train_right[j,:],y_i)/h)
                f_down += t
                f_up += t*Y_train_left[j,lyman]
            f_left_hat[i,lyman] = f_up/f_down
    return f_left_hat

#### 估计训练集的左边
f_left_hat = funcion_regression(Y_train,Y_train)
Y_test_left = Y_train[:,X < lyman_alpha]
m,n = Y_test_left.shape

aver_error = 0           # 计算训练误差
for i in range(m):
    aver_error += distant(Y_test_left[i,:],f_left_hat[i,:])
aver_error /= (m)
for i in range(m):        # 作图
    plt.plot(np.linspace(1150,1200,50),f_left_hat[i,:])
plt.show()
print("train data error=",aver_error)

#### 估计测试集的左边
f_left_hat = funcion_regression(Y_train,Y_test)
Y_test_left = Y_test[:,X<lyman_alpha]
h = np.mean(np.sum((f_left_hat-Y_test_left)**2,1))
print("test data error=",h)

### 绘制test集中第一个的smooth曲线和估计的曲线
plt.figure(2)
plt.plot(X,Y_test[0,:],label="1th smooth")
plt.plot(np.linspace(1150,1200,50),f_left_hat[0,:],label="1th estimated")
### 绘制test集中第六个的smooth曲线和估计的曲线
plt.figure(3)
plt.plot(X,Y_test[5,:],label="6th smooth")
plt.plot(np.linspace(1150,1200,50),f_left_hat[5,:],label="6th estimated")
plt.legend()
plt.show()