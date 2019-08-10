
import numpy as np # imports a fast numerical programming library
import matplotlib.pyplot as plt
import pandas as pd #lets us handle data as dataframes


df_x = pd.read_csv('data/logistic_x.txt', sep= "\ +", names=["x1","x2"] ,header=None, engine='python')
df_y  = pd.read_csv('data/logistic_y.txt', sep="\ +", names=["y"],header = None ,engine="python")

#df_y = df_y.astype(int)


x = np.hstack((np.ones((df_x.shape[0],1)),df_x.values))
y = df_y.values
m = x.shape[0]
n = x.shape[1]
theta = np.zeros((n,1))


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def grad_l(theta,x,y):
    z = y *x.dot(theta)
    res = (-1/m)*np.dot(x.T,sigmoid(-z)*y)  ## 此sigmoid 中的参数应为 -
    return res


def Hessian_l(theta,x,y):
    H = np.zeros((n,n))
    for i in range(m):
        a = np.array(x[i]).reshape((3, 1))
        h_theta = sigmoid(-y[i]*(np.dot(x[i],theta)))
        H += np.dot(a,a.T)*(h_theta*(1-h_theta))  # 向量的点积需要注意shape
    H = H/m
    return H

def Newton(theta,eps,x,y):
    for i in range(eps):
        G = grad_l(theta,x,y)
        H = Hessian_l(theta,x,y)
        theta = theta - np.linalg.inv(H).dot(G)
    return theta
    print("theta",theta)
theta = Newton(theta,20,x,y)

def scatter_data(theta):
    pos,_ = np.where(y>0)
    x1 = x[pos,1]
    x2 = x[pos,2]
    plt.plot(x1,x2,'go')  # 绘制当y>0的点
    pos, _ = np.where(y < 0)
    x1 = x[pos, 1]
    x2 = x[pos, 2]
    plt.plot(x1,x2,'rx')# 绘制当y<0的点
    x1 = np.linspace(np.min(x[:,1]),np.max(x[:,1]),100)
    x2 = -(theta[0]+theta[1]*x1)/theta[2]
    plt.plot(x1,x2,linewidth=2) #绘制超平面
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
scatter_data(theta)