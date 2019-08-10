import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd_train = pd.read_csv('data/quasar_train.csv')

Y = pd_train.head(1).values.T
X = np.vstack((np.ones(pd_train.columns.shape), pd_train.columns.values.astype(float))).T

def build_weight(r,X,x_i):
    return np.diag(np.exp(-(X-x_i)[:,1]**2/(2*r**2)))

def normal_equation(r,X,Y):
    Y_hat = np.zeros(Y.shape[0])
    for i in range(Y.shape[0]):
        W = build_weight(r,X,X[i,:])
        theta = np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(Y)
        Y_hat[i] = X[i,:].dot(theta)  # 对当前xi进行预测，存储预测结果
    return Y_hat
Y_hat = normal_equation(5,X,Y)

plt.plot(X[:,1],Y_hat,label="r=5")

Y_hat = normal_equation(1,X,Y)
plt.plot(X[:,1],Y_hat,label="r=1")

Y_hat = normal_equation(10,X,Y)
plt.plot(X[:,1],Y_hat,label="r=10")

Y_hat = normal_equation(100,X,Y)
plt.plot(X[:,1],Y_hat,label="r=100")

Y_hat = normal_equation(1000,X,Y)
plt.plot(X[:,1],Y_hat,label="r=1000")

plt.legend()
plt.plot(X[:,1],Y,'+',label = "Raw data")

plt.show()

## Question 5 c(i)           smooth the train and test data
Y_train_hat = np.zeros((pd_train.shape))
for i in range(Y_train_hat.shape[0]):
    Y = pd_train.values[i,:].T
    Y_train_hat[i,:] = normal_equation(5,X,Y)
Y_train_res = pd.DataFrame(Y_train_hat,columns=pd_train.columns.values.astype(float))
Y_train_res.to_csv('smooth_train.csv')

pd_test = pd.read_csv('quasar_test.csv')
Y_test_hat = np.zeros((pd_test.shape))
for i in range(Y_test_hat.shape[0]):
    Y = pd_test.values[i,:].T
    Y_test_hat[i,:] = normal_equation(5,X,Y)
Y_test_res = pd.DataFrame(Y_test_hat,columns=pd_test.columns.values.astype(float))
Y_test_res.to_csv('smooth_test.csv')
