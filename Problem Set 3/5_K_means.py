import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

A = imread('data/mandrill-large.tiff')
#plt.figure(figsize=(8,8))
#plt.imshow(A)

#B = imread('data/mandrill-small.tiff')
#new_B = B

def initialization(B,k):
    init_mu = np.zeros((k,3))
    for i in range(k):
        x,y = np.random.randint(low=0,high=B.shape[0]),np.random.randint(low=0,high=B.shape[1])
        init_mu[i,:] = B[x,y,:]
    return init_mu

def Kmeans(B,iter,k):
    m,n = B.shape[:2]
    mu = initialization(B,k)

    for it in range(iter):
        asignment_num = np.zeros((k, 3))  # 每个聚类中心分配的样本点总和
        nums = np.zeros((k, 1))           # 每个聚类中心分配的样本点个数
        for i in range(m):          # 求每个样本点最近的聚类中心
            for j in range(n):
                dist = [[0] for x in range(k)]
                for c in range(k):   #计算每个样本点到聚类中心的欧式距离
                    d = B[i,j,:] - mu[c,:]
                    dist[c] = d.dot(d.T)
                c_i = np.argmin(dist)
                asignment_num[c_i] += B[i,j,:]
                nums[c_i] += 1
        for i in range(k):           # 求聚类中心的平均值
            if nums[i]>0:
                mu[i,:] = asignment_num[i] / nums[i]
    return mu

def compression(B,new_B,mu,k):
    m,n = B.shape[:2]
    for i in range(m):
        for j in range(n):
            dist = [[0] for x in range(k)]
            for c in range(k):
                d = B[i,j,:] - mu[c]
                dist[c] = d.dot(d.T)
            c_i = np.argmin(dist)
            new_B[i,j,:] = mu[c_i,:]
    return new_B

if __name__ == '__main__':
    B =new_B= A
    k = 16
    iter = 30
    mu = Kmeans(B,iter,k)
    new_B = compression(B,new_B,mu,k)
    plt.figure(figsize=(8,8))
    plt.imshow(new_B)
    plt.show()



