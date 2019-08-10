from read_Matrix import read_matrix
import numpy as np
import matplotlib.pyplot as plt
import time

def svm_train(filename):
    _,_,Y_label,matrix = re  ad_matrix(filename)
    Y_label = np.array(Y_label)*2-1

    m,n = matrix.shape
    matrix = 1. * (matrix>0)

    square_matrix = np.sum(matrix**2,1).reshape(m,1)
    gram = matrix.dot(matrix.T)
    tau = 8
    m_lambda = 1/(64*m)
    Kerenl_matrix = np.exp(-(np.repeat(square_matrix,m,1)+np.repeat(square_matrix.T,m,0)-2*gram)/(2*tau**2))

    alpha = np.zeros((m,1))
    avg_alpha = np.zeros((m,1))
    for i in range(40*m):
        ind = int(np.ceil(np.random.rand()*m) )-1   #随机抽取样本进行训练
        margin = Y_label[ind]*np.dot(Kerenl_matrix[ind,:],alpha)
        grad_alpha = (-1*(margin<1)*Y_label[ind]*Kerenl_matrix[:,ind]).reshape((m,1)) + m*m_lambda*Kerenl_matrix[:,ind].reshape((m,1))*alpha
        alpha = alpha - grad_alpha/np.sqrt(i+1)
        avg_alpha += alpha
    avg_alpha /= (40*m)
    return avg_alpha,square_matrix,matrix

def svm_test(alpha,square_train,matrix_train):
    _,_,Y_test,matrix_test = read_matrix("data/MATRIX.TEST")
    Y_test = (np.array(Y_test)*2-1)
    matrix_test = 1. *(matrix_test>0)
    m_train = square_train.shape[0]
    m_test = matrix_test.shape[0]

    square_test = np.sum(matrix_test**2,1).reshape((m_test,1))
    gram_test = matrix_test.dot(matrix_train.T)
    tau = 8
    Kernel_test = np.exp(-(np.repeat(square_test,m_train,1)+np.repeat(square_train.T,m_test,0)-2*gram_test)/(2*tau**2))
    Y_pred = Kernel_test.dot(alpha)

    num = 0
    Y = Y_pred*Y_test.reshape((m_test,1))
    for i,y in enumerate(Y):
        if y<=0:
            num +=1
    error = num/m_test
    return error

def main():
    train_size = [50,100,200,400,800,1400]
    #train_size = [800]
    #error = []
    for i,size in enumerate(train_size):
        start = time.time()
        avg_alpha,square_train,matrix_train = svm_train("data/MATRIX.TRAIN."+str(size))
        error = svm_test(avg_alpha,square_train,matrix_train)*100
        times = time.time()-start
        print("Size:"+str(size)+"  error:"+str(error)+"  times:"+str(times))
main()
