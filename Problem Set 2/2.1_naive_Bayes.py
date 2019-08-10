from read_Matrix import read_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def nb_train(Y_lable,matrix):  ## 贝叶斯训练
    Y_lable = np.array(Y_lable)  # 训练集的标签
    param = {}
    N = matrix.shape[1]
    spam_matrix = matrix[Y_lable==1,:]   # 正样本（垃圾邮件）
    nonspam_matrix = matrix[Y_lable==0,:]  # 负样本

    n_spam= np.sum(spam_matrix,1)    # 每一样本单词个数
    n_nonspam = np.sum(nonspam_matrix,1)

    param["phi_spam"] = (np.sum((spam_matrix),0)+1)/(np.sum(n_spam)+N)  #Φk|y=1
    param["phi_nonspam"] = (np.sum((nonspam_matrix),0)+1)/(np.sum(n_nonspam)+N) #Φk|y=0
    param["phi"] = nonspam_matrix.shape[0]/matrix.shape[0]
    return param

def nb_test(param):
    _,_,Y,t_matrix = read_matrix("data/MATRIX.TEST")
    post_nonspam = np.zeros((t_matrix.shape[0],1))
    post_spam = np.zeros((t_matrix.shape[0],1))
    for i in range(t_matrix.shape[0]):
        p0 = 1
        p1 = 1
        for j in range(t_matrix.shape[1]):
            if t_matrix[i,j]!=0:
                p0 = p0* param["phi_nonspam"][j]**t_matrix[i,j]
                p1 = p1* param["phi_spam"][j]**t_matrix[i,j]
        post_nonspam[i] = p0 *param["phi"]
        post_spam[i] = p1 * (1-param["phi"])
    #post_nonspam = np.sum(np.log(t_matrix*param["phi_nonspam"]),1)+np.log(param["phi"])   # y=0时 p(xi|y=0)p(y=0) ，不需要计算p(x)
    #post_spam = np.sum(np.log(t_matrix*param["phi_spam"]),1)+np.log(1-param["phi"])  # y=1时 p(xi|y=1)p(y=1)
    Y_pred = np.zeros(t_matrix.shape[0])
    for i in range(t_matrix.shape[0]):
        if post_nonspam[i]<=post_spam[i]:
            Y_pred[i] = 1
        else:
            Y_pred[i] = 0
    return Y,Y_pred

def evaulate(Y_test,Y_pred):
    m_test = len(Y_test)
    n=0
    for i in range(m_test):
        if Y_test[i]!=Y_pred[i]:
            n += 1
    return n/m_test

def main_a():
    title,tokens,Y_label,matrix = read_matrix("data/MATRIX.TRAIN")
    param = nb_train(Y_lable, matrix)
    Y_test, Y_pred = nb_test(param)
    error = evaulate(Y_test, Y_pred)
    print("error:%1.4f" % error)

def main_b(): #### question (b) find the indicative token
    param = nb_train(Y_lable, matrix)
    a=np.argsort(np.log(param["phi_spam"]/param["phi_nonspam"]))[-5:]
    pd_token = pd.read_csv('data/TOKENS_LIST',sep='\ +',names=["index","token"],engine='python')
    tokens = pd_token.values[a,1]
    print(tokens)

def main_c():
    trian_size = [50,100,200,800,1400]
    res = []
    for i,size in enumerate(trian_size):
        _,_,Y_label,matrix = read_matrix("data/MATRIX.TRAIN."+str(size))
        param = nb_train(Y_label,matrix)
        Y_test,Y_pred = nb_test(param)
        res.append(evaulate(Y_test,Y_pred))
    plt.plot(trian_size,res)
    print(res)
    plt.show()
main_c()



