### Independent Components Analysis
###
### This program requires a working installation of:
###
### On Mac:
###     1. portaudio: On Mac: brew install portaudio
###     2. sounddevice: pip install sounddevice
###
### On windows:
###      pip install pyaudio sounddevice
###

import sounddevice as sd
import numpy as np

Fs = 11025



def normalize(dat):
    return 0.99 * dat / np.max(np.abs(dat))

def load_data():
    mix = np.loadtxt('data/mix.dat')
    return mix

def play(vec):
    sd.play(vec, Fs, blocking=True)

def unmixer(X):
    M, N = X.shape
    W = np.eye(N)

    anneal = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01,
              0.005, 0.005, 0.002, 0.002, 0.001, 0.001]  ## 逐渐变小学习速率
    print('Separating tracks ...')
    ######## Your code here ##########
    for alpha in anneal:  ## 使用每一个因子进行一次完整的训练
        order = np.arange(M)
        np.random.shuffle(order)   ## 将m个样本的更新顺序打乱
        x = X[order, :]
        for i in range(M):
            a = x[i, :]
            g =  1/(1+np.exp(-(W.dot(x[i,:].reshape(N,1)))))    ## sigmoid
            W += alpha * (np.dot((1-2*g.reshape(N,1)),x[i,:].reshape(1,N)) + np.linalg.inv(W.T))
    return W

def unmix(X, W):
    S = np.zeros(X.shape)

    ######### Your code here ##########
    S = np.dot(X,W.T)
    ##################################
    return S

def main():
    X = normalize(load_data())

    for i in range(X.shape[1]):
        print('Playing mixed track %d' % i)
        #play(X[:, i])

    W = unmixer(X)
    S = normalize(unmix(X, W))

    for i in range(S.shape[1]):
        print('Playing separated track %d' % i)
        play(S[:, i])

if __name__ == '__main__':
    main()
