import numpy as np

def read_matrix(filename):
    with open(filename,'r') as fd:
        title = fd.readline()
        rows,cols =[int(x) for x in fd.readline().strip().split()]
        matrix = np.zeros((rows,cols))
        tokens = fd.readline().strip().split()
        Y_label =[]
        for i,line in enumerate(fd.readlines()):
             num = [int(x) for x in line.strip().split()]
             Y_label.append(num[0])
             index = np.cumsum(np.array(num[1:-1:2]))
             times = np.array(num[2::2])
             matrix[i,index] = times
    return title,tokens,Y_label,matrix