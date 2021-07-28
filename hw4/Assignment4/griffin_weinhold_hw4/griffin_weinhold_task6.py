import numpy as np 
from numpy import linalg as LA
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
import sys

if __name__ == '__main__':

    edge_file = sys.argv[1]
    out_file = sys.argv[2]

    max_node = -1


    with open(edge_file, 'r') as reader:
        for line in reader.readlines():
            vertices = line.split() 
            u = int(vertices[0])
            v = int(vertices[1])


            if u > max_node:
                max_node = u 

            if v > max_node:
                max_node = v

    A = np.zeros((max_node+1, max_node+1))

    N = len(A)

    with open(edge_file, 'r') as reader:
        for line in reader.readlines():
            vertices = line.split() 
            u = int(vertices[0])
            v = int(vertices[1])
            if u != v:
                A[u, v] = 1

    A = A.T

    beta = 0.8
    colsum = A.sum(axis=0)            
    rowsum = A.sum(axis=1)

    TM = np.zeros((N, N))
    S = np.full((N, N), (1-beta) *(1/N))

    for i in range(len(A)):
        for j in range(len(A[i])) :


            if colsum[j] == 0:
                    S[i, j] = (1/N)

            if A[i, j] > 0:
                k = colsum[j]
                TM[i, j] = 1/k
              
            else:
                
                TM[i, j] = 0        



    tmp = (beta * TM) + S

    w, v = LA.eig(tmp)


    eigen = v[:, w.argmax()]
    asc = eigen.argsort()
    desc = asc[::-1]
    pr = desc[:20]

    

    myfile = open(out_file, 'w')
    for i in range(20):
        myfile.write(str(pr[i]))
        myfile.write('\n')
    

    myfile.close()




