import numpy as np 
from numpy import linalg as LA
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
import sys

if __name__ == '__main__':

    edge_file = sys.argv[1]
    out_file = sys.argv[2]
    k = int(sys.argv[3])



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
                A[u,v] = 1
                A[v,u] = 1

    D = np.zeros((N,N))

    for i in range(N):

        D[i, i] = A.sum(axis=0)[i] + A.sum(axis=1)[i]


    Lap = D - A


    w, v = LA.eigh(Lap)

    ordered = sorted(w.T)

    sorteds = v[:, w.argsort()]

    ct = 0
    for t in ordered:

        if t > 1e-5:
            idx = ct 
            break
        ct+=1


    df = pd.DataFrame(sorteds)


    X = df.iloc[:,  idx: idx+1]



    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)

    labels = kmeans.labels_

   
    myfile = open(out_file, 'w')
    for i in range(N):
        outs = str(i) + ' ' + str(labels[i])
        myfile.write(outs)
        myfile.write('\n')


    myfile.close()

