import numpy as np 
from numpy import linalg as LA
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import json
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import sys 


if __name__ == '__main__':

    edge_file = sys.argv[1]
    out_file = sys.argv[2]

    data = pd.read_json(edge_file)


    A = kneighbors_graph(data, 50, mode='connectivity', include_self=False).toarray()

    range_ = len(A)

    D = np.zeros((range_, range_))

    for i in range(range_):

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

    X = df.iloc[:,  idx: idx+2]


    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

    labels = kmeans.labels_

    data['cluster'] = labels


    x = data[0]
    y = data[1]
    colors = data['cluster']


    plt.scatter(x, y, c=colors)
    plt.title('Task 5 Partition')
    plt.savefig(out_file)

