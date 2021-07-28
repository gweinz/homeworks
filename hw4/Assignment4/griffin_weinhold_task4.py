import numpy as np 
from numpy import linalg as LA
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sys

if __name__ == '__main__':

    edge_file = sys.argv[1]
    train_file = sys.argv[2]
    test_file = sys.argv[3]
    out_file = sys.argv[4]

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
    N=len(A)
    with open(edge_file, 'r') as reader:
        for line in reader.readlines():
            vertices = line.split() 
            u = int(vertices[0])
            v = int(vertices[1])
            if u != v:
                A[u,v] = 1
                A[v,u] = 1

    D = np.zeros((max_node+1, max_node+1))

    for i in range(len(A)):

        D[i, i] = A.sum(axis=0)[i] + A.sum(axis=1)[i]


    train = []
    with open(train_file, 'r') as reader:
        for line in reader.readlines():
            row = line.split() 
            u = int(row[0])
            clust = int(row[1])
            train.append([u, clust])

    test = []
    with open(test_file, 'r') as reader:
        for line in reader.readlines():
            row = line.split() 
            u = int(row[0])
            clust = int(row[1])
            test.append([u, clust])


    train_df = pd.DataFrame(train, columns=['node', 'cluster'])


    test_df = pd.DataFrame(test, columns=['node', 'cluster'])


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

    X = df.iloc[:,  idx:idx+round(N/5)]


    embeddings = {}
    for index, row in X.iterrows():
        embeddings[index] = row.values



    train_df['embedding'] = train_df['node'].map(embeddings)

    X_train = pd.DataFrame(train_df["embedding"].to_list())

    y_train = train_df['cluster']


    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
   


    # testing now

    test_df['embedding'] = test_df['node'].map(embeddings)

    X_test = pd.DataFrame(test_df["embedding"].to_list())

    y_test = test_df['cluster']

    preds = clf.predict(X_test)
    test_df['cluster'] = preds

    myfile = open(out_file, 'w')
    for index, row in test_df.iterrows():
        outs = str(row['node']) + ' ' + str(row['cluster'])
        myfile.write(outs)
        myfile.write('\n')
    

    myfile.close()

    # check = []
    # with open('data/labels_test_truth.csv', 'r') as reader:
    #     for line in reader.readlines():
    #         row = line.split() 
    #         u = int(row[0])
    #         clust = int(row[1])
    #         check.append([u, clust])



    # check_df = pd.DataFrame(check, columns=['node', 'cluster'])
   

    # print('accuracy_score:', accuracy_score(check_df['cluster'], preds))
