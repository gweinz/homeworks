import numpy as np 
from numpy import linalg as LA
import networkx as nx

D = np.zeros((12,12))
A = np.zeros((12,12))
degrees = {1: 1, 2:1, 3:4, 4:1, 5:3, 6:1, 7:1, 8:4, 9:1, 10:3, 11:1, 12:1}
neighbors = {1: [3], 2:[3], 3:[1,2,5,8], 4:[5], 5:[3,4,6], 6:[5], 7:[8], 8:[3,7,9,10], 9:[8], 10:[8,11,12], 11:[10], 12:[10]}
for d in degrees:
    D[d-1, d-1] = degrees[d]


for n in neighbors:
    for v in neighbors[n]:
        A[n-1][v-1] = 1



Lap = D-A



w, v = LA.eig(Lap)
idx = np.argsort(w)
sorteds = v[:, w.argsort()]
for l in Lap:
    result2 = [str(int(x)) for x in l]
    output = (',').join(result2)
    print(output)

sec = [str(x) for x in sorteds[:,1]]

print((',').join(sec))

