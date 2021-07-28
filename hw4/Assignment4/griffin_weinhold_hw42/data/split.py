import pandas as pd
import numpy as np
import pdb

labels = pd.read_csv('email-Eu-core-department-labels.txt', header=None, sep=' ')

def train_test_split(labels):
    all_idx = np.random.permutation(len(labels))

    train = sorted(all_idx[:800])
    test = sorted(all_idx[800:])
    return labels.iloc[train], labels.iloc[test]

labels_train, labels_test = train_test_split(labels)
labels_train.to_csv('labels_train.csv', index=False, header=None, sep=' ')
labels_test.to_csv('labels_test_truth.csv', index=False, header=None, sep=' ')
e1 = set(labels_train.loc[:, 1])
print('#labels', len(e1))
e2 = set(labels_test.loc[:, 1])
print('#labels', len(e2))


labels_test.loc[:, 1] = 0
labels_test.to_csv('labels_test.csv', index=False, header=None, sep=' ')
