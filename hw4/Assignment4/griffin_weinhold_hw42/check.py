from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    true3 = []
    with open('data/email-Eu-core-department-labels.txt', 'r') as reader:
        for line in reader.readlines():
            vertices = line.split() 
            u = int(vertices[0])
            v = int(vertices[1])


            true3.append(v)

    test3 = []
    with open('test3.txt', 'r') as reader:
        for line in reader.readlines():
            vertices = line.split() 
            u = int(vertices[0])
            v = int(vertices[1])


            test3.append(v)

    print(adjusted_rand_score(true3, test3))
    true4 = []
    with open('data/labels_test_truth.csv', 'r') as reader:
        for line in reader.readlines():
            vertices = line.split() 
            u = int(vertices[0])
            v = int(vertices[1])


            true4.append(v)

    test4 = []
    with open('test4.txt', 'r') as reader:
        for line in reader.readlines():
            vertices = line.split() 
            u = int(vertices[0])
            v = int(vertices[1])


            test4.append(v)

    print(accuracy_score(true4, test4))