import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skpp import ProjectionPursuitRegressor
from sklearn.cluster import KMeans
import math
import time
from sklearn import random_projection

def read(name):

    matrix = []
    with open(name, encoding="utf8", errors='ignore') as f:
        for line in f.readlines():
            s = line.rstrip("\n")
            s = s.split("\t")
            l = []
            for i in range(len(s) - 1):
                val = float(s[i + 1])
                if val == 99:
                    val = 0
                l.append(val)
            matrix.append(l)
    return np.array(matrix)


def pca(x,handout=0.1,theta=0.70):
    pca = PCA()
    train = x[:int(x.shape[0]*handout)]
    pca.fit(train)
    egenval = pca.explained_variance_ratio_
    plt.plot([i for i in range(len(egenval))], egenval,'o-', linewidth=2, markersize=5)
    plt.show()
    dim = 0
    cnt = 0
    for val in egenval:
        cnt += val
        dim +=1
        if cnt >= theta:
            break
    pca = PCA(n_components=dim)
    decoposeX = pca.fit_transform(x)
    return decoposeX

def pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = float(sum(x)) / len(x)
    avg_y = float(sum(y)) / len(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff
    return diffprod / math.sqrt(xdiff2 * ydiff2)

def similarUser(matrix,query):
    q = matrix[query]
    result = []
    for row in matrix:
        result.append(pearson_def(q,row))
    return result
def pp (tweetMatrix):
    Y = np.size (tweetMatrix,0) - np.arange(np.size (tweetMatrix,0))
    estimator = ProjectionPursuitRegressor()
    X_t = estimator.fit_transform(tweetMatrix, Y)
    return X_t
def JL (tweetMatrix):
    transformer = random_projection.GaussianRandomProjection(n_components=37)
    JL = transformer.fit_transform(tweetMatrix)
    return JL

def Kmeans(matrix):
    kmeans = KMeans(n_clusters=10).fit(matrix)
    return kmeans.cluster_centers_

if __name__ == '__main__':
    print ("begin")
    file = './jokeRate.txt'
    tweetMatrix = read(file)

    # print (tweetMatrix.shape)
    # tweetMatrix = tweetMatrix[0:10000,]
    # print (tweetMatrix.shape)

    Y = np.size (tweetMatrix,0) - np.arange (np.size (tweetMatrix,0))
    tick1 = time.time()
    Kmeans(tweetMatrix)
    print (np.size (tweetMatrix, 0))
    print (np.size (tweetMatrix, 1))
    # similarUser(tweetMatrix, 0)
    tick2 = time.time()
    print("without reduction:{}s".format(tick2-tick1))

    pcaMatrix = pca(tweetMatrix, 10)
    Kmeans(pcaMatrix)
    print (np.size (pcaMatrix, 0))
    print (np.size (pcaMatrix, 1))
    # similarUser(pcaMatrix, 0)
    tick3 = time.time()
    print("pca reduction:{}s".format(tick3 - tick2))

    X_transformed = pp (tweetMatrix)
    Kmeans(X_transformed)
    tick4 = time.time()
    print("projection pursuit reduction:{}s".format(tick4 - tick3))

    JL_matrix = JL (tweetMatrix)
    Kmeans(JL_matrix)
    tick5 = time.time()
    print("JL reduction:{}s".format(tick5 - tick4))