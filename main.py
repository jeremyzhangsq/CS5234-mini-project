import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skpp import ProjectionPursuitRegressor
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
from sklearn.metrics import f1_score,precision_score,recall_score
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


def pca(x,handout=0.1,theta=0.66):
    pca = PCA()
    train = x[:int(x.shape[0]*handout)]
    pca.fit(train)
    egenval = pca.explained_variance_ratio_
    plt.plot([i for i in range(len(egenval))], egenval,'o-', linewidth=2, markersize=5)
    plt.ylabel("Eignvalues")
    plt.xlabel("Constructed Dimensions")
    plt.savefig("pca_importance.png")
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
    return pearsonr(x,y)[0]

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
    kmeans = KMeans(n_clusters=10, random_state=1).fit(matrix)
    return kmeans.labels_

def f1_score(truth,train):
    pass


if __name__ == '__main__':
    print ("begin")
    file = './jokeRate.txt'
    tweetMatrix = read(file)

    # print (tweetMatrix.shape)
    # tweetMatrix = tweetMatrix[0:10000,]
    # print (tweetMatrix.shape)

    Y = np.size (tweetMatrix,0) - np.arange (np.size (tweetMatrix,0))
    tick1 = time.time()
    clusters = Kmeans(tweetMatrix)
    print (np.size (tweetMatrix, 0))
    print (np.size (tweetMatrix, 1))
    # similarUser(tweetMatrix, 0)
    tick2 = time.time()
    print("without reduction:{}s".format(tick2-tick1))
    pcaMatrix = pca(tweetMatrix)
    clusters2 = Kmeans(pcaMatrix)
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