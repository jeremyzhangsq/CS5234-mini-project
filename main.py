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

def takeSecond(elem):
    return elem[1]

def similarUser(matrix,query):
    q = matrix[query]
    result = []
    for row in matrix:
        result.append(pearson_def(q,row))
    return result

    # tot = 0.0
    for i_row in range (len (matrix)):
        row = matrix[i_row]
        cur = [i_row, pearson_def(q,row)]
        result.append(cur)
        # tot = tot + pearson_def(q,row)
    # print (len (result))
    # return result
    # print ("result before, ", result)
    result.sort(key=takeSecond)
    # print ("result after, ", result)
    ret = []
    for i in result:
        ret.append (i[0])
    # print (ret)
    return ret

def SimilarAvg (matrix):
    n = np.size (matrix, 0)
    m = np.size (matrix, 1)
    tot = 0.0
    for i in range (10):
        # a = np.zeros (m)
        # a[i] = 1
        now = similarUser (matrix, i)
        print (now)
        tot += now
    return tot

def pp (tweetMatrix):
    Y = np.size (tweetMatrix,0) - np.arange(np.size (tweetMatrix,0))
    estimator = ProjectionPursuitRegressor()
    X_t = estimator.fit_transform(tweetMatrix, Y)
    return X_t

def JL (tweetMatrix, x):
    transformer = random_projection.GaussianRandomProjection(n_components=x)
    # transformer = random_projection.GaussianRandomProjection()
    JL = transformer.fit_transform(tweetMatrix)
    return JL

def Kmeans(matrix):
    kmeans = KMeans(n_clusters=10, random_state=1).fit(matrix)
    return kmeans.labels_

def f1_score(truth,train):
    pass


def avg (l):
    tot = 0.0
    for i in l:
        tot = tot + i
    return tot / len (l)

def CommonAns (matrix, trans):
    ret = []
    n = np.size (matrix, 0)
    for i in range (40):
        a_list = similarUser (matrix, i)
        b_list = similarUser (trans, i)
        now = 0
        for j in range (len (a_list)):
            if a_list [j] == b_list[j]:
                now = now + 1
        ret.append (now)
    return avg(ret)

def TopKAns (matrix, trans, k = 100):
    ret = []
    n = np.size (matrix, 0)
    for i in range (40):
        a_list = similarUser (matrix, i)
        b_list = similarUser (trans, i)
        now = 0
        st = set ()
        for j in range (k):
            st.add (a_list[j])
        for j in range (k):
            if b_list[j] in st:
            # if (st.find (b_list[j])):
                now = now + 1
        ret.append (now)
    return avg(ret)

if __name__ == '__main__':
    print ("begin")
    file = './jokeRate.txt'
    tweetMatrix = read(file)

    # print (tweetMatrix.shape)
    tweetMatrix = tweetMatrix[0:1000,]
    # print (tweetMatrix.shape)

    Y = np.size (tweetMatrix,0) - np.arange (np.size (tweetMatrix,0))
    tick1 = time.time()
    Kmeans(tweetMatrix)
    # similarUser(tweetMatrix, 0)
    tick2 = time.time()
    print("without reduction:{}s".format(tick2-tick1))
    pcaMatrix = pca(tweetMatrix)
    clusters2 = Kmeans(pcaMatrix)
    print (np.size (pcaMatrix, 0))
    print (np.size (pcaMatrix, 1))
    # similarUser(pcaMatrix, 0)
    tick3 = time.time()
    print (CommonAns (tweetMatrix, pcaMatrix))
    print (TopKAns (tweetMatrix, pcaMatrix))
    print("pca reduction:{}s".format(tick3 - tick2))

    X_transformed = pp (tweetMatrix)
    print (CommonAns (tweetMatrix, X_transformed))
    print (TopKAns (tweetMatrix, X_transformed))
    tick4 = time.time()
    print("projection pursuit reduction:{}s".format(tick4 - tick3))

    for i in range (1, 99):
        tick4 = time.time()
        JL_matrix = JL (tweetMatrix, i)
        print (str (i) + "," + str (CommonAns (tweetMatrix, JL_matrix)) + "," + str ((TopKAns (tweetMatrix, JL_matrix))))
        tick5 = time.time()
        print("JL reduction:{}s".format(tick5 - tick4))