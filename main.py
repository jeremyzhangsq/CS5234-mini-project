import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skpp import ProjectionPursuitRegressor
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import f1_score, precision_score, recall_score
import math
import time
from sklearn import random_projection
from sklearn.cluster import KMeans
from sklearn import metrics
import sys 


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


def pca(x, k, handout=0.1, theta=0.66):
    # pca = PCA()
    # train = x[:int(x.shape[0]*handout)]
    # pca.fit(train)
    # egenval = pca.explained_variance_ratio_
    # # plt.plot([i for i in range(len(egenval))], egenval,'o-', linewidth=2, markersize=5)
    # # plt.ylabel("Eignvalues")
    # # plt.xlabel("Constructed Dimensions")
    # # plt.savefig("pca_importance.png")
    # dim = 0
    # cnt = 0
    # for val in egenval:
    #     cnt += val
    #     dim +=1
    #     if cnt >= theta:
    #         break
    pca = PCA(n_components=k)
    decoposeX = pca.fit_transform(x)
    return decoposeX


def pcaVisual(x, handout=0.1):
    pca = PCA(n_components=3)
    train = x[:int(x.shape[0] * handout)]
    result = pca.fit_transform(train)
    x = []
    y = []
    z = []
    for each in result:
        x.append(each[0])
        y.append(each[1])
        z.append(each[2])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    plt.savefig("pca_visual.png")


def pearson_def(x, y):
    return pearsonr(x, y)[0]


def takeSecond(elem):
    return elem[1]


def similarUser(matrix, query):
    q = matrix[query]
    result = []
    # tot = 0.0
    for i_row in range(len(matrix)):
        row = matrix[i_row]
        cur = [i_row, pearson_def(q, row)]
        result.append(cur)
        # tot = tot + pearson_def(q,row)
    # print (len (result))
    # return result
    # print ("result before, ", result)
    result.sort(key=takeSecond)
    # print ("result after, ", result)
    ret = []
    for i in result:
        ret.append(i[0])
    # print (ret)
    return ret


def SimilarAvg(matrix):
    n = np.size(matrix, 0)
    m = np.size(matrix, 1)
    tot = 0.0
    for i in range(10):
        # a = np.zeros (m)
        # a[i] = 1
        now = similarUser(matrix, i)
        print(now)
        tot += now
    return tot


def pp(tweetMatrix, k):
    Y = np.size(tweetMatrix, 0) - np.arange(np.size(tweetMatrix, 0))
    estimator = ProjectionPursuitRegressor(r=k)
    X_t = estimator.fit_transform(tweetMatrix, Y)
    return X_t


def JL(tweetMatrix, x):
    transformer = random_projection.GaussianRandomProjection(n_components=x)
    # transformer = random_projection.GaussianRandomProjection()
    JL = transformer.fit_transform(tweetMatrix)
    return JL


def Kmeans(matrix):
    kmeans = KMeans(n_clusters=10, random_state=1).fit(matrix)
    return kmeans.labels_


def f1_score(truth, train):
    pass


def avg(l):
    tot = 0.0
    for i in l:
        tot = tot + i
    return tot / len(l)


def CommonAns(matrix, trans):
    ret = []
    n = np.size(matrix, 0)
    for i in range(40):
        a_list = similarUser(matrix, i)
        b_list = similarUser(trans, i)
        now = 0
        for j in range(len(a_list)):
            if a_list[j] == b_list[j]:
                now = now + 1
        ret.append(now)
    return avg(ret)


def TopKAns(matrix, trans, k=100):
    ret = []
    n = np.size(matrix, 0)
    for i in range(100):
        choice = random.randrange(0, n)
        a_list = similarUser(matrix, choice)
        b_list = similarUser(trans, choice)
        now = 0
        st = set()
        for j in range(k):
            st.add(a_list[j])
        for j in range(k):
            if b_list[j] in st:
                # if (st.find (b_list[j])):
                now = now + 1
        ret.append(now)
    return avg(ret)


def readDoc(name):
    with open(name) as f:
        N = int(f.readline().rstrip("\n"))
        D = int(f.readline().rstrip("\n"))
        C = int(f.readline().rstrip("\n"))
        matrix = np.zeros((N, D), dtype=int)
        for i in range(C):
            s = f.readline().rstrip("\n")
            s = s.split(" ")
            doc = int(s[0]) - 1
            word = int(s[1]) - 1
            cnt = int(s[2])
            matrix[doc][word] = cnt
    return matrix


def kmeans(data):
    estimator = KMeans(n_clusters=9)  # 构造聚类器
    estimator.fit(data)  # 聚类
    # print ("kmeans time : ", end - begin)
    label_pred = estimator.labels_
    # print (label_pred)
    return label_pred


if __name__ == '__main__':
    ReadJoke = sys.argv[1]
    topK = sys.argv[2]
    print("begin")
    print ('read joke ?', ReadJoke)
    print ('top k ?' ,topK)
    RunTime = []
    RunScore = []
    LessDimension = 32

    if ReadJoke == 1:
        file = './jokeRate.txt'
        TweetMatrix = read(file)
    else:
        file = './nips.txt'
        TweetMatrix = readDoc(file)


    if ReadJoke == 1:
        T_list = [100, 1000,2000,5000,10000,20000]
    else:
        T_list = [100, 500, 700, 1000, 1200, 1500]

    #T_list = [100,500]
    for t in T_list:
        CurTime = []
        CurScore = []
        print("data size =========================   ", t)
        tweetMatrix = TweetMatrix[0:t, ]
        # print (metrics.adjusted_rand_score (label, label))

        if topK == 1:
            begin = time.time()
            CurScore.append(TopKAns(tweetMatrix, tweetMatrix))
            end = time.time()
            CurTime.append(end - begin)
            # CurScore.append(1.0) # todo
            # print("no reduction:{}s".format(end - begin))
        else:
            begin = time.time()
        label = kmeans(tweetMatrix)
        end = time.time()
        CurTime.append(end - begin)
        CurScore.append(1.0)

        begin = time.time()
        pcaMatrix = pca(tweetMatrix, LessDimension)
        if topK == 1:
            CurScore.append(TopKAns(tweetMatrix, pcaMatrix))
        else:
            pcaLabel = kmeans(pcaMatrix)
        end = time.time()

        if topK == 1:
            CurTime.append(end - begin)
            # CurScore.append(1.0) // todo
        else:
            CurTime.append(end - begin)
            CurScore.append(metrics.adjusted_rand_score(label, pcaLabel))

        # print("pca reduction:{}s".format(end - begin))
        # print ("PCA score  :: ", )

        # begin = time.time ()
        # X_transformed = pp (tweetMatrix, LessDimension)
        # print (TopKAns (tweetMatrix, X_transformed))
        # end = time.time()
        # print("pp reduction:{}s".format(end - begin))
        # Xlabel = kmeans (X_transformed)
        # print (metrics.adjusted_rand_score (label, Xlabel))

        begin = time.time()
        JL_matrix = JL(tweetMatrix, LessDimension)
        if topK == 1:
            CurScore.append(TopKAns(tweetMatrix, JL_matrix))
        else:
            JL_label = kmeans(JL_matrix)
        end = time.time()

        if topK == 1:
            CurTime.append(end - begin)
            # CurScore.append(1.0) // todo
        else:
            CurTime.append(end - begin)
            CurScore.append(metrics.adjusted_rand_score(label, JL_label))

        RunTime.append(CurTime)
        RunScore.append(CurScore)

        # print("JL reduction:{}s".format(end - begin))
        # print ("JL score : ", metrics.adjusted_rand_score (label, JL_label))
    print(RunTime)
    print(RunScore)

    len = len (T_list)
    for i in range (len):
        print (T_list[i], end = " ")
        for j in RunTime[i]:
            print (j, end = " ")
        print ('\n', end = " ")
    for i in range (len):
        print (T_list[i], end = " ")
        for j in RunScore[i]:
            print (j, end = " ")
        print ('\n', end = " ")
