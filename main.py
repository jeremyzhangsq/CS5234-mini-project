
import numpy as np
from sklearn.decomposition import PCA

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



def pca(x,k):
    pca = PCA(n_components=k)
    pca.fit(x)
    decoposeX = pca.transform(x)
    return decoposeX

if __name__ == '__main__':
    file = './jokeRate.txt'
    tweetMatrix = read(file)
    pcaMatrix = pca(tweetMatrix,10)