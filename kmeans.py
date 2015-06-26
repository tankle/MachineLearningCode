#-*- coding: utf-8 -*-
from numpy import *

def loadDataSet(filename):
    dataMat = []
    with open(filename, "r") as f:
        for line in f.readlines():
            l = line.strip().split("\t")
            ll = map(float,l)
            dataMat.append(ll)
    
    return dataMat
    

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

def randCent(dataset, k):
    n = shape(dataset)[1]
    centroids = mat(zeros((k,n))) #����һ��ȫ��ľ���
    
    for i in range(n):
        mini = min(dataset[:,i]) #ÿһ��ά�ȵ���Сֵ
        rangei = float(max(dataset[:,i]) - mini)
        centroids[:,i] = mini + rangei * random.rand(k,1) # ��ÿһά���������ʼ��
    
    return centroids


def kMeans(dataset, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataset)[0]
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataset,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distij = distMeas(centroids[j,:], dataset[i,:])
                if distij < minDist:
                    minDist = distij
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist**2
        print(centroids)
        for cent in range(k):
            newclust = dataset[nonzero(clusterAssment[:,0].A == cent)[0]] # nonzero ��������ά��http://blog.csdn.net/roler_/article/details/42395393
            centroids[cent,:] = mean(newclust, axis=0)
    
    return centroids, clusterAssment