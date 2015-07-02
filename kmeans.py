#-*- coding: utf-8 -*-
from numpy import *

def loadDataSet(filename):
    '''
    每一行为一个训练数据，以\t隔开
    '''
    dataMat = []
    with open(filename, "r") as f:
        for line in f.readlines():
            l = line.strip().split("\t")
            ll = map(float,l) # 将l中的每个元素转换为float格式
            dataMat.append(ll)
    
    return dataMat
    

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

def randCent(dataset, k):
    n = shape(dataset)[1]
    centroids = mat(zeros((k,n))) #创建一个全零的矩阵
    
    for i in range(n):
        mini = min(dataset[:,i]) #每一个维度的最小值
        rangei = float(max(dataset[:,i]) - mini)
        centroids[:,i] = mini + rangei * random.rand(k,1) # 对每一维进行随机初始化
    
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
        print("the centroids is ",centroids)
        for cent in range(k):
            newclust = dataset[nonzero(clusterAssment[:,0].A == cent)[0]] # nonzero 返回两个维度, 包含了相应维度上非零元素的目录值    , [0] 表示行号数组http://blog.csdn.net/roler_/article/details/42395393
                                                                          # mat.A 将结果转换为array格式
            centroids[cent,:] = mean(newclust, axis=0)
    
    return centroids, clusterAssment

    
    
    
def biKmeans(dataset, k, distMeas=distEclud):
    '''
    二分k-means
    根据SSE（误差平方和）最小原则，不停的切分簇，直到达到K簇
    '''
    m = shape(dataset)[0] # 训练数据的个数
    clusterAssment = mat(zeros((m,2))) 
    centroid0 = mean(dataset, axis=0).tolist()[0] # 转换为一个数组
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0), dataset[j,:])**2 #计算距离时两个元素的格式需要一样
    
    while (len(centList) < k):
        lowerSSE = inf
        for i in range(len(centList)):
            ptsCurCluster = dataset[nonzero(clusterAssment[:,0].A == i)[0],:] #获得所有属于第i簇的点
            centroidMat, splitCulstAss = kMeans(ptsCurCluster, 2, distMeas) # 分成两簇
            
            sseSplit = sum(splitCulstAss[:,1]) #获得该簇切分后的SSE值
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0],1]) # 获得其他没有被切分的SSE值
            
            print("sseSplit, and sseNotSplit : %f \t %f " %(sseSplit, sseNotSplit))
            # 选择SSE最小的
            if (sseSplit + sseNotSplit) < lowerSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitCulstAss.copy()
                lowerSSE = sseSplit + sseNotSplit
            
        # 更新簇的分配结果
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) # 将等于1 的簇赋值为当前centList的长度
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit # 将等于0 的簇还是保持为原来的簇的值
        
        print("The bestCentToSplit is: ", bestCentToSplit)
        print("The len of bestClustAss is: ", len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0] #更新簇中心列表
        centList.append(bestNewCents[1,:].tolist()[0]) # 增加新的簇中心
        
        # 更新每个点到质心的距离
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0], :] = bestClustAss
        
    
    return mat(centList), clusterAssment



if __name__== "__main__":
    pass