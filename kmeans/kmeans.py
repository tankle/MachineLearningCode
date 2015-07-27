#-*- coding: utf-8 -*-
from numpy import *

def loadDataSet(filename):
    '''
    ÿһ��Ϊһ��ѵ�����ݣ���\t����
    '''
    dataMat = []
    with open(filename, "r") as f:
        for line in f.readlines():
            l = line.strip().split("\t")
            ll = map(float,l) # ��l�е�ÿ��Ԫ��ת��Ϊfloat��ʽ
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
        print("the centroids is ",centroids)
        for cent in range(k):
            newclust = dataset[nonzero(clusterAssment[:,0].A == cent)[0]] # nonzero ��������ά��, ��������Ӧά���Ϸ���Ԫ�ص�Ŀ¼ֵ    , [0] ��ʾ�к�����http://blog.csdn.net/roler_/article/details/42395393
                                                                          # mat.A �����ת��Ϊarray��ʽ
            centroids[cent,:] = mean(newclust, axis=0)
    
    return centroids, clusterAssment

    
    
    
def biKmeans(dataset, k, distMeas=distEclud):
    '''
    ����k-means
    ����SSE�����ƽ���ͣ���Сԭ�򣬲�ͣ���зִأ�ֱ���ﵽK��
    '''
    m = shape(dataset)[0] # ѵ�����ݵĸ���
    clusterAssment = mat(zeros((m,2))) 
    centroid0 = mean(dataset, axis=0).tolist()[0] # ת��Ϊһ������
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0), dataset[j,:])**2 #�������ʱ����Ԫ�صĸ�ʽ��Ҫһ��
    
    while (len(centList) < k):
        lowerSSE = inf
        for i in range(len(centList)):
            ptsCurCluster = dataset[nonzero(clusterAssment[:,0].A == i)[0],:] #����������ڵ�i�صĵ�
            centroidMat, splitCulstAss = kMeans(ptsCurCluster, 2, distMeas) # �ֳ�����
            
            sseSplit = sum(splitCulstAss[:,1]) #��øô��зֺ��SSEֵ
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0],1]) # �������û�б��зֵ�SSEֵ
            
            print("sseSplit, and sseNotSplit : %f \t %f " %(sseSplit, sseNotSplit))
            # ѡ��SSE��С��
            if (sseSplit + sseNotSplit) < lowerSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitCulstAss.copy()
                lowerSSE = sseSplit + sseNotSplit
            
        # ���´صķ�����
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) # ������1 �Ĵظ�ֵΪ��ǰcentList�ĳ���
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit # ������0 �Ĵػ��Ǳ���Ϊԭ���Ĵص�ֵ
        
        print("The bestCentToSplit is: ", bestCentToSplit)
        print("The len of bestClustAss is: ", len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0] #���´������б�
        centList.append(bestNewCents[1,:].tolist()[0]) # �����µĴ�����
        
        # ����ÿ���㵽���ĵľ���
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0], :] = bestClustAss
        
    
    return mat(centList), clusterAssment



if __name__== "__main__":
    pass