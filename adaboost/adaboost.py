#-*- coding:  utf-8 -*-

from numpy import *

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    '''
    对数据进行分类，一边分为+1，一边分为-1
    dataMatrix: 输入训练矩阵
    dimen： 用于检测的特征维度
    threshVal：阈值
    threshIneq：测试类型
    '''
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr,labels,D):
    '''
    单层决策桩函数，D为权重数组
    '''
    dataMatrix = mat(dataArr)
    labelMat = mat(labels).T

    m,n = shape(dataMatrix)
    bestStump = {}
    bestClassEst = mat(zeros((m,1)))
    minError = inf
    numSteps = 10.0

    for i in range(n): #对每一维特征
        rangMin = dataMatrix[:,i].min()
        rangMax = dataMatrix[:,i].max()
        stepSize = (rangMax-rangMin)/numSteps
        for j in range(-1,int(numSteps)+1):#步长
            for inequal in ['lt','gt']:
                threshVal = (rangMin + float(j)*stepSize)
                predictVal = stumpClassify(dataMatrix,i,threshVal,inequal) #决策桩结果预测

                errArr = mat(ones((m,1))) #默认错误率为1
                errArr[predictVal == labelMat] = 0 #预测对的错误率为0

                weightedError = D.T*errArr #加权0/1错误率

                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictVal.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal

    #返回三个参数，第一个为最佳分割点，第二个为最小误差，第三个为预测值
    return bestStump,minError,bestClassEst



def adaboostTrain(dataArr, labels, iterations):
    '''
    ieterations: 迭代次数

    返回一个若分类器数组
    '''
    weakArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m) # 初始化权值w
    aggClassEst = mat(zeros((m,1))) #累加误差

    for i in range(iterations):
        bestStump, error, classEst = buildStump(dataArr,labels,D)
        print("D is ",D.T)

        alpha = float(0.5*log((1-error)/max(error,1e-16))) #计算G（x)的权值alpha
        bestStump['alpha'] = alpha
        weakArr.append(bestStump)

        print("classEst is ",classEst.T)

        expon = multiply(-1*alpha*mat(labels).T, classEst) # 计算指数部分
        D = multiply(D,exp(expon))
        D = D/D.sum() #更新权值w数组D

        aggClassEst += alpha*classEst
        print("aggClassEst is ", aggClassEst)

        aggErrors = multiply(sign(aggClassEst) != mat(labels).T, ones((m,1))) #预测值和真实值相等为0， 不相等为1， 

        errorRate = aggErrors.sum()/m
        print("Total error:", errorRate)

        if errorRate - 0.0 < 1e-20:
            break
    return weakArr


def adaboostPredict(dataArr, classifierArr):
    dataMat = mat(dataArr)
    m = shape(dataMat)[0]
    aggClassEst = mat(zeros((m,1)))

    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMat, classifierArr[i]['dim'],\
            classifierArr[i]['thresh'],classifierArr[i]['ineq'])

        aggClassEst += classifierArr[i]['alpha'] * classEst

        print("aggclassEst is ", aggClassEst)

    return sign(aggClassEst)


if __name__ == "__main__":
    #《统计学习方法》例子
    data = [[i] for i in range(10)]
    labels = [1,1,1,-1,-1,-1,1,1,1,-1]

    test = [[1.5], [3.4],[10]]
    test_y = [1,-1,-1]

    weakarr = adaboostTrain(data,labels,40)
    print(weakarr)
    print("alpha is ", [w['alpha'] for w in weakarr])

    predict = adaboostPredict(test,weakarr)

    print("predict result is ")
    print(predict)

