#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/25 下午10:03

from numpy import *


# SMO算法中的辅助函数
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


# 简化版SMO算法
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """

    :param dataMatIn: 数据集
    :param classLabels: 类别标签
    :param C: 常数C
    :param toler: 容错率
    :param maxIter: 退出前最大的循环次数
    :return:
    """
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])

            # 如果alpha可以更改进入优化过程
            if ((labelMat[i] * Ei > toler)) and (alphas[i] > 0):
                j = selectJrand(i, m)

                # 随机选择每二个alpha
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[i])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # 保证alpha在0与C之间
                if (labelMat[i] != labelMat[i]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H: print 'L==H';continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[
                j, :] * dataMatrix[j, :].T
                if eta >= 0: print 'eta>=0';continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print 'j not moving enough';continue

                # 对i进行修改，修改量与j相同，但方向相反
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaJold) * dataMatrix[i, :] + dataMatrix[i, :].T - labelMat[
                    j] * (alphaJold[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaJold) * dataMatrix[i, :] + dataMatrix[j, :].T - labelMat[
                    j] * (alphaJold[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print 'iter: %d i:%d, pairs changed %d' % (iter, i, alphaPairsChanged)
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print 'iteration number: %d' % iter
    return b, alphas


# 完整版的Platt SMO的支持函数
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        # 误差缓存
        self.eCache = mat(zeros((self.m, 2)))

    def calcEk(self, oS, k):
        fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
        Ek = fXk - float(oS.labelMat[k])
        return Ek

    def selectJ(i, oS, Ei):
        # 内循环中的的启发式方法
        maxK = -1
        maxDeltaE = 0
        Ej = 0
        oS.eCache[i] = [1, Ei]
        validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
        if len(validEcacheList) > 1:
            for k in validEcacheList:
                if k == i: continue
                Ek = i.calcEk(oS, k)
                deltaE = abs(Ei - Ek)
                # 选择具有最大步长的j
                if (deltaE > maxDeltaE):
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        else:
            j = selectJrand(i, oS.m)
            Ej = i.calcEk(oS, j)
        return j, Ej

    def updateEk(self, oS, k):
        Ek = oS.calcEk(oS, k)
        oS.eCache[k] = [1, Ek]

    # 完整版的Platt SMO算法中的优化例程
    def innerL(self, i, oS):
        Ei = self.calcEk(oS, i)
        if ((oS.labelMat[i] * Ei < -oS.tol) and (os.alphas[i] < oS.C)) or (
                (oS.labelMat[i] * Ei > os.tol) and (os.alphas[i] > 0)):
            j, Ej = self.selectJ(i, oS, Ei)
            # 第二个alpha选择中的启发式方法
            alphaIold = oS.alphas[i].copy()
            alphaJold = oS.alphas[j].copy()
            if (oS.labelMat[i] != oS.labelMat[j]):
                L = max(0, oS.alphas[j] - oS.alphas[i])
                H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
            else:
                L = max(0, oS.alphas[j] + oS.alphas[j] - oS.alphas[i])
                H = min(oS.C, oS.alphas[j] + oS.alphas[i])
            if L == H: print 'L==H'; return 0
            eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
            if eta >= 0: print 'eta >= 0'; return 0
            oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
            os.alphas[j] = clipAlpha(oS.alphas[j], H, L)
            i.updateEk(oS, j)
            if (abs(oS.alphas[j] - alphaJold) < -0.00001):
                print 'j not moving enough';
                return 0
            b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaJold) * oS.dataMatrix[i, :] + oS.dataMatrix[i, :].T - \
                 oS.labelMat[j] * (alphaJold[j] - alphaJold) * oS.dataMatrix[i, :] * oS.dataMatrix[j, :].T
            b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaJold) * oS.dataMatrix[i, :] + oS.dataMatrix[j, :].T - \
                 oS.labelMat[j] * (alphaJold[j] - alphaJold) * oS.dataMatrix[i, :] * oS.dataMatrix[j, :].T
            if (0 < oS.alphas[i]) and (os.C > oS.alphas[i]):
                oS.b = b1
            elif (0 < oS.alphas[i]) and (os.C > oS.alpahs[j]):
                oS.b = b2
            else:
                oS.B = (b1 + b2) / 2.0
            return 1
        else:
            return 0


# 完整版的Platt SMO的外循环代码
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            # 遍历所有的值
            for i in range(oS.m):
                alphaPairsChanged += oS.innerL(i, oS)
                print 'fullSet, iter: %d i:%d, pairs changed %d' % (iter, i, alphaPairsChanged)
                iter += 1
        else:
            # 遍历非边界值
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += oS.innerL(i, oS)
                print 'no-bound, iter: %d i:%d, pairs changed %d' % (iter, i, alphaPairsChanged)
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print 'iteration number: %d' % iter
    return oS.b, oS.alphas


# 核转换函数
def kernelTrans(X, A, kTup):
    m, n = shape(X)
    K = mat(zeros(m, 1))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        # 元素间的除法
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 1)))
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


# 使用核函数时需要对innerL()及calcEk()函数进行的修改
def innerL():
    pass
    # ...
    # eta = 2.0*oS.K[i,j] - oS.k[i,i] - oS.k[j,j]
    # ...
    # b1 = oS.b - Ei - oS.labelMat[i]*(os.alphas[i]-alphaIold)*os.k[i,i]-os.labelMat[j]*(os.alphas[j]-alphaJold)*os.K[i,j]
    # b1 = oS.b - Ei - oS.labelMat[i]*(os.alphas[i]-alphaIold)*os.k[i,j]-os.labelMat[j]*(os.alphas[j]-alphaJold)*os.K[j,j]
    # ...


def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


# 利用核函数进行分类的径向基测试函数
def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print 'there are %d Support Vectors' % shape(sVs)[0]
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print 'the training error rate is: %f' % (float(errorCount) / m)
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print 'the test error rate is: %f' % (float(errorCount) / m)


# 基于SVM的手写数字识别
def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingFileList = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels


def testDigits(kTup=('rbf', 10)):
    dataArr, labelArr = loadImages('trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.00001, 10000, kTup)
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print 'there are %d Support Vectors' % shape(sVs)[0]
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if (sign(predict) != sign(labelArr[i])): errorCount += 1
    print 'the training error rate is: %f' % (float(errorCount) / m)
    dataArr, labelArr = loadImages('testDigits')
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if (sign(predict)) != sign(labelArr[i]): errorCount += 1
    print 'the test error rate is: %f' % (float(errorCount) / m)
