#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/22 下午10:42

from numpy import *
from os import listdir


# k-近邻算法
def classify0(inX, dataSet, labels, k):
    """
    classify0()函数有4个输入参数：用于分类的输入向量是inX,输入的训练样本集为dataSet，标签向量为labels，最后的参数k表示用于选择最近邻居的数目，其中标签向量的元素数目和矩阵dataSet的行数相同。
    """

    # 距离计算
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    sortedDistIndicies = distances.argsort()
    classCount = {}

    # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 排序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


# 将文本记录转换为NumPy的解析程序
def file2matrix(filename):
    # 得到文件行数
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)

    # 创建返回的NumPy矩阵
    returnMat = zeros((numberOfLines, 3))

    classLabelVector = []
    index = 0

    # 解析文件数据到列表
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    return returnMat, classLabelVector


# 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))

    # 特征值相除
    normDataSet = normDataSet / tile(ranges, (m, 1))

    return normDataSet, ranges, minVals


# 分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print('the classifier came back with: %d, the real answer is: %d' % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print('the total error rate is: %f' % (errorCount / float(numTestVecs)))


# 约会网站测试函数
def classifyPersion():
    resultList = ['not at all', 'in small doses', 'in large does']
    percentTats = float(raw_input('percentage of time spent playing video games?'))
    ffMiles = float(raw_input('frequent flier miles earned per year?'))
    iceCream = float(raw_input('liters of ice cream consumed per year?'))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print 'You will probably like this persion: ', resultList[classifierResult - 1]


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


# 手写数字识别系统的测试代码
def handwritingClassTest():
    hwLabels = []

    # 获取目录内容
    trainingFileList = listdir('trainingDigits')

    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))

    # 从文件名解析分类数字
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)

    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)

    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print 'the classifier came back with: %d, the real answer is: %d' % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print '\nthe total number of errors is: %d' % errorCount
    print '\nthe total error rate is: %f' % (errorCount / float(mTest))
