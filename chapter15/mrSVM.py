#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/29 下午9:34

from numpy import *

# SVM的Pegasos算法
def predict(w, x):
    return w*x.T

def batchPegasos(dataSet, labels, lam, T, k):
    m, n = shape(dataSet)
    w = zeros(n)
    dataIndex = range(m)
    for t in range(1, T+1):
        wDelta = mat(zeros(n))
        eta = 1.0/(lam*t)
        random.shuffle(dataIndex)
        for j in range(k):
            i = dataIndex[j]
            p = predict(w, dataSet[i,:])
            if labels[i]*p < 1:
                wDelta += labels[i]*dataSet[i,:].A
        # 将待更新值累加
        w = (1.0 - 1/t)*w + (eta/k)*wDelta

    return w



