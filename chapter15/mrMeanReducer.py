#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/29 下午9:15

from numpy import *

# 分布式均值和方差计算的reducer
import sys
from numpy import mat, mean, power

def read_input(file):
    for line in file:
        yield line.rstrip()

input = read_input(sys.stdin)
mapperOut = [line.split('\t') for line in input]
cumVal = 0.0
cumSumSq = 0.0
cumN = 0.0
for instance in mapperOut:
    nj = float(instance[0])
    cumN += nj
    cumVal += nj*float(instance[1])
    cumSumSq += nj*float(instance[2])

mean = cumVal/cumN
varSum = (cumSumSq - 2*mean*cumVal + cumN * mean * mean)/cumN
print '%d\t%f\t%f' % (cumN, mean, varSum)
print >> sys.stderr, 'report: still alive'

