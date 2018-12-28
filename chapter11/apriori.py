#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/28 下午10:27

from numpy import *

# Apriori算法中的辅助函数
def loadDataSet():
    return [[1,3,4], [2,3,5], [1,2,3,5], [2,5]]

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    # 对C1中每个项构建一个不变集合
    return map(frozenset, C1)

def scanD(D, Ck, minSupport):
    ssCnt = []
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can): ssCnt[can] = 1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = []
    for key in ssCnt:
        support = ssCnt[key]/numItems
        # 计算所有项集的支持度
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

# Apriori算法
def aprioriGen(Lk, k): # creates Ck
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(lenLk):
            # 前k-2个项相同时，将两个集合合并
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        # 扫描数据集，从Ck得到Lk
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

# 关联规则生成函数
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    # 只获取有两个或更多元素的集合
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item] for item in freqSet)]
            if (i>1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

def calcConf(freqSet, H, supportData, br1, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            print freqSet.conseq, '--->', conseq, 'conf:', conf
            br1.append(freqSet-conseq, conseq, conf)
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, br1, minConf=0.7):
    m = len(H[0])
    # 尝试进一步合并
    if (len(freqSet) > (m+1)):
        Hmp1 = aprioriGen(H, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minConf)
        # 创建Hm+1条新候选规则
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)



# 收集美国国会议案中action ID的函数
from time import sleep
from votesmart import votesmart

votesmart.apikey = '490124thereonecewasamanfromnantucker94040'

def getActionsIds():
    actionIdList = []
    billTitleList= []
    fr = open('recent20bills.txt')
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum)
            for action in billDetail.actions:
                if action.level == 'House' and (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print 'bill: %d has actionId: %d' % (billNum, actionId)
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print 'problem getting bill %d' % billNum
        # 为礼貌访问网站而做些延迟
        sleep(1)
    return actionIdList, billTitleList


# 基于投票数据的事务列表填充函数
def getTransList(actionIdList, billTitleList):
    itemMeaning = ['Republican', 'Democratic']
    # 填充itemMeaning列表
    for billTitle in billTitleList:
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print 'getting votes for actionId： %d' % actionId
        try:
            voteList = votesmart.votes.gitBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candiateName):
                    transDict[vote.candiateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.vote.candiateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.vote.candiateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candiateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candiateName].append(voteCount + 1)
        except:
            print 'problem getting actionId: %d' % actionId
        voteCount += 2

    return transDict, itemMeaning