#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/28 下午9:25

from numpy import *
from Tkinter import *
from . import regTrees

# 用于构建树管理器界面的Tinker小部件
def reDraw(tolS, tolN):
    pass

def drawNewTree():
    pass

root = Tk()

Label(root, text='Plot Place Holder').grid(row=0, columnspan=3)
Label(root, text='tolN').grid(row=1, column=0)
tolNertry = Entry(root)
tolNertry.grid(row=1, column=1)
tolNertry.insert(0, '10')
Label(root, text='tolS').grid(row=2, column=0)
tolSentry = Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, '1.0')
Button(root, text='ReDraw', command=drawNewTree).grid(row=1, column=2, rowspan=3)
chkBtnVar = IntVar()
chkBtn = Checkbutton(root, text='Model Tree', variable=chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)
reDraw.rawDat = mat(regTrees.loadDataSet('sine.txt'))
reDraw(1.0, 10)
root.mainloop()

# Matplotlib和Tkinter的代码集成
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def reDraw(tolS, tolN):
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    # 检查复选框是否选中
    if chkBtnVar.get():
        if tolN < 3: tolN = 2
        myTree = regTrees.createTree(reDraw.rawDat, regTrees.modelLeaf, regTrees.modelErr, (tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat, regTrees.modelTreeEval)
    else:
        myTree = regTrees.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.rawDat[:,1], s=5)
    reDraw.a.scallter(reDraw.rawDat[:,0], reDraw.rawDat[:,1], s=5)
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)
    reDraw.canvas.show()

def getInputs():
    try: tolN = int(tolNertry.get())
    except:
        tolN = 10
        print 'enter Integer for tolN'
        tolNertry.delete(0, END)
        tolNertry.insert(0, '10')
    try: tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print 'enter Float for tolS'
        tolSentry.delete(0, END)
        tolSentry.insert(0,'1.0')
    return tolN, tolS

def drawNewTree():
    tolN, tolS = getInputs()
    reDraw(tolS, tolN)

