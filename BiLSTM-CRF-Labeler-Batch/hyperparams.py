# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 21:21:47 2018

@author: zuoquan gong
"""

class HyperParams:
    def __init__(self):
        self.vocabSize = 0##
        self.labelSize = 0##

#        self.unk = '-unk-'
#        self.padding = '-padding-'
#        self.unkId = 0
#        self.paddingId = 0

        self.maxIter = 500
        self.verboseIter = 10#设定显示训练数据的迭代数
        self.cutOff = 0
        self.embedDim = 100
        self.embedChange = True#
        #self.wordEmbFile = "E:\\py_workspace\\my_rnn_crf\\data\\glove.twitter.27B.100d.txt"
        self.embedFile = ""#
        self.dropProb = 0.5
        self.hiddenSize = 50
        self.thread = 1#
        self.learningRate = 0.001
        self.maxSentSize = 60#
        self.batch = 40#

        self.cutMode=0
        self.wordIndex=1
        self.labelIndex=3
        self.separator='\t'
        self.miniCut=''
        self.cutOut='#'#滤去原始文本中开头为‘#’的行
        self.cutOff=0
        self.vocabSize=None
        
    def show(self):
        print('cut_off = ', self.cutOff)
        print('embed_size = ', self.embedDim)
        print('embed_change = ', self.embedChange)
        print('hidden_size = ', self.hiddenSize)
        print('learning_rate = ', self.learningRate)
        print('batch = ', self.batch)

        print('max_instance = ', self.maxSentSize)
        print('max_iter =', self.maxIter)
        print('thread = ', self.thread)
        print('verboseIter = ', self.verboseIter)

