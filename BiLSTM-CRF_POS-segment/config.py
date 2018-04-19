# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:18:09 2018

@author: zuoquan gong
"""
import os
import sys
#==============================================================================
#     一、process模块参数
#==============================================================================
class ProcessParams:

    def __init__(self):
        
        #1.用于提取信息
        self.path=os.getcwd().replace('\\','/')+'/data/'#指定的数据文件存储路径
        self.trainFile = 'zh-ud-train.conllu-seg'#训练集所在文件**
        self.devFile = 'zh-ud-dev.conllu-seg'#开发集所在文件**
        self.testFile = 'zh-ud-test.conllu-seg'#测试集所在文件**
        
        self.cutMode = 0#切分方式： 0--垂直切割   1--横向切割   2--序列标注类型切割
        self.wordIndex = 1#特征（词）所在位置
        self.labelIndex = 3#标签所在位置
        
        #2.用于切分句子
        self.separator = '\t'#主分割符
        self.miniCut = ''#细分割符
        self.cutOut = '#'#滤去原始文本中以cutOut开头的行
        
        #3.用于词表裁剪
        self.cutOff = 0#按词频裁剪
        self.vocabSize = None#按词表大小裁剪 | 词表大小
        self.labelSize = None#标签表大小
        
        #4.用于保存词表
        self.save_word_mapping='save_mapping/word_mapping.save'
        self.save_label_mapping='save_mapping/label_mapping.save'
        
#==============================================================================
#     二、model模块参数
#==============================================================================
class ModelParams:

    def __init__(self):
        
        #1.模型（用于模型载入函数--model_loader）
        self.modelPath=os.getcwd().replace('\\','/')+'/train_eval/model/'
        self.modelNum = 2#载入的模型数**
        #self.modelList = ['BiLSTMBatch','CRFBatch']#模型名称列表（按构造顺序）**
        self.modelList = []
        self.modelDefaultList = [f[:-3] for f in os.listdir(self.modelPath) if f!='modelloader.py' and f[0]!='_']
        #注：第一个模型必须提供embedding接口，最后一个模型必须提供预测函数
        
        if self.modelList == []:
            self.modelList=self.modelDefaultList#加载默认模型
        
        #2.预处理
        self.pretrain=None
        self.vocabSize=0#**
        self.embedDim = 100#词向量维数
        self.isEmbedChange = True#词向量是否随着训练而更新
        self.embedFile = ""#词向量所在文件
        
        #3.模型参数
        self.dropProb = 0.5#dropout概率
        self.hiddenSize = 50#隐层大小
        self.outputSize = 15#**
        
        #4.模型加载与保存
        self.savePath = os.getcwd().replace('\\','/')+'/model_save/'
        self.saveFile = [name+'.pkl' for name in self.modelList]
        self.loadDefaultPath = os.getcwd().replace('\\','/')+'/model_save/'
#        self.loadDefaultFile = [f for f in os.listdir(self.loadDefaultPath)]
        self.log_clear=False
#        self.loadFile=[]
        self.loadFile=['']*self.modelNum

        if '.pkl' in [file[-4:] for file in os.listdir(self.savePath)] and self.loadFile==['']*self.modelNum:
            change=input('warning: Do you want to replace the old saved files and start a new train ? [y/n]: ')
            while change!='':
                if change=='n':
                    self.loadFile=[]
                    break
                elif change=='y':
                    self.log_clear=True
                    break
                else:
                    change=input('Input Error.Please input your choice again: ')
            
        if self.loadFile==[]:
            self.loadFile=[self.loadDefaultPath+f for f in os.listdir(self.loadDefaultPath) if f[-4:]=='.pkl']

        #5.模型batch填充符，开始符
        self.padId = 0
        self.startId=0
        
        #6.优化器参数（optimizer）
        self.learningRate = 0.001#学习率
#==============================================================================
#     三、train&eval模块参数
#==============================================================================
class TrainEvalParams:

    def __init__(self):
        
        #1.训练参数
        self.maxIter = 500#迭代次数
        self.showIter = 10#设定显示训练数据的迭代数
        self.thread = 1#**
        self.hasBatch = True#**
        
        #2.批处理（batch）参数
        self.maxSentSize = 80#最大句长
        self.batch_size = 40#每批次大小
        
        self.threshold = 0.0001
    
#==============================================================================
#     总、集成的参数类
#==============================================================================
class Params:

    def __init__(self):
        self.process_params = ProcessParams()
        self.model_params = ModelParams()
        self.train_eval_params = TrainEvalParams()
        
    def __show__(self):
        #__dict__函数可以获得一个dict，形式为{ 实例的属性 ：属性值 }
        print('\n[  Process Parameters  ]\n')
        for item in self.process_params.__dict__.items():
            print('{}: \'{}\''.format(item[0],item[1]))
        print('\n[  Model Parameters  ]\n')
        for item in self.model_params.__dict__.items():
            print('{}: \'{}\''.format(item[0],item[1]))
        print('\n[  Train&Eval Parameters  ]\n')
        for item in self.train_eval_params.__dict__.items():
            print('{}: \'{}\''.format(item[0],item[1]))
















