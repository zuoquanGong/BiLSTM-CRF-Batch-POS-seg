# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 08:19:12 2018

@author: zuoquan gong
"""

import torch.nn as  nn
import torch
import torch.autograd
#==============================================================================
#   BiLSTM+batch
#==============================================================================
class Model(nn.Module):
    def __init__(self,hyperParams):
        self.info_name='BiLSTM-batch'
        self.info_task='POS'
        self.info_func='encode'
        
        self.embed_dim=hyperParams.embedDim
        self.embed_change=hyperParams.isEmbedChange
        self.vocab_size=hyperParams.vocabSize
        self.hidden_size=hyperParams.hiddenSize
        self.label_size=hyperParams.outputSize
        self.dropout_prob=hyperParams.dropProb
        super(Model,self).__init__()
        
        self.embed=nn.Embedding(self.vocab_size,self.embed_dim)
        pretrain=hyperParams.pretrain
        if pretrain!=None:
            pretrain_weight=torch.FloatTensor(pretrain)
            self.embed.weight.data.copy_(pretrain_weight)
            
        self.dropOut=nn.Dropout(self.dropout_prob)
        self.embed.weight.requires_grad=self.embed_change
        self.lstm=nn.LSTM(self.embed_dim,self.hidden_size//2,batch_first=True,bidirectional=True)
        self.linear=nn.Linear(self.hidden_size,self.label_size,bias=True)
        nn.init.xavier_normal(self.linear.weight)#
        
    def init_hidden(self, batch):
       return (torch.autograd.Variable(torch.randn(2, batch, self.hidden_size//2)),
                torch.autograd.Variable(torch.randn(2, batch, self.hidden_size//2)))
       
    def forward(self,data):
        batch=data.batch_size
        word_mat=data.batch_words
        senSize=word_mat.size(1)
        word_input=self.embed(word_mat)#b,s,l
        word_dropout=self.dropOut(word_input)
        lstmHidden=self.init_hidden(batch)
        lstmOutput,_=self.lstm(word_dropout.view(batch,senSize,-1),lstmHidden)#
        labelscores=self.linear(lstmOutput)
        b=labelscores.size()[0]
        s=labelscores.size()[1]
        l=labelscores.size()[2]
        
        labelscores=labelscores.view(b*s,l)
        return labelscores
        
    def show_info(self):
        print("model_name: {:^20s}| task: {:^10s}| function: {:^10s}".format(self.info_name,self.info_task,self.info_func))