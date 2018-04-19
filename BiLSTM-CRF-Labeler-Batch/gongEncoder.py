# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 08:19:12 2018

@author: zuoquan gong
"""

import torch.nn as  nn
import torch
import torch.autograd
class Encoder(nn.Module):
    def __init__(self,hyperParams,pretrain):
        
        self.embed_dim=hyperParams.embedDim
        self.embed_change=hyperParams.embedChange
        self.vocab_size=hyperParams.vocabSize
        self.hidden_size=hyperParams.hiddenSize
        self.label_size=hyperParams.labelSize
        self.dropout_prob=hyperParams.dropProb
        super(Encoder,self).__init__()
        self.embed=nn.Embedding(self.vocab_size,self.embed_dim)
        if pretrain!=None:
            pretrain_weight=torch.FloatTensor(pretrain)
            self.embed.weight.data.copy_(pretrain_weight)
            
        self.dropOut=nn.Dropout(self.dropout_prob)
        self.embed.weight.requires_grad=self.embed_change
        self.lstm=nn.LSTM(self.embed_dim,self.hidden_size//2,batch_first=True,bidirectional=True)
        self.linear=nn.Linear(self.hidden_size,self.label_size,bias=True)
        #nn.init.xavier_normal(self.linear.weight)#
        
    def init_hidden(self, batch):
       return (torch.autograd.Variable(torch.randn(2, batch, self.hidden_size//2)),
                torch.autograd.Variable(torch.randn(2, batch, self.hidden_size//2)))
       
    def forward(self,word_mat,batch=1):
#        print(word_mat.size())
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