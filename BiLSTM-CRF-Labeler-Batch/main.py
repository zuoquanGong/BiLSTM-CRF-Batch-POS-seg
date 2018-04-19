# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 20:35:25 2018

@author: zuoquan gong
"""
from optparse import OptionParser 
from dataprocess import Alphabet,DataSet,IndexSet,PretrainEmb,BatchBucket
#from xdataprocess import Alphabet,DataSet,IndexSet,PretrainEmb,BatchBucket
import random
import  torch
import torch.nn
import torch.autograd
import torch.nn.functional
from hyperparams import HyperParams
from gongEncoder import Encoder
from gongCRF import CRF
random.seed(1)
class Labeler:
    def __init__(self):
        self.hyperParams=HyperParams()
        pass
    def train(self,trainFile,devFile,testFile):
        self.hyperParams.show()
        train=DataSet(trainFile)
        dev=DataSet(devFile)
        test=DataSet(testFile)
        
        vocab=Alphabet(train)
        self.hyperParams.vocabSize=vocab.size()
        self.hyperParams.labelSize=vocab.label_size()
        print('vocab_size:',self.hyperParams.vocabSize)
        print('label_size:',self.hyperParams.labelSize)
        
        train=IndexSet(train,vocab)
        dev=IndexSet(dev,vocab)
        test=IndexSet(test,vocab)
        
        print('trainset_size: ',train.size())
        print('devset_size: ',dev.size())
        print('testset_size: ',test.size())
        if self.hyperParams.embedFile!='':
            pretrain=PretrainEmb(self.hyperParams.embedFile,vocab.word2id)
        else:
            pretrain=None
            
        ##############################            
        self.model = Encoder(self.hyperParams,pretrain) #encoder
        self.crf = CRF(self.hyperParams.labelSize,vocab.label2id['<start>'],vocab.label2id['<padding>'],vocab)#decoder
        
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        
        optimizer_rnn = torch.optim.Adam(parameters, lr = self.hyperParams.learningRate)
        optimizer_crf = torch.optim.Adam(self.crf.parameters(), lr = self.hyperParams.learningRate)
        ##############################
        
        indexes = []
        for idx in range(train.size()):
            indexes.append(idx)
    
        batchBlock = len(train.word_mat) // self.hyperParams.batch
        for iter in range(self.hyperParams.maxIter):#################
            print('###Iteration' + str(iter) + "###")
            random.shuffle(indexes)
            
            self.model.train()###
            
            for updateIter in range(batchBlock):
                #self.model.zero_grad()
                optimizer_rnn.zero_grad()
                optimizer_crf.zero_grad()
    
                start_pos = updateIter * self.hyperParams.batch
                end_pos = (updateIter + 1) * self.hyperParams.batch
                feats=[]
                labels=[]
                for idx in range(start_pos, end_pos):
                    feats.append(train.word_mat[indexes[idx]])
                    labels.append(train.label_mat[indexes[idx]])
                batch=BatchBucket(len(feats),self.hyperParams.maxSentSize,feats,labels,vocab.word2id['<padding>'],vocab.label2id['<padding>'])
                tag_scores = self.model(batch.batch_words, self.hyperParams.batch)
                #print(tag_scores.size())
                loss = self.crf.neg_log_likelihood(tag_scores, batch.batch_labels,batch.masks)
                loss.backward()
    
                optimizer_rnn.step()
                optimizer_crf.step()
    
                if (updateIter + 1) % self.hyperParams.verboseIter == 0:
                    print('current: ', idx + 1, ", cost:", loss.data[0])
    
            self.model.eval()###
            self.eval_predict(dev,vocab)
            self.eval_predict(test,vocab)
            
    def eval_predict(self,indexset,vocab) :
        correct_num=0
        total_num=0
        batchBlock=len(indexset.label_mat)//self.hyperParams.batch
        for updateIter in range(batchBlock):
           
            start_pos = updateIter * self.hyperParams.batch
            end_pos = (updateIter + 1) * self.hyperParams.batch
            feats=[]
            labels=[]
            for idx in range(start_pos, end_pos):
                feats.append(indexset.word_mat[idx])
                labels.append(indexset.label_mat[idx])
            batch=BatchBucket(len(feats),self.hyperParams.maxSentSize,feats,labels,vocab.word2id['<padding>'],vocab.label2id['<padding>'])
            tag_scores = self.model(batch.batch_words, self.hyperParams.batch)
            predict_labels=self.crf.viterbi_decode(tag_scores,batch.masks)
            predict_labels=predict_labels.masked_select(batch.masks)
            gold_labels=batch.batch_labels.masked_select(batch.masks)
            correct_num+=torch.sum(torch.gt(predict_labels.float(),gold_labels.float())).data[0]
            total_num+=torch.sum(batch.masks).data[0]
        
        rate=correct_num/total_num
        print('total_num: {} , correct_num: {}'.format(total_num,correct_num))
        print('rate: ',rate)

if __name__=="__main__":
    parser = OptionParser()
        
    parser.add_option("--train", dest="trainFile",
                  help="train dataset")

    parser.add_option("--dev", dest="devFile",
                  help="dev dataset")

    parser.add_option("--test", dest="testFile",
                  help="test dataset")


    (options, args) = parser.parse_args()
    l = Labeler()
    l.train(options.trainFile, options.devFile, options.testFile)