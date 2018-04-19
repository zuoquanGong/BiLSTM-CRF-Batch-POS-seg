# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 20:29:05 2018

@author: zuoquan gong
"""
import text_utils as utils
from torch.autograd import Variable
from torch import LongTensor,ByteTensor
from hyperparams import HyperParams
hyperParams=HyperParams()
class Alphabet:
    def __init__(self,dataset, vocab_size=hyperParams.vocabSize,cutoff=hyperParams.cutOff):
        self.word2id,self.id2word=utils.create_mapping(dataset.words,vocab_size=vocab_size,cut_off=cutoff)
        self.label2id,self.id2label=utils.create_mapping(dataset.labels,vocab_size=vocab_size,cut_off=cutoff,is_label=True)
    def txt2mat(self,dataset):
        word_mat=utils.txt2mat(dataset.words,self.word2id)
        label_mat=utils.txt2mat(dataset.labels,self.label2id)
        return word_mat,label_mat
    def size(self):
        return len(self.word2id)
    def label_size(self):
        return len(self.label2id)

class DataSet:
    def __init__(self,path):
        sentences=utils.text_loader(path,mode=hyperParams.cutMode,separator=hyperParams.separator,
                                    mini_cut=hyperParams.miniCut,cut_out=hyperParams.cutOut)
        self.words=utils.extract_part(sentences,hyperParams.wordIndex)
        self.labels=utils.extract_part(sentences,hyperParams.labelIndex)
    def size(self):
        return len(self.words)

class IndexSet:
    def __init__(self,dataset,vocab):
        self.word_mat,self.label_mat=vocab.txt2mat(dataset)
    def size(self):
        return len(self.word_mat)
  
class PretrainEmb:
    def __init__(self,file,vocab):
        self.emb=utils.load_pretrain(file,vocab)

class BatchBucket:
    def __init__(self,batch_size,sent_size,word_mat,label_mat,padding_id,padding_label_id):
        self.batch_words=Variable(LongTensor(batch_size,sent_size))
        self.batch_labels=Variable(LongTensor(batch_size*sent_size))
        self.masks=Variable(ByteTensor(batch_size,sent_size))###############
        for i in range(batch_size):
            for idx in range(sent_size):
                if idx<len(word_mat[i]):
                    self.batch_words.data[i][idx]=word_mat[i][idx]
                    self.batch_labels.data[i*sent_size+idx]=label_mat[i][idx]
                    self.masks.data[i][idx]=1
                else:
                    self.batch_words.data[i][idx]=padding_id
                    self.batch_labels.data[i*sent_size+idx]=padding_label_id##############
                    self.masks.data[i][idx]=0
        
#以下代码用于测试utils中的load_pretrain函数
#if __name__ == '__main__':
#
#    trainFile='./parser_corpus/dev.ctb51.conll'
#    train=DataSet(trainFile)
#    vocab=Alphabet(train)
#    print('vocab.size: ',vocab.size())
#    train_mat=IndexSet(train,vocab)
#    pretrain_emb=utils.load_pretrain('vectors200.txt',vocab)
#    print(pretrain_emb)
