# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:12:06 2018

@author: zuoquan gong
"""
import train_eval.process.utils as utils
from torch.autograd import Variable
from torch import LongTensor,ByteTensor
class Vocabulary:
    def __init__(self,dataset, params,is_load=False):
        
        self.save_word_mapping=params.save_word_mapping
        self.save_label_mapping=params.save_label_mapping
        self.vocab_size=params.vocabSize
        self.cutoff=params.cutOff
        if is_load:
            self.word2id,self.id2word=utils.load_mapping(self.save_word_mapping)
            self.label2id,self.id2label=utils.load_mapping(self.save_label_mapping)
        else:
            self.word2id,self.id2word=utils.create_mapping(dataset.words,vocab_size=self.vocab_size,cut_off=self.cutoff)
            self.label2id,self.id2label=utils.create_mapping(dataset.labels,vocab_size=self.vocab_size,cut_off=self.cutoff,is_label=True)
        
        self.word_vocab_size=len(self.word2id)
        self.label_vocab_size=len(self.label2id)
        
        self.unk_id=self.word2id['<unk>']
        self.padding_id=self.word2id['<padding>']
        self.start_id=self.label2id['<start>']

    def txt2mat(self,dataset,is_save=True):
        if is_save:
            word_mat=utils.txt2mat(dataset.words,self.word2id,save_path=self.save_word_mapping)
            label_mat=utils.txt2mat(dataset.labels,self.label2id,save_path=self.save_label_mapping)
            return word_mat,label_mat
        else:
            word_mat=utils.txt2mat(dataset,self.word2id)
            return word_mat

class DataSet:
    def __init__(self,path,params):
        sentences=utils.text_loader(path,mode=params.cutMode,separator=params.separator,
                                    mini_cut=params.miniCut,cut_out=params.cutOut)
        self.words=utils.extract_part(sentences,params.wordIndex)
        self.labels=utils.extract_part(sentences,params.labelIndex)
        self.size=len(self.words)

class IndexSet:
    def __init__(self,dataset,vocab):
        self.word_mat,self.label_mat=vocab.txt2mat(dataset)
        self.size=len(self.word_mat)
  
class PretrainEmbed:
    def __init__(self,file,vocab):
        self.emb=utils.load_pretrain(file,vocab)

class BatchBucket:
    def __init__(self,word_mat,label_mat,params,padding_id):#word和label的padId是相同的
        batch_size=params.batch_size
        sent_size=params.maxSentSize
        
        self.batch_words=Variable(LongTensor(batch_size,sent_size))#b,s
        self.batch_labels=Variable(LongTensor(batch_size,sent_size))#b,s
        self.masks=Variable(ByteTensor(batch_size,sent_size))#b,s
        self.batch_size=params.batch_size
        self.sent_size=params.maxSentSize
        
        for i in range(batch_size):
            for idx in range(sent_size):
                if idx<len(word_mat[i]):
                    self.batch_words.data[i][idx]=word_mat[i][idx]
                    self.batch_labels.data[i][idx]=label_mat[i][idx]
                    self.masks.data[i][idx]=1
                else:
                    self.batch_words.data[i][idx]=padding_id
                    self.batch_labels.data[i][idx]=padding_id
                    self.masks.data[i][idx]=0
    def show(self):
        print(self.batch_words.data[:2])
        print(self.batch_labels.data[:2])
        print(self.masks.data[:2])


























