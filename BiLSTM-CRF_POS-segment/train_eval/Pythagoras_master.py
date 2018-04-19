# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 19:29:36 2018

@author: zuoquan gong
"""
import torch
import sys
import random
from train_eval.process.template import DataSet,Vocabulary,IndexSet,BatchBucket,PretrainEmbed
from train_eval.model.modelloader import ModelLoader

class Pythagoras:
#==============================================================================
#     'Everything is number.'  ——Pythagoras
#==============================================================================
    def __init__(self,params):
        
        params.__show__()
        self.proc_params=params.process_params
        self.model_params=params.model_params
        self.train_eval_params=params.train_eval_params
        
        self.train_set=[]
        self.dev_set=[]
        self.test_set=[]
        self.vocab=[]
        
#==============================================================================
#     一、train model
#==============================================================================
    def process(self):
        
#       ******************  1.data process  ***********************************
                
        train =DataSet(self.proc_params.path+self.proc_params.trainFile,self.proc_params)
        dev =DataSet(self.proc_params.path+self.proc_params.devFile,self.proc_params)
        test =DataSet(self.proc_params.path+self.proc_params.testFile,self.proc_params)
        
        vocab=Vocabulary(train,self.proc_params)
        self.proc_params.vocabSize=vocab.word_vocab_size
        self.proc_params.labelSize=vocab.label_vocab_size
        print('vocab_size:',self.proc_params.vocabSize)
        print('label_size:',self.proc_params.labelSize)
        self.model_params.vocabSize=vocab.word_vocab_size
        self.model_params.outputSize=vocab.label_vocab_size
        self.model_params.padId=vocab.padding_id
        self.model_params.startId=vocab.start_id
        
        train=IndexSet(train,vocab)
        dev=IndexSet(dev,vocab)
        test=IndexSet(test,vocab)
        
        print('\n[  DataSet Parameters  ]\n')
        print('trainset_size: ',train.size)
        print('devset_size: ',dev.size)
        print('testset_size: ',test.size)
        
        if self.model_params.embedFile!='':
            self.model_params.pretrain=PretrainEmbed(self.model_params.embedFile,
                                                     vocab.word2id)
        else:
            self.model_params.pretrain=None
            
        self.train_set=train
        self.dev_set=dev
        self.test_set=test
        self.vocab=vocab

    def train(self):
#       ******************  2.model train  ************************************
        
        self.process()
        print('\n[  Model Info  ]\n')
        self.models=ModelLoader(self.model_params)#models: model+optimizer##
#        params=self.models.model_list[0].state_dict() 
        
#        with open('D:/Desktop/NewWork/BiLSTM-CRF_POS-segment/model_params/params.save','w',encoding='utf-8') as fsave:
#            for k,v in params.items():
#                print >> fsave,k
#        

#        __stdout__ = sys.stdout
#        sys.stdout = open('D:/Desktop/NewWork/BiLSTM-CRF_POS-segment/model_params/params.save','w',encoding='utf-8')
#        for k,v in params.items():
#                print(k,':\n',v)
#        sys.stdout.close()
#        sys.stdout = __stdout__
#
#        
#        print(params['lstm.weight_ih_l0']) 
#        self.models.save_model()        

        indexes = [idx for idx in range(self.train_set.size)]
            
        batch_size=self.train_eval_params.batch_size
        batchBlock = self.train_set.size// batch_size
        
        print('\n[  Train Iterator  ]\n')
        for iter in range(self.train_eval_params.maxIter):
            
            print('\n###  Iteration ' + str(iter) + "  ###")
            random.shuffle(indexes)
            
            self.models.train()###
            
            for updateIter in range(batchBlock):
                
                self.models.clear_grads()##
                
                #--start batch--
                start_pos = updateIter * batch_size
                end_pos = (updateIter + 1) * batch_size
                feats=[]
                labels=[]
                for idx in range(start_pos, end_pos):
                    feats.append(self.train_set.word_mat[indexes[idx]])
                    labels.append(self.train_set.label_mat[indexes[idx]])
                batch_instance=BatchBucket( feats,labels, self.train_eval_params,
                                           self.vocab.padding_id)
                
                #--end batch--
                
                self.models.forward_all(batch_instance)
                self.models.optim_step()##
                
                if (updateIter + 1) % self.train_eval_params.showIter == 0:
                   self.models.batch_show(idx)
    
            self.models.eval()###
            self.models.best_show()
            dev_rate=self.eval_predict(self.dev_set,)
            test_rate=self.eval_predict(self.test_set)
            self.models.save_best_model(dev_rate,test_rate,iter)
#            difference=self.models.save_best_model(dev_rate,test_rate,iter)
#            if difference<self.train_eval_params.threshold:
#                break
            
#==============================================================================
#     二、evaluate model    
#==============================================================================

#    def evaluate_shell(self):
#        pass
    
    def eval_predict(self,indexset):
        vocab=self.vocab
        correct_num=0####
        total_num=0####
        batchBlock=len(indexset.label_mat)//self.train_eval_params.batch_size
        for updateIter in range(batchBlock):
            
            #--start batch--
            start_pos = updateIter * self.train_eval_params.batch_size
            end_pos = (updateIter + 1) * self.train_eval_params.batch_size
            feats=[]
            labels=[]
            for idx in range(start_pos, end_pos):
                feats.append(indexset.word_mat[idx])
                labels.append(indexset.label_mat[idx])
            batch_instance=BatchBucket( feats,labels, self.train_eval_params,
                                           vocab.padding_id)
            #--end batch--
            
            predict_labels=self.models.predict_labels(batch_instance).transpose(0,1)
            
            predict_labels=predict_labels.masked_select(batch_instance.masks)
            gold_labels=batch_instance.batch_labels.masked_select(batch_instance.masks)
            
            for i in range(predict_labels.size()[0]):
                if predict_labels.data[i]==gold_labels.data[i]:
                    correct_num+=1
                total_num+=1
#            if updateIter==0:
#                print(predict_labels.data[:20])
#                print(gold_labels.data[:20])
        
        rate=correct_num/total_num
        print('total_num: {:^10d} | correct_num: {:^10d}'.format(total_num,correct_num))
        print('correct_rate: {:^10f}'.format(rate))
        return rate

            



