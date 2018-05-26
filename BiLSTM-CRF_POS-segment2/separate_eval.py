# -*- coding: utf-8 -*-

"""
Created on Fri Apr 13 14:46:28 2018

@author: zuoquan gong
"""
from config import Params
from process.template import DataSet,Vocabulary,IndexSet,BatchBucket

from model.modelloader import ModelLoader

params=Params()
model_params=params.model_params
proc_params=params.process_params
train_eval_params=params.train_eval_params
vocab=Vocabulary(None,params.process_params,is_load=True)
model_params.vocabSize=vocab.word_vocab_size
model_params.outputSize=vocab.label_vocab_size
model_params.padId=vocab.padding_id
model_params.startId=vocab.start_id

test =DataSet(proc_params.path+proc_params.testFile,proc_params)
test=IndexSet(test,vocab)

models=ModelLoader(model_params)
    
models.eval()

correct_num=0####
total_num=0####
seg_correct_num=0
pos_correct_num=0
batchBlock=len(test.label_mat)//train_eval_params.batch_size
for updateIter in range(batchBlock):
    
    #--start batch--
    start_pos = updateIter * train_eval_params.batch_size
    end_pos = (updateIter + 1) * train_eval_params.batch_size
    feats=[]
    labels=[]
    for idx in range(start_pos, end_pos):
        feats.append(test.word_mat[idx])
        labels.append(test.label_mat[idx])
    batch_instance=BatchBucket( feats,labels, train_eval_params,
                                   vocab.padding_id)
    #--end batch--
    
    predict_labels=models.predict_labels(batch_instance).transpose(0,1)
    
    predict_labels=predict_labels.masked_select(batch_instance.masks)
    gold_labels=batch_instance.batch_labels.masked_select(batch_instance.masks)
    
    predict_labels=predict_labels.long()
    for i in range(predict_labels.size()[0]):
        predict=vocab.id2label[predict_labels.data[i].item()].split("-")
        gold=vocab.id2label[gold_labels.data[i].item()].split("-")
        if predict[0]==gold[0]:
            seg_correct_num+=1
        if predict[1]==gold[1]:
            pos_correct_num+=1
        if predict_labels.data[i]==gold_labels.data[i]:
            correct_num+=1
        total_num+=1
#            if updateIter==0:
#                print(predict_labels.data[:20])
#                print(gold_labels.data[:20])

rate=correct_num/total_num
print('total_num: {:^10d} | correct_num: {:^10d}'.format(total_num,correct_num))
print('correct_rate: {:^10f}'.format(rate))
print('seg_rate: {:^10f} | pos_rate: {:^10f}'.format(seg_correct_num/total_num,pos_correct_num/total_num))


































































