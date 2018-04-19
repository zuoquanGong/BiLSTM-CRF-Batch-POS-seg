# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 21:58:44 2018

@author: zuoquan gong
"""

import torch
import datetime

class ModelLoader:
    def __init__(self,params):
        self.params=params
        self.best_model_eval=0
        self.model_name_list=self.params.modelList
        self.model_list=[]
        self.optim_list=[]
        self.best_model_eval=0
       
        try:
            if self.params.loadFile!=['']*self.params.modelNum:
                if not self.params.log_clear:#是否载入__log__中的数据
                    with open(self.params.savePath+'__log__','r') as flog:
                        #print(flog.readlines()[1].strip().split()[1])
                        self.best_model_eval=float(flog.readlines()[1].strip().split()[1])
        except:
            pass
        
        for i,model_name in enumerate(self.model_name_list):
            model_file=__import__('train_eval.model.'+model_name,fromlist=['Model'])
            model_item=model_file.Model(params)
            
            if self.params.loadFile[i]!='':    
                model_item.load_state_dict(torch.load(params.loadFile[i]))
                print('( model loading ... )')
            
            try:
                model_item.show_info()#模型类可以未创建show_info函数
            except:
                pass
            
            if i==0:
                parameters = filter(lambda p: p.requires_grad, model_item.parameters())
                optim_item=torch.optim.Adam(parameters, lr = params.learningRate)
            else:
                optim_item=torch.optim.Adam(model_item.parameters(), lr = params.learningRate)
            
            self.model_list.append(model_item)#所有模型的实例组成的list
            self.optim_list.append(optim_item)
    
    def forward_all(self,data):#data 即为 BatchBucket类实例: 包含 feats, labels, masks
        tmp_result=self.model_list[0](data)
        if len(self.model_list)>=2:
            for model_item in self.model_list[1:]:
                tmp_result=model_item(tmp_result,data)
        self.loss=tmp_result
        self.loss.backward()
    
    def predict_labels(self,data):
        if len(self.model_list)>=2:
            tmp_result=self.model_list[0](data)
            for model_item in self.model_list[1:-1]:
                tmp_result=model_item(tmp_result,data)
            predict_labels=self.model_list[-1].predict(tmp_result,data)
        else:
            predict_labels=self.model_list[0](data)
        return predict_labels
    
    def clear_grads(self):
        for optim_item in self.optim_list:
            optim_item.zero_grad()
    def optim_step(self):
        for optim_item in self.optim_list:
            optim_item.step()
    def train(self):
        for model_item in self.model_list:
            model_item.train()
    def eval(self):
        for model_item in self.model_list:
            model_item.eval()
    def save_best_model(self,model_eval,test_rate,iter_num):
        difference=10000
        if model_eval>=self.best_model_eval:
            print('model saving ...')
            difference=model_eval-self.best_model_eval
            self.best_model_eval=model_eval
            for i in range(len(self.model_list)):
#                if i==1:
#                    print(self.model_list[i].T)#**
                torch.save(self.model_list[i].state_dict(), self.params.savePath+self.params.saveFile[i])
            with open(self.params.savePath+'__log__','w') as flog:
                flog.write(datetime.datetime.now().strftime('%b-%d-%Y %H:%M:%S')+'\n')
                flog.write('best_model_eval: '+str(self.best_model_eval)+'\n')
                flog.write('test_rate: '+str(test_rate)+'\n')
                flog.write('iter num: '+str(iter_num)+'\n')
        else:
            pass
        return difference
            
    def batch_show(self,idx):
        print('current:{:^10d}, cost:{:^10f}'.format(idx + 1, self.loss.data[0]))
    def best_show(self):
        print('best rate:{:^10f}'.format(self.best_model_eval))
    
    def save_model(self):
        for i in range(len(self.model_list)):
            torch.save(self.model_list[i].state_dict(), self.params.savePath+self.params.saveFile[i])
        print('extra model saving ... ')
    
    
    
    
    
    
    
    