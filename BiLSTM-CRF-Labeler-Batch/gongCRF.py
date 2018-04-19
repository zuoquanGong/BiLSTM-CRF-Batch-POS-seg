# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:36:17 2018

@author: zuoquan gong
"""

import torch.nn as nn
import torch.autograd
import torch
def log_sum_exp(scores, label_nums):

    batch_size = scores.size(0)
    max_scores, max_index = torch.max(scores, dim=1)
    max_score_broadcast = max_scores.unsqueeze(1).view(batch_size, 1, label_nums).expand(batch_size, label_nums, label_nums)
    return max_scores.view(batch_size, label_nums) + torch.log(torch.sum(torch.exp(scores - max_score_broadcast), 1)).view(batch_size, label_nums)

def log_sum_exp_low_dim(scores):
    
    batch_size = scores.size(0)
    label_nums = scores.size(1)
    max_score, max_index = torch.max(scores, 1)
    max_score_broadcast = max_score.unsqueeze(1).expand(batch_size, label_nums)
    return max_score + torch.log(torch.sum(torch.exp(scores - max_score_broadcast), 1))

class CRF(nn.Module):
    def __init__(self,label_size,start_id,padding_id,vocab):
        super(CRF, self).__init__()######
        self.label_size=label_size
        self.start_id=start_id
        self.padding_id=padding_id
        self.t=torch.zeros(label_size,label_size)#l*l 
        #规定第一维是原标签，第二维是转移后的标签
        self.t[:,start_id]=-10000
        self.t[padding_id,:]=-10000
        self.T=nn.Parameter(self.t)
        self.id2label=vocab.id2label
        print(self.id2label.items())
    def sentence_cal(self,emit_scores,batch_labels, masks):#emit_scores:(b*s,l) batch_labels:(b*s) masks:(b,l)
        batch_size=masks.size(0)#b
        seq_size=masks.size(1)#s
        batch_labels=batch_labels.view(batch_size,seq_size)#b,s
        list_labels=list(map(lambda t: [self.start_id] + list(t), batch_labels.data.tolist()))#b,s
        
        emit_scores=emit_scores.unsqueeze(1).expand(batch_size*seq_size,self.label_size,self.label_size)#b*s,l,l
        trans_scores=self.T.unsqueeze(0).expand(batch_size*seq_size,self.label_size,self.label_size)#(b*s,l,l)
        raw_scores=trans_scores+emit_scores#(b*s,l,l)
        raw_scores=raw_scores.view(batch_size,seq_size,self.label_size,self.label_size).view(batch_size,seq_size,self.label_size*self.label_size)
        #(b,s,l*l)
        label_group=[[label[i]*self.label_size+label[i+1] for i in range(seq_size)] for label in list_labels]
        label_group=torch.autograd.Variable(torch.LongTensor(label_group))#b,s
        
        scores=torch.gather(raw_scores,2,label_group.unsqueeze(2)).squeeze(2)#b,s
        scores=scores.masked_select(masks)
        
        end_indexs=torch.LongTensor([x.count(1)-1 for x in masks.data.tolist()])
        end_masks=torch.gather(batch_labels,1,end_indexs.unsqueeze(1))#b*1
#        print([self.id2label[i] for i in end_masks.squeeze(1).data.tolist()])
        end_scores=torch.gather(self.T[:,self.padding_id].unsqueeze(0).expand(batch_size,self.label_size),1,end_masks)
        return scores.sum()+end_scores.sum()
        
    def forward_cal(self,emit_scores,masks):#emit_scores:b*s*l   masks:b*l
    
        batch_size = masks.size(0)
        seq_size = masks.size(1)
        
        emit_scores=emit_scores.view(batch_size,seq_size,self.label_size).transpose(0,1)#s*b*l
        masks=masks.transpose(0,1)#s*b
        
        start_emit_scores=emit_scores[0]#b*l
        start_trans_scores=self.T[self.start_id,:].unsqueeze(0).expand(batch_size,self.label_size)#b*l
        forward_scores=start_emit_scores+start_trans_scores#b*l
        
        for i in range(1,seq_size):
            emit_score=emit_scores[i].unsqueeze(1).expand(batch_size,self.label_size,self.label_size)#b*l*l
            trans_score=self.T.unsqueeze(0).expand(batch_size,self.label_size,self.label_size)#b*l*l
            forward_score=forward_scores.unsqueeze(2).expand(batch_size,self.label_size,self.label_size).clone()#b*l*l
            
            scores=emit_score+trans_score+forward_score#b*l*l
            scores=log_sum_exp(scores,self.label_size)#b*l
            
            mask=masks[i].unsqueeze(1).expand(batch_size,self.label_size)#b*l
            masked_score=scores.masked_select(mask)
            forward_scores.masked_scatter_(mask, masked_score)#b*l
        
        end_scores=self.T[:,self.padding_id].unsqueeze(0).expand(batch_size,self.label_size)
        final_scores=log_sum_exp_low_dim(forward_scores+end_scores)
        return final_scores.sum()
    
    def neg_log_likelihood(self,tag_scores,batch_labels,masks):
        forward_scores=self.forward_cal(tag_scores,masks)
        gold_scores=self.sentence_cal(tag_scores,batch_labels,masks)
        return (forward_scores-gold_scores)/masks.size(0)
    

    def viterbi_decode(self,emit_scores,masks):#emit_scores:b*s*l   masks:b*l
        
            batch_size = masks.size(0)
            seq_size = masks.size(1)
            
            emit_scores=emit_scores.view(batch_size,seq_size,self.label_size).transpose(0,1)#s*b*l
            masks=masks.transpose(0,1)#s*b
            
            back_path=[]
            
            start_emit_scores=emit_scores[0]#b*l
            start_trans_scores=self.T[self.start_id,:].unsqueeze(0).expand(batch_size,self.label_size)#b*l
            forward_scores=start_emit_scores+start_trans_scores#b*l
            
            for i in range(1,seq_size):
                emit_score=emit_scores[i].unsqueeze(1).expand(batch_size,self.label_size,self.label_size)#b*l*l
                trans_score=self.T.unsqueeze(0).expand(batch_size,self.label_size,self.label_size)#b*l*l
                forward_score=forward_scores.unsqueeze(2).expand(batch_size,self.label_size,self.label_size).clone()#b*l*l
                
                scores=emit_score+trans_score+forward_score#b*l*l
                max_scores, max_indexs = torch.max(scores, dim=1)#b*l
                
                mask=masks[i].unsqueeze(1).expand(batch_size,self.label_size)#b*l
                masked_score=max_scores.masked_select(mask)
                forward_scores.masked_scatter_(mask, masked_score)#b*l
                
                mask = (1 + (-1) * mask.long()).byte()
                max_indexs.masked_fill(mask,self.padding_id)
                back_path.append(max_indexs.data.tolist())
            
            end_scores=self.T[:,self.padding_id].unsqueeze(0).expand(batch_size,self.label_size)#b,l
            scores=forward_scores+end_scores#b,l
            max_scores, ends_max_indexs = torch.max(scores, dim=1)#b
            
            
            #back_path
            back_path.append([[self.padding_id]*self.label_size  for _ in range(batch_size)])#s,b,l
            back_path=torch.autograd.Variable(torch.LongTensor(back_path).transpose(0,1))#b,s,l
            
            batch_length=torch.sum(masks,dim=0).long()-1#b
            ends_position=batch_length.unsqueeze(1).expand(batch_size,self.label_size)
            ends_max_indexs_broadcast=ends_max_indexs.unsqueeze(1).expand(batch_size,self.label_size)#b,l
            
            back_path.scatter_(1,ends_position.unsqueeze(1),ends_max_indexs_broadcast.contiguous().unsqueeze(1))
            back_path=back_path.transpose(0,1)
            
            decode_path=torch.autograd.Variable(torch.zeros(seq_size,batch_size))#s*b
            decode_path[-1]=ends_max_indexs#b
            ends_max_indexs=ends_max_indexs.unsqueeze(1)
            for i in range(seq_size-2,-1,-1):
                ends_max_indexs=torch.gather(back_path[i],1,ends_max_indexs)
                decode_path[i]=ends_max_indexs
            return decode_path




























