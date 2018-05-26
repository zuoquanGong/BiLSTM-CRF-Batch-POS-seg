# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:01:17 2018

@author: zuoquan gong
"""

#实用工具集合：文本载入，文本切分，建立映射，数据保存

from collections import OrderedDict
import numpy as np

def _vertical_cut(text_lines,separator='\t',mini_cut=None):
#==============================================================================
#     一、垂直切分
#     示例：1	（	_	PU	PU	_	2	P	_	_
#          2	完	_	VV	VV	_	0	ROOT	_	_
#          3	）	_	PU	PU	_	2	P	_	_
#==============================================================================
    parts_template=[]
    sentences=[]#结构化的句子列表
    part_num=len(text_lines[0].strip().split(separator))
    
    head_index,end_index=-1,-1
        
    while '\n' in text_lines[end_index+1:]:
        head_index=end_index
        end_index=text_lines.index('\n',head_index+1,)
        #print(head_index,end_index)
        sentence={}
        parts_template.clear()
        for count in range(part_num):
            parts_template.append([])
        sentence['part']=parts_template.copy()
        for i in range(end_index-head_index-1):
            line=text_lines[head_index+i+1].strip().split(separator)
            for index in range(part_num):
                sentence['part'][index].append(line[index])
                
        sentences.append(sentence)
    return sentences


def _horizontal_cut1(text_lines,separator='\t',mini_cut=' '):
#==============================================================================
#     二、水平切分1
#     示例；0	环境 不错 ， 适合 集体 出游 ， 位置 也 不 是 很 远 ！	位置 不错	61
#==============================================================================
    part_template=[]
    sentences=[]
    part_num=len(text_lines[0].strip().split(separator))
    for line in text_lines:
        sentence={}
        part_template.clear()
        for count in range(part_num):
            part_template.append([])
        sentence['part']=part_template
        for i,part in enumerate(line.strip().split(separator)):
            for x in part.strip().split(mini_cut):
                sentence['part'][i].append(x)
        sentences.append(sentence)
    return sentences
    
def _horizontal_cut2(text_lines,separator=' ',mini_cut='_'):
#==============================================================================
#     三、水平切分2
#     示例：新华社_NR 巴黎_NR ９月_NT １日_NT 电_NN （_PU 记者_NN 张浩_NR ）_PU 
#==============================================================================
    sentences=[]
    for line in text_lines:
        sentence={}
        part=[[],[]]
        sentence['part']=part
        for team in line.strip().split(separator):
            word,label=team.split(mini_cut)
            sentence['part'][0].append(word)
            sentence['part'][1].append(label)
        sentences.append(sentence)
    return sentences

def text_loader(path,mode=0,cut_out='#',separator='\t',mini_cut=' '):
    if mode==0:
        cut=_vertical_cut
    elif(mode==1):
        cut=_horizontal_cut1
    else:
        cut=_horizontal_cut2
    
    fin=open(path,'r',encoding='utf-8')
    text_in=fin.readlines()
    if cut_out!=None:
        text_in=[line for line in text_in if line[0]!=cut_out]
    fin.close()
    text=cut(text_in,separator,mini_cut)
    return text
    
#******************************************************************************    
    
def create_mapping(raw_list,vocab_size=None,cut_off=0,is_label=False):#
    assert(isinstance(raw_list,list))
    word_list=[]
    for line in raw_list:
        for word in line :
            word_list.append(word)
    raw_dict={}
    for element in word_list:
        if element in raw_dict.keys():
            raw_dict[element]+=1
        else:
            raw_dict[element]=1

    freq_list=sorted(raw_dict.items(),key=lambda x:x[1],reverse=True)
    if is_label:
        start=('<start>',freq_list[0][1]+1)
        freq_list.insert(0,start)
        
    else:
        unk=('<unk>',freq_list[0][1]+1)
        freq_list.insert(0,unk)
        
    padding=('<padding>',freq_list[0][1]+1)
    freq_list.insert(0,padding)
    
    if vocab_size!=None:
        freq_list=freq_list[:vocab_size]
    item_to_id={element[0]:i for i,element in enumerate(freq_list)}
    id_to_item={i:element[0] for i,element in enumerate(freq_list)}
    return item_to_id, id_to_item
    
    
    
def txt2mat(txt_list,item_to_id,vocab_size=None,save_path=''):
    assert(isinstance(txt_list,list))
    assert(isinstance(item_to_id,dict))
    index_mat=[]
    for line in txt_list:
        index_line=[]
        for word in line :
            if word in item_to_id.keys():
                index_line.append(item_to_id[word])
            else:
                index_line.append(item_to_id['<unk>'])
        index_mat.append(index_line)
    #*************************************************
    if save_path!='':
        with open(save_path,'w',encoding='utf-8') as fsave:
            vocab=OrderedDict(sorted(item_to_id.items(), key=lambda x: x[1]))
            for k,v in vocab.items():
                fsave.write(k)
                fsave.write('\t')
                fsave.write(str(v))
                fsave.write('\n')
    return index_mat

def extract_part(sentences,part):
    return [sentence['part'][part] for sentence in sentences]
  

def load_pretrain(path,vocab,dim=None):
    embed_dict={}
    with open(path,'r',encoding='utf-8') as fin:
        lines=fin.readlines()
        embed_dim=-1
        for line in lines:
            line_split=line.strip().split(' ')
            if len(line_split)<3:continue
            if embed_dim<1:
                embed_dim=len(line_split)-1
                if dim==None:
                    dim=embed_dim
                else:
                    assert(dim==embed_dim)
            else:
                assert(embed_dim==len(line_split)-1)
            embed=np.zeros((1,dim))
            embed[:]=line_split[1:]
            embed_dict[line_split[0]]=embed
    pretrain_emb=np.zeros((vocab.size(),dim))
    match=0
    for word in vocab.word2id.keys():
        if word in embed_dict:
            pretrain_emb[vocab.word2id[word],:]=embed_dict[word]
            match+=1
    print('Pretrain: match--{},no_match--{}'.format(match,vocab.size()-match))
    return pretrain_emb





def load_mapping(path,separator='\t'):
    with open(path,'r',encoding='utf-8') as fin:
        item2id={}
        id2item={}
        for line in fin.readlines():
#            print(line.strip().split(separator))
            item,id=line.strip().split(separator)
            item2id[item]=int(id)
            id2item[int(id)]=item
    return item2id,id2item
#
#def mat_writer(mat,path,separator='\t'):
#    with open(path,'w') as fout:
#        for line in mat:
#            for n in line:
#                fout.write(str(n))
#                fout.write(separator)
#            fout.write('\n')
#def txt_writer(txt,path,separator=' '):
#    with open(path,'w') as fout:
#        for line in txt:
#            for n in line:
#                fout.write(str(n))
#                fout.write(separator)
#            fout.write('\n')
#def recovery_txt(mapping,mat):
#    txt=[]
#    for line in mat:
#        txt.append([mapping[n] for n in line])
#    return txt


