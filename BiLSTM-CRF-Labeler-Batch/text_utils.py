#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 13:54:36 2018

@author: zuoquan gong
"""
#text_loader -------vertical_cut-0
#          ╰-------horizontal_cut-1
#
from collections import OrderedDict
import numpy as np

def vertical_cut(text_lines,separator='\t',mini_cut=None):
    #cols=[]
    parts_template=[]
#    print('text_lines:',len(text_lines))
    sentences=[]#结构化的句子列表
    part_num=len(text_lines[0].strip().split(separator))
    
    head_index,end_index=-1,-1
#    inst = []
#    for line in text_lines:
#        line = line.strip()
#        if line == "":
#            inst.clear()
#        else:
#            inst.append(line)
        
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
         #  print(line)
            for index in range(part_num):
                #cols[index].append(line[index])
                sentence['part'][index].append(line[index])
                
        sentences.append(sentence)
    return sentences

def horizontal_cut1(text_lines,separator='\t',mini_cut=' '):
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
    
def horizontal_cut2(text_lines,separator=' ',mini_cut='_'):
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

def text_loader(path,mode=0,cut_out='#',separator='\t',mini_cut=' '):#1.庖丁切割
    #
    #
#    if mode==None:
#        mode=[[0]*len(paths)]
#    assert(len(paths)==len(mode))
    if mode==0:
        cut=vertical_cut
    elif(mode==1):
        cut=horizontal_cut1
    else:
        cut=horizontal_cut2
    
    fin=open(path,'r',encoding='utf-8')
    text_in=fin.readlines()
    #print(text_in)
    if cut_out!=None:
        text_in=[line for line in text_in if line[0]!=cut_out]
    fin.close()
    with open('tmp.txt','w',encoding='utf-8') as fout:
        for line in text_in :
            fout.write(line)
    text=cut(text_in,separator,mini_cut)
    return text

def format_convert(out_path,sentences):
    pass



def create_mapping(raw_list,word_list=None,vocab_size=None,cut_off=0,is_label=False):#
    assert(isinstance(raw_list,list))
    if word_list==None:
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
            
        if not is_label:
            unk=('<UNK>',freq_list[0][1]+1)
            freq_list.insert(0,unk)
            
        padding=('<padding>',freq_list[0][1]+1)
        freq_list.insert(0,padding)
    else:
        freq_list=word_list
    
    if vocab_size!=None:
        freq_list=freq_list[:vocab_size]
    item_to_id={element[0]:i for i,element in enumerate(freq_list)}
    id_to_item={i:element[0] for i,element in enumerate(freq_list)}
    #print(item_to_id)
    return item_to_id, id_to_item
    
def txt2mat(txt_list,item_to_id,vocab_size=None,save_mapping=False):#2.织女穿针
    assert(isinstance(txt_list,list))
    assert(isinstance(item_to_id,dict))
    index_mat=[]
    for line in txt_list:
        index_line=[]
        for word in line :
            if word in item_to_id.keys():
                index_line.append(item_to_id[word])
            else:
                index_line.append(item_to_id['<UNK>'])
        index_mat.append(index_line)
    ########################*********##########################
    if save_mapping:
        with open('vocab.txt','w') as fsave:
            vocab=OrderedDict(sorted(item_to_id.items(), key=lambda x: x[1]))
            for k,v in vocab.items():
                fsave.write(k)
                fsave.write('\t')
                fsave.write(str(v))
                fsave.write('\n')
    return index_mat

def extract_part(sentences,part):
    return [sentence['part'][part] for sentence in sentences]
  
def load_vocab(path,separator='\t'):
    with open(path,'r',encoding='utf-8') as fin:
        vocab=[]
        for line in fin.readlines():
#            print(line.strip().split(separator))
            word,freq=line.strip().split(separator)
            vocab.append((word,freq))
    return vocab

def mat_writer(mat,path,separator='\t'):
    with open(path,'w') as fout:
        for line in mat:
            for n in line:
                fout.write(str(n))
                fout.write(separator)
            fout.write('\n')
def txt_writer(txt,path,separator=' '):
    with open(path,'w') as fout:
        for line in txt:
            for n in line:
                fout.write(str(n))
                fout.write(separator)
            fout.write('\n')
def recovery_txt(mapping,mat):
    txt=[]
    for line in mat:
        txt.append([mapping[n] for n in line])
    return txt
    
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
            
#test code:     
#if __name__ == '__main__':
#    text=text_loader('./odd_data/hi-ud-test.conllu',mode=0)[0]
#    text=text_loader('./odd_data/ar-ud-test.conllu',0)[0]
#    txt=extract_col(text,1)
#    labels=extract_col(text,3)
#    tmp=load_vocab('./vocab.txt')
#    item_to_id,id_to_item=create_mapping(txt)
#    txt_mat=txt2mat(txt,item_to_id,save_mapping=True)
#    mat_writer(txt_mat,'./mat_sample.txt')
#    item_to_id,id_to_item=create_mapping(txt,tmp)
#    txt_mat=txt2mat(txt,item_to_id)
#    txt_writer(recovery_txt(id_to_item,txt_mat),'./chinese_txt.txt')
#    print(labels)

#    text=text_loader('./data/test',mode=1)[0]
























