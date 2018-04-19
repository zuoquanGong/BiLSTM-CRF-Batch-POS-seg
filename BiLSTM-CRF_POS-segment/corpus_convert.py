# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 14:21:07 2018

@author: zuoquan gong
"""

#import utils
import train_eval.process.utils as utils
#使用 conllu 繁体中文 词性标注语料 生成 分词-词性标注 联合语料
def POS_to_segmentPOS(path):
    sentences=utils.text_loader(path,mode=0,cut_out='#',separator='\t',mini_cut=' ')
    word_lines=utils.extract_part(sentences,1)
    label_lines=utils.extract_part(sentences,3)
#    print(word_lines[3])
#    print(label_lines[3])
    new_feat_lines=[]
    new_label_lines=[]

    seg_labels=['S','B','M','E']
    for l in range(len(word_lines)):
        words=word_lines[l]
        labels=label_lines[l]
        new_feats=[]
        new_labels=[]
        for i in range(len(words)):
            for c,char in enumerate(words[i]):
                if c==0 and len(words[i])==1:
                    new_labels.append(seg_labels[0]+'-'+labels[i])
                elif c==0:
                    new_labels.append(seg_labels[1]+'-'+labels[i])
                elif c==len(words[i])-1:
                    new_labels.append(seg_labels[3]+'-'+labels[i])
                else:
                    new_labels.append(seg_labels[2]+'-'+labels[i])
                new_feats.append(char)
        new_feat_lines.append(new_feats)
        new_label_lines.append(new_labels)
    return new_feat_lines,new_label_lines
def generate_new_corpus(path):
    feats,labels=POS_to_segmentPOS(path)
    with open(path+'-seg','w',encoding='utf-8') as fout:
        for i in range(len(feats)):
            for j in range(len(feats[i])):
                fout.write(str(j)+'\t')
                fout.write(feats[i][j]+'\t')
                fout.write('_\t')
                fout.write(labels[i][j])
                fout.write('\n')
            fout.write('\n')
    
if __name__ == '__main__':
    path='./data/zh-ud-train.conllu'
    generate_new_corpus(path)
    path='./data/zh-ud-dev.conllu'
    generate_new_corpus(path)
    path='./data/zh-ud-test.conllu'
    generate_new_corpus(path)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    