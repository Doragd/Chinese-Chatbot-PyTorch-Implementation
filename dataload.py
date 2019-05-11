# -*- coding: utf-8 -*- 

import torch
import itertools
from torch.utils import data as dataimport

def zeroPadding(l, fillvalue):
    '''
    l是多个长度不同的句子(list)，使用zip_longest padding成定长，长度为最长句子的长度
    在zeroPadding函数中隐式转置
    [batch_size, max_seq_len] ==> [max_seq_len, batch_size]
    '''
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value):
    '''
    生成mask矩阵, 0表示padding,1表示未padding
    shape同l,即[max_seq_len, batch_size]
    '''
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def create_collate_fn(padding, eos):
    '''
    说明dataloader如何包装一个batch,传入的参数为</PAD>的索引padding,</EOS>字符索引eos
    collate_fn传入的参数是由一个batch的__getitem__方法的返回值组成的corpus_item

    corpus_item: 
        lsit, 形如[(inputVar1, targetVar1, index1),(inputVar2, targetVar2, index2),...]
        inputVar1: [word_ix, word_ix,word_ix,...]
        targetVar1: [word_ix, word_ix,word_ix,...]
    inputs: 
        取出所有inputVar组成的list,形如[inputVar1,inputVar2,inputVar3,...], 
        padding后(这里有隐式转置)转为tensor后形状为:[max_seq_len, batch_size]
    targets:
        取出所有targetVar组成的list,形如[targetVar1,targetVar2,targetVar3,...]
        padding后(这里有隐式转置)转为tensor后形状为:[max_seq_len, batch_size]
    input_lengths: 
        在padding前要记录原来的inputVar的长度, 用于pad_packed_sequence
        形如: [length_inputVar1, length_inputVar2, length_inputVar3, ...]
    max_targets_length:
        该批次的所有target的最大长度
    mask:
        形状: [max_seq_len, batch_size]
    indexes:
        记录一个batch中每个 句子对 在corpus数据集中的位置
        形如: [index1, index2, ...]

    '''
    def collate_fn(corpus_item):
        #按照inputVar的长度进行排序,是调用pad_packed_sequence方法的要求
        corpus_item.sort(key=lambda p: len(p[0]), reverse=True) 
        inputs, targets, indexes = zip(*corpus_item)
        input_lengths = torch.tensor([len(inputVar) for inputVar in inputs])
        inputs = zeroPadding(inputs, padding)
        inputs = torch.LongTensor(inputs) #注意这里要LongTensor
        
        max_target_length = max([len(targetVar) for targetVar in targets])
        targets = zeroPadding(targets, padding)
        mask = binaryMatrix(targets, padding)
        mask = torch.ByteTensor(mask)
        targets = torch.LongTensor(targets)
        
        
        return inputs, targets, mask, input_lengths, max_target_length, indexes

    return collate_fn




class CorpusDataset(dataimport.Dataset):

    def __init__(self, opt):
        self.opt = opt
        self._data = torch.load(opt.corpus_data_path)
        self.word2ix = self._data['word2ix']
        self.corpus = self._data['corpus']
        self.padding = self.word2ix.get(self._data.get('padding'))
        self.eos = self.word2ix.get(self._data.get('eos'))
        self.sos = self.word2ix.get(self._data.get('sos'))
        
    def __getitem__(self, index):
        inputVar = self.corpus[index][0]
        targetVar = self.corpus[index][1]
        return inputVar,targetVar, index

    def __len__(self):
        return len(self.corpus)


def get_dataloader(opt):
    dataset = CorpusDataset(opt)
    dataloader = dataimport.DataLoader(dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=opt.shuffle, #是否打乱数据
                                 num_workers=opt.num_workers, #多进程提取数据
                                 drop_last=True, #丢掉最后一个不足一个batch的数据
                                 collate_fn=create_collate_fn(dataset.padding, dataset.eos))
    return dataloader