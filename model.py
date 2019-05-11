# -*- coding: utf-8 -*- 
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from utils.greedysearch import GreedySearchDecoder


class EncoderRNN(nn.Module):
    def __init__(self, opt, voc_length):
        '''
        voc_length: 字典长度,即输入的单词的one-hot编码长度
        '''
        super(EncoderRNN, self).__init__()
        self.num_layers = opt.num_layers
        self.hidden_size = opt.hidden_size
        #nn.Embedding输入向量维度是字典长度,输出向量维度是词向量维度
        self.embedding = nn.Embedding(voc_length, opt.embedding_dim)
        #双向GRU作为Encoder
        self.gru = nn.GRU(opt.embedding_dim, self.hidden_size, self.num_layers,
                          dropout=(0 if opt.num_layers == 1 else opt.dropout), bidirectional=opt.bidirectional)

    def forward(self, input_seq, input_lengths, hidden=None):
        '''
        input_seq: 
            shape: [max_seq_len, batch_size]
        input_lengths: 
            一批次中每个句子对应的句子长度列表
            shape:[batch_size]
        hidden:
            Encoder的初始hidden输入，默认为None
            shape: [num_layers*num_directions, batch_size, hidden_size]
            实际排列顺序是num_directions在前面, 
            即对于4层双向的GRU, num_layers*num_directions = 8
            前4层是正向: [:4, batch_size, hidden_size]
            后4层是反向: [4:, batch_size, hidden_size]
        embedded:
            经过词嵌入后的词向量
            shape: [max_seq_len, batch_size, embedding_dim]
        outputs:
            所有时刻的hidden层输出
            一开始的shape: [max_seq_len, batch_size, hidden_size*num_directions]
            注意: num_directions在前面, 即前面hidden_size个是正向的,后面hidden_size个是反向的
            正向: [:, :, :hidden_size] 反向: [:, :, hidden_size:]
            最后对双向GRU求和,得到最终的outputs: shape为[max_seq_len, batch_size, hidden_size]
        输出的hidden:
            [num_layers*num_directions, batch_size, hidden_size]
        '''
        
        embedded = self.embedding(input_seq) 
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        return outputs, hidden



class Attn(torch.nn.Module):
    def __init__(self, attn_method, hidden_size):
        super(Attn, self).__init__()
        self.method = attn_method #attention方法
        self.hidden_size = hidden_size
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(self.hidden_size))

    def dot_score(self, hidden, encoder_outputs):
        '''
        encoder_outputs:
            encoder(双向GRU)的所有时刻的最后一层的hidden输出
            shape: [max_seq_len, batch_size, hidden_size]
            数学符号表示: h_s
        hidden:
            decoder(单向GRU)的所有时刻的最后一层的hidden输出,即decoder_ouputs
            shape: [max_seq_len, batch_size, hidden_size]
            数学符号表示: h_t
        注意: attention method: 'dot', Hadamard乘法,对应元素相乘，用*就好了
            torch.matmul是矩阵乘法, 所以最后的结果是h_s * h_t
            h_s的元素是一个hidden_size向量, 要得到score值,需要在dim=2上求和
            相当于先不看batch_size,h_s * h_t 要得到的是 [max_seq_len]
            即每个时刻都要得到一个分数值, 最后把batch_size加进来,
            最终shape为: [max_seq_len, batch_size]   
        '''

        return torch.sum(hidden * encoder_outputs, dim=2)

    def general_score(self, hidden, encoder_outputs):
        #先学习到一个线性变换Wh_s,即energy
        #然后进行点乘,所以最后结果为 h_t*Wh_s
        energy = self.attn(encoder_outputs)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_outputs):
        '''
        hidden:
            h_t, shape: [max_seq_len, batch_size, hidden_size]
            expand(max_seq_len, -1,-1) ==> [max_seq_len, batch_size, hidden_size]
        与encoder_outputs在第2维上进行cat, 最后shape: [max_seq_len, batch_size, hidden_size*2]
        经过attn后得到[max_seq_len, batch_size, hidden_size],再进行tanh,shape不变
        最后与v乘
        '''
        energy = self.attn(torch.cat((hidden.expand(encoder_outputs.size(0), -1, -1), 
				      encoder_outputs), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)
    
    def forward(self, hidden, encoder_outputs):
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        #得到score,shape为[max_seq_len, batch_size],然后转置为[batch_size, max_seq_len]
        attn_energies = attn_energies.t()
        #对dim=1进行softmax,然后插入维度[batch_size, 1, max_seq_len]
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, opt, voc_length):
        super(LuongAttnDecoderRNN, self).__init__()

        self.attn_method = opt.method
        self.hidden_size = opt.hidden_size
        self.output_size = voc_length
        self.num_layers = opt.num_layers
        self.dropout = opt.dropout
        self.embedding = nn.Embedding(voc_length, opt.embedding_dim)
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.gru = nn.GRU(opt.embedding_dim, self.hidden_size, self.num_layers, dropout=(0 if self.num_layers == 1 else self.dropout))
        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.attn = Attn(self.attn_method, self.hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        '''
        input_step: 
            decoder是逐字生成的,即每个timestep产生一个字,
            decoder接收的输入: input_step='/SOS'的索引 和 encoder的最后时刻的最后一层hidden输出
            故shape:[1, batch_size]
        last_hidden:
            上一个GRUCell的hidden输出
            初始值为encoder的最后时刻的最后一层hidden输出，传入的是encoder_hidden的正向部分,
            即encoder_hidden[:decoder.num_layers], 为了和decoder对应,所以取的是decoder的num_layers
            shape为[num_layers, batch_size, hidden_size]
        encoder_outputs:
            这里还接收了encoder_outputs输入,用于计算attention
        '''
        #转为词向量[1, batch_size, embedding_dim]
        embedded = self.embedding(input_step) 
        embedded = self.embedding_dropout(embedded)
        #rnn_output: [1, batch_size, hidden_size]
        #hidden: [num_layers, batch_size, hidden_size]
        rnn_output, hidden = self.gru(embedded, last_hidden)
        #attn_weights: [batch_size, 1, max_seq_len]
        attn_weights = self.attn(rnn_output, encoder_outputs)
        #bmm批量矩阵相乘, 
        #attn_weights是batch_size个矩阵[1, max_seq_len]
        #encoder_outputs.transpose(0, 1)是batch_size个[max_seq_len, hidden_size]
        #相乘结果context为: [batch_size, 1, hidden_size]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        #压缩维度后,rnn_output[batch_size, hidden_size],context[batch_size, hidden_size]
        #在dim=1上连接: [batch_size, hidden_size * 2], contact后[batch_size, hidden_size]
        #tanh后shape不变: [batch_size, hidden_size]
        #最后进行全连接,映射为[batch_size, voc_length]
        #最后进行softmax: [batch_size, voc_length]
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))  
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)

        return output, hidden