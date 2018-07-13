#-*- encoding:utf-8 -*-
import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.5, debug=False):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)
        self.debug = debug
    
    def forward(self, src, hidden=None):
        embedded = self.embed(src)   # src [Q, B] --> [Q, B, E]
        embedded = self.dropout(embedded)
        rnn_outputs, hidden = self.gru(embedded, hidden)  # rnn_out [Q, B, 2H], hidden [2*layers, B, H]
        # sum bidirectional outputs
#         outputs = (rnn_outputs[:, :, :self.hidden_size] +
#                    rnn_outputs[:, :, self.hidden_size:])
        outputs = rnn_outputs
        if self.debug:
            print (u"encoder src :", src.size())
            print (u"encoder rnn output :", rnn_outputs.size())
            print (u"encoder hidden :", hidden.size())
            print (u"encoder output :", outputs.size())
        return outputs, hidden #[Q, B, 2H]  [2*layers, B, H]


class Attention(nn.Module):
    def __init__(self, hidden_size, debug=False):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)
        self.debug = debug

    def forward(self, hidden, encoder_outputs, src_mask):
        timestep = encoder_outputs.size(0)
        #h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
#         print (u"att hidden :", hidden.size())
        h = hidden.unsqueeze(2)#[B, 2H] --> [B, 2H, 1]
#         print (u"h size :", h.size())
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*2H]
        scores = self.score(h, encoder_outputs)
        attn_energies = scores
        attn_energies.data.masked_fill_(src_mask.data, -float('inf'))
        if self.debug:
            print (u"att :", F.softmax(attn_energies, dim=1).unsqueeze(1).size())
        return F.softmax(attn_energies, dim=1).unsqueeze(1), attn_energies.unsqueeze(1) #[B, 1, Q]

    def score(self, hidden, encoder_outputs): 
        energy = torch.bmm(encoder_outputs, hidden)#[B, T, 2H] * [B, 2H, 1] --> [B, T, 1]
        return energy.squeeze(2)#[B, T]

#         # [B*T*2H]->[B*T*H]
#        energy = self.attn(torch.cat([hidden, encoder_outputs], 2))
#        energy = energy.transpose(1, 2)  # [B*H*T]
#        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
#        energy = torch.bmm(v, energy)  # [B*1*T]
#        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2, debug=False):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.debug = debug

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(2*hidden_size, self.debug)
        self.gru = nn.GRU(hidden_size * 4 + embed_size, hidden_size*2,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 4, output_size)

    def forward(self, last_output, input, last_hidden, encoder_outputs, src_mask):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,E)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights, attn_energies = self.attention(last_output[-1], encoder_outputs, src_mask)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,2H)
        context = context.transpose(0, 1)  # (1,B,2H)
        
        #context = encoder_outputs[0].unsqueeze(0)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([last_output, embedded, context], 2)#[2H + E + 2H]
        rnn_output, hidden = self.gru(rnn_input, last_hidden)
        output = rnn_output.squeeze(0)  # (1,B,2H) -> (B,2H)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)#[B, V]
        if self.debug:
            print (u"decoder rnn output :", rnn_output.size())
            print (u"decoder :", output.size())
            print (u"decoder hidden size :", hidden.size())
        return rnn_input, rnn_output, output, hidden, attn_weights, attn_energies


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, debug=False):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.debug = debug

    def forward(self, src, trg, src_mask, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

        encoder_output, hidden = self.encoder(src)#[Q, B, 2H]  [2*L, B, H]
        #假定encoder和decoder的层数一致 
#         hidden = torch.cat([hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]], 2)#[L, B, 2H]
        output = Variable(trg.data[0, :])  # <sos> word for the first timestep
        his_attn_weights = []
        his_attn_energies = []
        his_hidden = []
        his_de_input = []
        de_output = Variable(torch.zeros(1, batch_size, self.decoder.hidden_size * 2)).cuda()
        hidden = Variable(torch.zeros(1, batch_size, self.decoder.hidden_size * 2)).cuda()
        for t in range(1, max_len):
            his_hidden.append(de_output)
            de_input, de_output, output, hidden, attn_weights, attn_energies = self.decoder(
                    de_output, output, hidden, encoder_output, src_mask)
            his_attn_weights.append(attn_weights)
            his_attn_energies.append(attn_energies)
            his_de_input.append(de_input)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(trg.data[t] if is_teacher else top1).cuda()
            if self.debug:
                print (u"decoder att weight :", attn_weights.size())
        attn_weights = torch.cat(his_attn_weights, 1)#[B, L, Q]
        attn_energies = torch.cat(his_attn_energies, 1)
        de_input = torch.cat(his_de_input, 0).transpose(0,1)
        hidden = torch.cat(his_hidden, 0).transpose(0,1)
        if self.debug:
            print (u"src_mask size :", src_mask.size())
            print (u"decoder rnn hidden :", hidden.size())
        
        return de_input, outputs, hidden, attn_weights, attn_energies, encoder_output #[L, B, V]
