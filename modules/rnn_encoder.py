#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
RNN Encoder
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.utils import rnn_factory, rnn_init

class RNNEncoder(nn.Module):
    def __init__(self,
                 config,
                 embedding=None):
        super(RNNEncoder, self).__init__()

        # embedding
        self.embedding = embedding
        self.embedding_size = embedding.embedding_dim

        self.bidirection_num = 2 if config.bidirectional else 1
        self.hidden_size = config.hidden_size // self.bidirection_num
        self.n_classes = config.n_classes
        self.num_layers = config.num_layers

        # dropout
        self.dropout = nn.Dropout(config.dropout)

        # rnn
        self.rnn = nn.GRU(
            input_size=config.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=config.bidirectional,
            dropout=config.dropout if self.num_layers > 1 else 0
        )

        self.linear_final = nn.Linear(
            self.hidden_size * self.bidirection_num, self.n_classes)

    def forward(self, inputs, lengths=None, hidden_state=None, inputs_pos=None):
        '''
        params:
            inputs: [seq_len, batch_size]  LongTensor
            hidden_state: [num_layers * bidirectional, batch_size, hidden_size]
        :return
            outputs: [batch_size, n_classes]
        '''
        # embedded
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)

        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths)

        if hidden_state is not None:
            outputs, hidden_state = self.rnn(embedded, hidden_state)
        else:
            outputs, hidden_state = self.rnn(embedded)

        if lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

            # last step output [batch_size, hidden_state]
            outputs = self.linear_final(outputs)

        return outputs, attns

