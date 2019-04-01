#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.

"""

"""

import torch
import torch.nn as nn
from modules.rnn_encoder import RNNEcoder
from modules.rnn_decoder import RNNDecoder
from modules.reduce_state import ReduceState

class Generator(nn.module):
    def __init__(self, config, embedding):
        super(Generator, self).__init__()

        self.config = config
        self.embedding = embedding

        self.dropout = nn.Dropout(config.dropout)

        self.encoder = RNNEcoder(config)

        self.reduce_state = ReduceState()

        self.decoder = RNNDecoder(config)


    def forward(self, q_inputs, q_inputs_len, dec_inputs):
        '''
        Args:
            q_embedded: [max_len, batch_size]
            q_inputs_len: [batch_size]
            dec_inputs: [max_len, batch_size]
        Return:
            outputs: [max_len, batch_size, vocab_size]
        '''
        q_embedded = self.embedding(q_inputs)
        q_embedded = self.dropout(q_embedded)

        # encoder
        q_outputs, q_hidden_state = self.encoder(q_embedded, q_inputs_len)

        # [num_layers, batch_size, hidden_size]
        r_hidden_state = self.reduce_state(q_hidden_state)

        # decoder
        # [max_len, batch_size, vocab_size]
        outputs = self.decoder(dec_inputs, r_hidden_state, self.embedding, is_ael=False)

        return outputs

    def approximate(self, q_inputs, q_inputs_len, sos_input):
        '''
        Args:
            q_embedded: [max_len, batch_size, embedding_size]
            q_inputs_len: [batch_size]
        Return:
            ael_embeddeds: [max_len, batch_size, embedding_size]
        '''
        q_embedded = self.embedding(q_inputs)

        # encoder
        q_outputs, q_hidden_state = self.encoder(q_embedded, q_inputs_len)

        # [num_layers, batch_size, hidden_size]
        r_hidden_state = self.reduce_state(q_hidden_state)

        # decoder
        ael_embeddeds = list()

        #  [1, batch_size, embedding_size]
        r_embedded = self.embedding(sos_input)

        for i in range(self.config.max_len):
            r_embedded = self.dropout(r_embedded)
            # [batch_size, embedding_size], [batch_sizes, vocab_siz]
            ael_embedded, ael_weight, r_hidden_state = self.decoder(r_embedded, r_hidden_state, self.embedding, is_ael=True)
            ael_embeddeds.append(ael_embedded)
            r_embedded = ael_embedded

        # [max_len, batch_size, embedding_size]
        ael_embeddeds = torch.cat(ael_embeddeds, dim=0)

        return ael_embeddeds
