#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.

"""
RNNDecoder with
the Approximate Embedding layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.ael import AEL


class RNNDecoder(nn.Module):
    def __init__(self, config):
        super(RNNDecoder, self).__init__()

        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        # dropout
        self.dropout = nn.Dropout(config.dropout)

        # rnn
        self.rnn = nn.GRU(
            input_size=config.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=config.dropout if self.num_layers > 1 else 0
        )

        self.ael = AEL(config.device)

    def forward(self, embedded, hidden_state, embedding, is_ael=False):
        '''
        Args:
            embedded: [1, batch_size, embedding_size]
            hidden_state: [num_layers * bidirection_num, batch_size, hidden_size]
        Return:
            ael_output: [batch_size, embedding]
            ael_weight: [batch_size, voab_size]
        '''

        if hidden_state is not None:
            rnn_output, hidden_state = self.rnn(embedded, hidden_state)
        else:
            rnn_output, hidden_state = self.rnn(embedded)

        if is_ael:
            ael_output, ael_weight = self.ael(hidden_state[-1], embedding)
            return ael_output, ael_weight, hidden_state
        else:
            # [max_len, batch_size, vocab_size]
            output = self.ael(rnn_output, embedding)
            return output


