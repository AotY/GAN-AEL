#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.

"""
Approximate Embedding Layer
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class AEL(nn.module):
    def __init__(self, config):
        super(AEL, self).__init__()
        self.device = config.device

        self.softmax = nn.Softmax()

        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def noise(self, hidden_satet):
        noise = torch.rand(hidden_state.size()).normal_(0., 0.1), requires_grad=False).to(self.device)
        hidden_state = hidden_state + noise
        return hidden_state

    def forward(self, hidden_state, embedding, is_generator=False):
        '''
        Args:
            hidden_state: [batch_size, hidden_size] or [max_len, batch_size, hidden_size]
        Return:

        '''
        hidden_state = self.noise(hidden_state)

        # fully connected layer, [batch_size, vocab_size], or [max_len, batch_size, vocab_size]
        output = self.fc(hidden_state)

        if is_generator:
            # word distribution
            ael_weight = self.softmax(output, dim=2)

            # row weighted op
            # embedding: [vocab_size, embedding_size]
            # fc_output_weight: [batch_size, vocab_size]

            # [batch_size, embedding]
            ael_output = ael_weight @ embedding.weight

            return ael_output, ael_weight
        else:
            return output



