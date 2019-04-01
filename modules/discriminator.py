#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.

"""
Discriminator
"""

import torch
import torch.nn as nn
from modules.cnn_encoder import CNNEncoder


class Discriminator(nn.Module):
    def __init__(self, config, embedding):
        super(Discriminator, self).__init__()
        self.config = config
        self.embedding = embedding

        self.q_cnn = CNNEncoder(config)
        self.r_cnn = CNNEncoder(config)

        # one fc, in order to share fc parameters
        self.fc = nn.Linear(config.out_channels * 2, 1)

        self.sigmoid = nn.Sigmoid()


    def forward(self, q_inputs, r_inputs, is_ael=False):
        '''
        Args:
            q_embedded: [max_len, batch_size, embedding_size] r_embedded: [max_len, batch_size, embedding_size]
        Return:
            outputs: [batch_size]
        '''
        q_embedded = self.embedding(q_inputs)
        if not is_ael:
            r_embedded = self.embedding(r_inputs)
        else:
            r_embedded = r_inputs

        # [batch_size, out_channels]
        q_outputs = self.q_cnn(q_embedded)
        # [batch_size, out_channels]
        r_outputs = self.r_cnn(r_embedded)

        # [batch_size, out_channels * 2] -> [batch_size, 1]
        outputs = self.fc(torch.cat((q_outputs, r_outputs), dim=2))

        # [batch_size]
        outputs = self.sigmoid(outputs).view(-1)

        return outputs
