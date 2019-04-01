#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.

"""
GAN with Approximate Embedding Layer
"""

import torch.nn as nn
from modules.generator import Generator
from modules.discriminator import Discriminator


class GANAEL(nn.module):
    def __init__(self, config):
        super(GANAEL, self).__init__()

        self.config = config

        embedding = nn.Embedding(config.vocab_size, config.embedding_size)

        self.generator = Generator(config, embedding)

        self.discriminator = Discriminator(config, embedding)

        # tied embedding
        self.generator.embedding.data = self.discriminator.embedding.data

    # adversarial
    def forward(self, q_inputs, q_inputs_len, r_inputs, sos_input):
        '''
        Args:
            q_inputs: [max_len, batch_size]
            q_inputs_len: [batch_size]

            r_inputs: [max_len, batch_size] (true responses)
        '''

        # generator [max_len, batch_size, embedding_size], is_generator=True
        ael_embedded = self.generator.approximate(q_inputs, q_inputs_len, sos_input)

        r_embedded = self.embedding(r_inputs)

        # discriminator
        fake_outputs = self.discriminator(q_inputs, ael_embedded)

        real_outputs = self.discriminator(q_inputs, r_embedded)

        return fake_outputs, real_outputs

    def generator_forward(self, q_inputs, q_inputs_len, r_inputs):

        # generator [max_len, batch_size, embedding_size], is_generator=False
        outputs = self.generator(q_inputs, q_inputs_len)

        return outputs

    def discriminator_forward(self, q_inputs, r_inputs, ael_embedded):

        fake_outputs = self.discriminator(q_inputs, ael_embedded, is_ael=True)
        real_outputs = self.discriminator(q_inputs, r_inputs)

        return fake_outputs, real_outputs
