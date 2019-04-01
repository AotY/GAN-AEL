#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
CNN Classification
https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/CNN.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    def __init__(self, config):
        super(CNNEncoder, self).__init__()

        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.kernel_heights = config.kernel_heights

        self.stride = config.stride
        self.padding = config.padding

        self.conv1 = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            (self.kernel_heights[0], config.embedding_size),
            self.stride,
            self.padding
        )

        self.conv2 = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            (self.kernel_heights[1], config.embedding_size),
            self.stride,
            self.padding
        )

        self.conv3 = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            (self.kernel_heights[2], config.embedding_size),
            self.stride,
            self.padding
        )

        self.dropout = nn.Dropout(config.dropout)

        self.linear_final = nn.Linear(len(self.kernel_heights) * self.out_channels, self.out_channels)

        self.relu = nn.ReLU()

    def conv_block(self, inputs, conv_layer):
        # print('inputs shape: ', inputs.shape)
        # [batch_size, out_channels, dim, 1]
        conv_out = conv_layer(inputs)
        # print('conv_out shape: ', conv_out.shape)

        # [batch_size, out_channels, dim]
        activation = F.relu(conv_out.squeeze(3))
        # print('activation shape: ', activation.shape)

        # [batch_size, out_channels], kernel_sizes: the size of the window to
        # take a max over
        max_out = F.max_pool1d(activation, activation.size(2)).squeeze(2)
        # print('max_out shape: ', max_out.shape)

        return max_out

    def forward(self, embedded):
        '''
        Args:
            embedded: [max_len, batch_size, embedding_size]
        '''
        # [batch_size, max_len, embedding_size]
        embedded = embedded.transpose(0, 1)

        # [batch_size, 1, max_len, embedding_size]
        embedded = embedded.unsqueeze(1)

        max_out1 = self.conv_block(embedded, self.conv1)

        max_out2 = self.conv_block(embedded, self.conv2)

        max_out3 = self.conv_block(embedded, self.conv3)

        # [batch_size, num_kernels * out_channels]
        outputs = torch.cat((max_out1, max_out2, max_out3), dim=1)

        outputs = self.dropout(outputs)

        # [batch_size, out_channels]
        outputs = self.relu(self.linear_final(outputs))

