#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.
import torch
import torch.nn as nn

"""
hidden_state state
num_layers * num_directions, batch_size, hidden_size
->
num_layers, batch_size, hidden_size
"""


class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()
        pass


    def forward(self, hidden_state):
        """ Init decoder state with last state of the encoder """
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            hidden = torch.cat([hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]], dim=2)
            return hidden

        # [num_layers * bidirection_num, batch_size, hidden_size]
        # -> [num_layers, batch_size, hidden_state]
        reduce_h = _fix_enc_hidden(hidden_state)
        return reduce_h


