#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positionwise feed forward layer definition."""

import torch


class ExpLogLinear(torch.nn.Module):
    """ExpLogLinear feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, attn_dim, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(ExpLogLinear, self).__init__()
        self.w_1 = torch.nn.Linear(idim, idim * 4)
        self.w_2 = torch.nn.Linear(idim, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation
        self.idim = idim

    def forward(self, x):
        """Forward function."""
        b, t, d = x.size()
        intermediate = self.w_1(x)
        intermediate = intermediate.view(b, t, 4, d)
        intermediate[:, :, 1, :] = torch.exp(intermediate[:, :, 1, :])
        intermediate[:, :, 2, :] = torch.log(intermediate[:, :, 2, :])
        intermediate[:, :, 3, :] = torch.sin(intermediate[:, :, 3, :])
        intermediate = intermediate.view(b, t, d)
        return self.w_2(self.dropout(self.activation(intermediate)))
