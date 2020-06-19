#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File: gnn.py
# Author: anon
# Email: anon@cs.anon.edu
# Created on: 2020-03-05
# 
# This file is part of MGN
# Distributed under terms of the MIT License

import inspect
from collections import OrderedDict

import torch
from torch.nn import Linear as Lin, BatchNorm1d as BN
from torch.nn import Sequential as Seq, ReLU
import torch.nn.functional as F

from torch_scatter import scatter_add
from torch_geometric.utils import scatter_
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn import GINConv

__all__ = ['GIN', 'MLP']

class GIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, batch_norm=False,
                 cat=True, lin=True):
        super(GIN, self).__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.cat = cat
        self.lin = lin

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP(in_channels, out_channels, 2, batch_norm)
            self.convs.append(GINConv(mlp, train_eps=True))
            in_channels = out_channels

        if self.cat:
            in_channels = self.in_channels + num_layers * out_channels
        else:
            in_channels = out_channels

        if self.lin:
            self.out_channels = out_channels
            self.final = Lin(in_channels, out_channels)
        else:
            self.out_channels = in_channels

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.lin:
            self.final.reset_parameters()

    def forward(self, x, edge_index, *args):
        """"""
        xs = [x]

        for conv in self.convs:
            xs += [conv(xs[-1], edge_index)]

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        x = self.final(x) if self.lin else x
        return x

    def __repr__(self):
        return ('{}({}, {}, num_layers={}, batch_norm={}, cat={}, '
                'lin={})').format(self.__class__.__name__, self.in_channels,
                                  self.out_channels, self.num_layers,
                                  self.batch_norm, self.cat, self.lin)

class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, batch_norm=False,
                 dropout=0.0):
        super(MLP, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.lins = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Lin(in_channels, out_channels))
            self.batch_norms.append(BN(out_channels))
            in_channels = out_channels

        self.reset_parameters()

    def reset_parameters(self):
        for lin, batch_norm in zip(self.lins, self.batch_norms):
            lin.reset_parameters()
            batch_norm.reset_parameters()

    def forward(self, x, *args):
        for i, (lin, bn) in enumerate(zip(self.lins, self.batch_norms)):
            if i == self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = bn(x) if self.batch_norm else x
        return x

    def __repr__(self):
        return '{}({}, {}, num_layers={}, batch_norm={}, dropout={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_layers, self.batch_norm, self.dropout)
