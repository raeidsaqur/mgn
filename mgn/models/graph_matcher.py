#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File: graph_matcher.py
# Author: anon
# Email: anon@cs.anon.edu
# Created on: 2020-05-10
# 
# This file is part of MGN
# Distributed under terms of the MIT License

import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.utils import to_dense_batch, to_dense_adj

__all__ = ['GraphMatcher']

######################################### UTILS ############################
def masked_softmax(src, mask, dim=-1):
    out = src.masked_fill(~mask, float('-inf'))
    out = torch.softmax(out, dim=dim)
    out = out.masked_fill(~mask, 0)
    return out

def to_sparse(x, mask):
    return x[mask]

def to_dense(x, mask):
    out = x.new_zeros(tuple(mask.size()) + (x.size(-1), ))
    out[mask] = x
    return out

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)
######################################### END UTILS #########################

class GraphMatcher(torch.nn.Module):
    r"""
    The *Graph Matcher* module which first matches nodes
    locally via a graph neural network :math:`\Psi_{\theta_1}`, and then
    updates correspondence scores iteratively.

    Args:
        psi_1 (torch.nn.Module): The first GNN :math:`\Psi_{\theta_1}` which
            takes in node features :obj:`x`, edge connectivity
            :obj:`edge_index`, and optional edge features :obj:`edge_attr` and
            computes node embeddings.
        psi_2 (torch.nn.Module): The second GNN :math:`\Psi_{\theta_2}`optional decoder
        aggregation: str: ('mean' | 'cat') Embedding representation for coalescing Gs, Gt embeddings
    """
    def __init__(self, psi_1, psi_2=None, gembd_vec_dim=96,
                 aggregation='mean', detach=False):
        super(GraphMatcher, self).__init__()

        self.psi_1 = psi_1
        self.psi_2 = psi_2
        self.gembd_vec_dim = gembd_vec_dim
        self.aggregation = aggregation
        self.detach = detach
        mlp_in_sz = psi_1.out_channels * 2      # default
        if self.aggregation == 'mean':
            mlp_in_sz = psi_1.out_channels
        self.mlp = Seq(
            Lin(mlp_in_sz, psi_1.out_channels),
            ReLU(),
            Lin(psi_1.out_channels, gembd_vec_dim),
        )

    def reset_parameters(self):
        self.psi_1.reset_parameters()
        if self.psi_2:
            self.psi_2.reset_parameters()
        self.reset(self.mlp)

    def forward(self, x_s, edge_index_s, edge_attr_s, batch_s,
                        x_t, edge_index_t, edge_attr_t, batch_t, y=None):
        r"""
        Args:
            x_s (Tensor): Source graph node features of shape
                :param batch_t:
                :obj:`[batch_size * num_nodes, C_in]`.
            edge_index_s (LongTensor): Source graph edge connectivity of shape
                :obj:`[2, num_edges]`.
            edge_attr_s (Tensor): Source graph edge features of shape
                :obj:`[num_edges, D]`. Set to :obj:`None` if the GNNs are not
                taking edge features into account.
            batch_s (LongTensor): Source graph batch vector of shape
                :obj:`[batch_size * num_nodes]` indicating node to graph
                assignment. Set to :obj:`None` if operating on single graphs.
            x_t (Tensor): Target graph node features of shape
                :obj:`[batch_size * num_nodes, C_in]`.
            edge_index_t (LongTensor): Target graph edge connectivity of shape
                :obj:`[2, num_edges]`.
            edge_attr_t (Tensor): Target graph edge features of shape
                :obj:`[num_edges, D]`. Set to :obj:`None` if the GNNs are not
                taking edge features into account.
            batch_s (LongTensor): Target graph batch vector of shape
                :obj:`[batch_size * num_nodes]` indicating node to graph
                assignment. Set to :obj:`None` if operating on single graphs.
            y (LongTensor, optional): Ground-truth matchings of shape
                :obj:`[2, num_ground_truths]` to include ground-truth values
                when training against sparse correspondences. Ground-truths
                are only used in case the model is in training mode.
                (default: :obj:`None`)

        Returns: A joint embedding vector for Gs, Gt of shape `[bsz * gembd_vec_dim]`
        """
        h_s = self.psi_1(x_s, edge_index_s, edge_attr_s)
        h_t = self.psi_1(x_t, edge_index_t, edge_attr_t)
        h_s, h_t = (h_s.detach(), h_t.detach()) if self.detach else (h_s, h_t)
        h_s, s_mask = to_dense_batch(h_s, batch_s, fill_value=0)
        h_t, t_mask = to_dense_batch(h_t, batch_t, fill_value=0)    # [64, 50, 48] or [bsz, Vt, dim]
        assert h_s.size(0) == h_t.size(0), 'Encountered unequal batch-sizes'

        (B, N_s, C_out), N_t = h_s.size(), h_t.size(1)
        S_hat = h_s @ h_t.transpose(-1, -2)
        # Use S_hat to map any node func in L(Gs) -> L(Gt)
        S_mask = s_mask.view(B, N_s, 1) & t_mask.view(B, 1, N_t)
        S_0 = masked_softmax(S_hat, S_mask)  # [64, 17, 50]
        r_s = S_0 @ h_t
        h_st = torch.cat((h_s, r_s), dim=2)
        # --------------------------------------------------------- #
        if self.aggregation == 'mean':
            h_st = torch.mean(h_st, dim=1).squeeze()
        out = self.mlp(h_st)

        return out, S_0

    def reset(self, nn):
        def _reset(item):
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()
        if nn is not None:
            if hasattr(nn, 'children') and len(list(nn.children())) > 0:
                for item in nn.children():
                    _reset(item)
            else:
                _reset(nn)

    def __repr__(self):
        return ('{}(\n'
                '    psi_1={},\n'
                '    psi_2={},\n'
                '    gembd_vec_dim={}\n').format(self.__class__.__name__,
                                                    self.psi_1, self.psi_2,
                                                    self.gembd_vec_dim)
