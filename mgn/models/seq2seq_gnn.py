#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File: seq2seq_gnn.py
# Author: anon
# Email: anon@cs.anon.edu
# Created on: 2020-05-04
# 
# This file is part of MGN
# Distributed under terms of the MIT License

import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data, Batch

class Seq2seqGNN(nn.Module):
    """Seq2seqGNN model module
    To do: add docstring to methods
    """

    def __init__(self, encoder, decoder, gnn=None):
        super(Seq2seqGNN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.gnn = gnn

    def _gnn_forward(self, g_data):
        """
        Two feeding mechanisms:
        1. End-to-end: GNN (GraphMatcher, GIN(gs), GIN(gt)) are learnt
        2. Pre-trained: pretrained embedding vector (dim=g_embd_dim) is directly used.

        :param g_data: Graph data
        :return: g_embd, the joint Gs, Gt embedding used for augmenting seq2seq
        """
        is_end2end_flow = self.gnn and type(g_data) == tuple and type(g_data[0]).__name__ == 'Batch'
        Phi = None
        if is_end2end_flow:
            BATCH_S, BATCH_T = g_data
            x_s, edge_index_s, edge_attr_s, batch_s = BATCH_S.x, BATCH_S.edge_index, BATCH_S.edge_attr, BATCH_S.batch
            x_t, edge_index_t, edge_attr_t, batch_t = BATCH_T.x, BATCH_T.edge_index, BATCH_T.edge_attr, BATCH_T.batch
            g_embd = self.gnn(x_s, edge_index_s, edge_attr_s, batch_s,
                               x_t, edge_index_t, edge_attr_t, batch_t)
        else:
            g_embd = g_data  # when pre-trained g_embd is directed used as embd vec
        # Graphmatcher only returning S
        return g_embd, Phi

    def forward(self, x, y, g_data, input_lengths=None):
        """
        Notes:
        1. Feed the g_data -> GNN -> g_embeds
        2. Use g_embeds [1] as usual. Keep g_embeds dim same for now (i.e. 96)

        :param x: questions
        :param y: programs (ground truths)
        :param g_data: graph data (Gs, Gt or Gu: variants)
        :param input_lengths: len of x, used for RNN (un)packing
        :return: decoder outputs
        """
        g_embd, Phi = self._gnn_forward(g_data)
        # g_embd should be the embeddings of only the corresponding vectors not all the nodes
        # in Gs, Gt.
        #  : Keeping encoder, decoder the same)
        encoder_outputs, encoder_hidden = self.encoder(x, g_embd, input_lengths)
        decoder_outputs, decoder_hidden = self.decoder(y, encoder_outputs, encoder_hidden)
        return decoder_outputs

    def sample_output(self, x, g_data, input_lengths=None):
        g_embd, Phi = self._gnn_forward(g_data)

        encoder_outputs, encoder_hidden = self.encoder(x, g_embd, input_lengths)
        output_symbols, _ = self.decoder.forward_sample(encoder_outputs, encoder_hidden)
        return torch.stack(output_symbols).transpose(0, 1)

    def reinforce_forward(self, x, g_data, input_lengths=None):
        g_embd, Phi = self._gnn_forward(g_data)

        encoder_outputs, encoder_hidden = self.encoder(x, g_embd, input_lengths)
        self.output_symbols, self.output_logprobs = self.decoder.forward_sample(encoder_outputs, encoder_hidden,
                                                                                reinforce_sample=True)
        return torch.stack(self.output_symbols).transpose(0, 1)

    def reinforce_backward(self, reward, entropy_factor=0.0):
        assert self.output_logprobs is not None and self.output_symbols is not None, 'must call reinforce_forward first'
        losses = []
        grad_output = []
        # the output_symbols, output_logprobs were calculated in the reinforce_forward step.
        for i, symbol in enumerate(self.output_symbols):
            if len(self.output_symbols[0].shape) == 1:  # one-dim index values
                logprob_pred_symbol = torch.index_select(self.output_logprobs[i], 1, symbol)
                logprob_pred = torch.diag(logprob_pred_symbol).sum()
                probs_i = torch.exp(self.output_logprobs[i])
                plogp = self.output_logprobs[i] * probs_i
                entropy_offset = entropy_factor * plogp.sum()

                loss = - logprob_pred * reward + entropy_offset
                print(f"i = {i}: loss = - {logprob_pred} * {reward} + {entropy_offset} = {loss}")
            else:
                loss = - self.output_logprobs[i] * reward
            losses.append(loss.sum())
            grad_output.append(None)
        torch.autograd.backward(losses, grad_output, retain_graph=True)

    def __repr__(self):
        return ('{}(\n'
                '   encoder={}\n'
                '   decoder={}\n'
                '   gnn={}\n)').format(self.__class__.__name__,
                                       self.encoder,
                                       self.decoder,
                                       self.gnn)
