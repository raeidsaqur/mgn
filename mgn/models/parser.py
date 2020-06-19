#! /usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch_geometric.data import Data, Batch

from . import get_vocab
from . import create_seq2seq_gnn_net

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
from rsmlkit.logging import set_default_level

set_default_level(logging.INFO)
# logger = get_logger(__file__)

class Seq2seqParser():
    """Model interface for seq2seq parser"""

    def __init__(self, opt):
        """Initialize a Seq2seq model by either new initialization
        for pre-training or loading a pre-trained checkpoint for
        fine-tuning using reinforce
        """
        self.opt = opt              # See class TrainOptions for details
        self.vocab = get_vocab(opt)

        if opt.load_checkpoint_path is not None:
            self.load_checkpoint(opt)
        else:
            print('| creating new network')
            self.net_params = self._get_net_params(self.opt, self.vocab)
            self.seq2seq = create_seq2seq_gnn_net(**self.net_params)        # E2E flow seq2seq with GNN

        self.variable_lengths = self.net_params['variable_lengths']
        self.end_id = self.net_params['end_id']
        self.gpu_ids = opt.gpu_ids
        logging.debug(f"opt.gpu_ids: {opt.gpu_ids}")
        self.criterion = nn.NLLLoss()

        if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
            self.seq2seq.cuda(opt.gpu_ids[0])

    def load_checkpoint(self, opt):
        load_path = opt.load_checkpoint_path
        is_baseline = False
        if 'is_baseline_model' in vars(opt):
            is_baseline = opt.is_baseline_model
        logging.info('| loading checkpoint from %s' % load_path)
        checkpoint = torch.load(load_path)
        self.net_params = checkpoint['net_params']
        if 'fix_embedding' in vars(self.opt): # To do: change condition input to run mode
            self.net_params['fix_embedding'] = self.opt.fix_embedding
        _g = 'gembd_vec_dim'
        if is_baseline:
            self.net_params[_g] = 0
        elif self.net_params.get(_g) is None:
            self.net_params[_g] = self.opt.gembd_vec_dim if \
                _g in vars(self.opt) else 96
        else:
            self.net_params[_g] = 96

        self.seq2seq = create_seq2seq_gnn_net(**self.net_params)
        self.seq2seq.load_state_dict(checkpoint['net_state'])

    def save_checkpoint(self, save_path):
        checkpoint = {
            'net_params': self.net_params,
            'net_state': self.seq2seq.cpu().state_dict()
        }
        torch.save(checkpoint, save_path)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            self.seq2seq.cuda(self.gpu_ids[0])

    def set_input(self, x, y=None, g_data=None):
        """Sorts the mini-batch (64) samples by question seq len"""
        input_lengths, idx_sorted = None, None
        if self.variable_lengths:
            x, y, g_data, input_lengths, idx_sorted = self._sort_batch_with_graph(x, y, g_data)
        self.x = self._to_var(x)
        if y is not None:
            self.y = self._to_var(y)
        else:
            self.y = None

        self.g_data = None
        if g_data is not None:
            if type(g_data) == tuple:
                batch_s = self._to_var(g_data[0])
                batch_t = self._to_var(g_data[1])
                self.g_data = (batch_s, batch_t)
            else:
                self.g_data = self._to_var(g_data)

        self.input_lengths = input_lengths
        self.idx_sorted = idx_sorted

    def set_reward(self, reward):
        self.reward = reward

    def supervised_forward(self):
        assert self.y is not None, 'Must set y value'
        # ---- Regular Flow ---- #
        output_logprob = self.seq2seq(self.x, self.y, self.g_data, self.input_lengths)
        self.loss = self.criterion(output_logprob[:,:-1,:].contiguous().view(-1, output_logprob.size(2)), self.y[:,1:].contiguous().view(-1))
        return self._to_numpy(self.loss).sum()

    def supervised_backward(self):
        assert self.loss is not None, 'Loss not defined, must call supervised_forward first'
        self.loss.backward()

    def reinforce_forward(self):
        self.rl_seq = self.seq2seq.reinforce_forward(self.x, self.g_data, self.input_lengths)
        self.rl_seq = self._restore_order(self.rl_seq.data.cpu())
        self.reward = None # Need to recompute reward from environment each time a new sequence is sampled
        return self.rl_seq

    def reinforce_backward(self, entropy_factor=0.0):
        assert self.reward is not None, 'Must run forward sampling and set reward before REINFORCE'
        self.seq2seq.reinforce_backward(self.reward, entropy_factor)

    def parse(self):
        #output_sequence = self.seq2seq.sample_output(self.x, self.input_lengths)
        output_sequence = self.seq2seq.sample_output(self.x, self.g_data, self.input_lengths)
        output_sequence = self._restore_order(output_sequence.data.cpu())
        return output_sequence

    def _get_net_params(self, opt, vocab):
        net_params = {
            'input_vocab_size': len(vocab['question_token_to_idx']),
            'output_vocab_size': len(vocab['program_token_to_idx']),
            'hidden_size': opt.hidden_size,
            'word_vec_dim': opt.word_vec_dim,
            'n_layers': opt.n_layers,
            'bidirectional': opt.bidirectional,
            'variable_lengths': opt.variable_lengths,
            'use_attention': opt.use_attention,
            'encoder_max_len': opt.encoder_max_len,
            'decoder_max_len': opt.decoder_max_len,
            'start_id': opt.start_id,
            'end_id': opt.end_id,
            'word2vec_path': opt.word2vec_path,
            'fix_embedding': opt.fix_embedding,
            'gnn': opt.gnn,
            'gembd_vec_dim': opt.gembd_vec_dim,
            'is_padding_pos': opt.is_padding_pos
        }
        return net_params

    def _sort_batch(self, x, y):
        """Helper to sort a batch by questions seq len (descening) """
        _, lengths = torch.eq(x, self.end_id).max(1)
        lengths += 1
        lengths_sorted, idx_sorted = lengths.sort(0, descending=True)
        x_sorted = x[idx_sorted]
        y_sorted = None
        if y is not None:
            y_sorted = y[idx_sorted]
        lengths_list = lengths_sorted.numpy()
        return x_sorted, y_sorted, lengths_list, idx_sorted

    def _sort_batch_with_graph(self, x, y, g_data=None):
        """Helper to sort a batch by questions seq len (descening) including graph data"""
        _, lengths = torch.eq(x, self.end_id).max(1)
        lengths += 1
        lengths_sorted, idx_sorted = lengths.sort(0, descending=True)
        x_sorted = x[idx_sorted]
        y_sorted = None
        if y is not None:
            y_sorted = y[idx_sorted]
        # Sort graph (Gs, Gt) data objects by idx_sorted #
        g_data_sorted = None
        if g_data is not None:
            logging.info(f"g_data: {g_data}: {type(g_data).__module__}")
            if isinstance(g_data[0],Data):
                # Handle when g_data = (batch_s, batch_t)
                batch_s, batch_t = g_data
                data_s_list = batch_s.to_data_list()
                data_t_list = batch_t.to_data_list()
                assert len(data_s_list) == len(data_t_list)
                data_s_list_sorted = [data_s_list[i] for i in idx_sorted]
                data_t_list_sorted = [data_t_list[i] for i in idx_sorted]
                batch_s_sorted = Batch.from_data_list(data_s_list_sorted)
                batch_t_sorted = Batch.from_data_list(data_t_list_sorted)
                g_data_sorted = (batch_s_sorted, batch_t_sorted)
            else:
                g_data_sorted = g_data[idx_sorted]

        lengths_list = lengths_sorted.numpy()
        return x_sorted, y_sorted, g_data_sorted, lengths_list, idx_sorted

    def _restore_order(self, x):
        if self.idx_sorted is not None:
            inv_idxs = self.idx_sorted.clone()
            inv_idxs.scatter_(0, self.idx_sorted, torch.arange(x.size(0)).long())
            return x[inv_idxs]
        return x

    def _to_var(self, x):
        device = 'cuda' if (len(self.gpu_ids) > 0 and torch.cuda.is_available()) else 'cpu'
        x = x.to(device)
        if type(x).__name__ == 'Batch':
            for k in x.keys:
                var = x[k]
                if torch.is_tensor(var) and torch.is_floating_point(var):
                    var.requires_grad = True
            return x

        return Variable(x)

    def _to_numpy(self, x):
        return x.data.cpu().numpy().astype(float)