from .encoder import Encoder
from .decoder import Decoder
from .gnn import GIN, MLP
from .graph_matcher import GraphMatcher

from .seq2seq import Seq2seq
from .seq2seq_gnn import Seq2seqGNN
import utils.utils as utils

import torch
import torch.nn as nn

import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")

def get_vocab(opt):
    if opt.dataset == 'clevr':
        vocab_json = opt.clevr_vocab_path
    else:
        raise ValueError('Invalid dataset')
    vocab = utils.load_vocab(vocab_json)
    return vocab


def create_seq2seq_net(input_vocab_size, output_vocab_size, hidden_size, 
                       word_vec_dim, n_layers, bidirectional, variable_lengths, 
                       use_attention, encoder_max_len, decoder_max_len, start_id, 
                       end_id, word2vec_path=None, fix_embedding=False, gembd_vec_dim=0):
    word2vec = None
    if word2vec_path is not None:
        word2vec = utils.load_embedding(word2vec_path)

    encoder = Encoder(input_vocab_size, encoder_max_len, 
                      word_vec_dim, hidden_size, n_layers,
                      bidirectional=bidirectional, variable_lengths=variable_lengths,
                      word2vec=word2vec, fix_embedding=fix_embedding, gembd_vec_dim=gembd_vec_dim)
    decoder = Decoder(output_vocab_size, decoder_max_len,
                      word_vec_dim, hidden_size, n_layers, start_id, end_id,
                      bidirectional=bidirectional, use_attention=use_attention)

    return Seq2seq(encoder, decoder)


def create_seq2seq_gnn_net(input_vocab_size, output_vocab_size, hidden_size,
                       word_vec_dim, n_layers, bidirectional, variable_lengths,
                       use_attention, encoder_max_len, decoder_max_len, start_id,
                       end_id, gnn, gembd_vec_dim, is_padding_pos, word2vec_path=None, fix_embedding=False, **kwargs):
    """ Creates a Seq2seqGNN model for e2e training for Gs, Gt embedding """
    word2vec = None
    if word2vec_path is not None:
        word2vec = utils.load_embedding(word2vec_path)

    encoder = Encoder(input_vocab_size, encoder_max_len,
                      word_vec_dim, hidden_size, n_layers,
                      bidirectional=bidirectional, variable_lengths=variable_lengths,
                      word2vec=word2vec, fix_embedding=fix_embedding, gembd_vec_dim=gembd_vec_dim)
    decoder = Decoder(output_vocab_size, decoder_max_len,
                      word_vec_dim, hidden_size, n_layers, start_id, end_id,
                      bidirectional=bidirectional, use_attention=use_attention)

    logging.info(f"Instantiating GNN type: {gnn}")
    in_channels = gembd_vec_dim+3 if is_padding_pos else gembd_vec_dim      # 99 | 96
    psi_1 = GIN(in_channels=in_channels, out_channels=48, num_layers=2)
    gnn = GraphMatcher(psi_1, gembd_vec_dim=gembd_vec_dim, aggregation='cat')

    return Seq2seqGNN(encoder, decoder, gnn)