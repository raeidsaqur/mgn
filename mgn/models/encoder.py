import torch
import torch.nn as nn
from .base_rnn import BaseRNN


class Encoder(BaseRNN):
    """Encoder RNN module"""
    
    def __init__(self, vocab_size, max_len, word_vec_dim, hidden_size, n_layers,
                 input_dropout_p=0, dropout_p=0, bidirectional=False, rnn_cell='lstm',
                 variable_lengths=False, word2vec=None, fix_embedding=False, gembd_vec_dim=0):
        super(Encoder, self).__init__(vocab_size, max_len, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell)
        self.variable_lengths = variable_lengths
        if word2vec is not None:
            assert word2vec.size(0) == vocab_size
            self.word_vec_dim = word2vec.size(1)
            self.embedding = nn.Embedding(vocab_size, self.word_vec_dim)
            self.embedding.weight = nn.Parameter(word2vec)
        else:
            self.word_vec_dim = word_vec_dim
            self.embedding = nn.Embedding(vocab_size, word_vec_dim)
        if fix_embedding:
            self.embedding.weight.requires_grad = False

        self.gembd_vec_dim = gembd_vec_dim
        rnn_input_dim = self.word_vec_dim + self.gembd_vec_dim
        self.rnn = self.rnn_cell( rnn_input_dim, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bool(bidirectional), dropout=dropout_p)

    def forward(self, x, g_embd, input_lengths=None):
        """
        :param input_lengths:
        :param x question token sequence  (bsz, max(q_seq_len)) e.g. (64, 41)
        :param g_embd averaged Gs, Gt embedding (bsz, g_embd_vec) e.g. (64, 96)
        To do: add input, output dimensions to docstring
        """
        if self.gembd_vec_dim > 0:
            g_embedding = g_embd.unsqueeze(1).repeat(1, x.size(1), 1)
            word_embedding = self.embedding(x)
            embedded = torch.cat((word_embedding, g_embedding), dim=2)
        else:
            # Baseline model
            embedded = self.embedding(x)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden

    def forward2(self, input_var, input_lengths=None):
        """
                :param input_var question token sequence
                To do: add input, output dimensions to docstring
        """
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden
