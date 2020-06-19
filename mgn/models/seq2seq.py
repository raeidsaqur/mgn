import torch
import torch.nn as nn


class Seq2seq(nn.Module):
    """Seq2seq model module
    To do: add docstring to methods
    """
    
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y, g_embd, input_lengths=None):
        encoder_outputs, encoder_hidden = self.encoder(x, g_embd, input_lengths)
        decoder_outputs, decoder_hidden = self.decoder(y, encoder_outputs, encoder_hidden)
        return decoder_outputs

    def sample_output(self, x, g_embd, input_lengths=None):
        encoder_outputs, encoder_hidden = self.encoder(x, g_embd, input_lengths)
        output_symbols, _ = self.decoder.forward_sample(encoder_outputs, encoder_hidden)
        return torch.stack(output_symbols).transpose(0,1)

    def reinforce_forward(self, x, g_embd, input_lengths=None):
        encoder_outputs, encoder_hidden = self.encoder(x, g_embd, input_lengths)
        self.output_symbols, self.output_logprobs = self.decoder.forward_sample(encoder_outputs, encoder_hidden, reinforce_sample=True)
        return torch.stack(self.output_symbols).transpose(0,1)

    def reinforce_backward(self, reward, entropy_factor=0.0):
        assert self.output_logprobs is not None and self.output_symbols is not None, 'must call reinforce_forward first'
        losses = []
        grad_output = []
        for i, symbol in enumerate(self.output_symbols):
            if len(self.output_symbols[0].shape) == 1: # one-dim index values
                logprob_pred_symbol = torch.index_select(self.output_logprobs[i], 1, symbol)
                logprob_pred = torch.diag(logprob_pred_symbol).sum()

                probs_i = torch.exp(self.output_logprobs[i])
                plogp = self.output_logprobs[i] * probs_i
                entropy_offset = entropy_factor * plogp.sum()

                loss = - logprob_pred*reward + entropy_offset
                print(f"i = {i}: loss = - {logprob_pred} * {reward} + {entropy_offset} = {loss}")
            else:
                loss = - self.output_logprobs[i]*reward
            losses.append(loss.sum())
            grad_output.append(None)
        torch.autograd.backward(losses, grad_output, retain_graph=True)