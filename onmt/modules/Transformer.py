# -*- coding: utf-8 -*-
"""
__AUTHOR__: Alanili
__EMAIL__: waajoenglei@gmail.com
"""
from onmt.modules import attention
from onmt.modules import layers
from onmt.Models import EncoderBase
from onmt.Models import DecoderState
from onmt.Utils import aeq

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

        Args:
            sizbe (int): the size of input for the first-layer of the FFN.
            hidden_size (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """
    def __init__(self, size, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, size)
        self.layer_norm = onmt.modules.LayerNorm(size)
        # Save a little memory, by doing inplace.
        self.dropout_1 = nn.Dropout(dropout, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class EncoderLayer(nn.Module):
    """
    sigle encoder layer used for Encoder
    """
    def __init__(self, hidden_size, dropout,
                 num_heads=8, filter_size=2048):
        super(EncoderLayer, self).__init__()
        self.num_heads = num_heads
        self.ma = attention.MultiheadAttention(hidden_size,
                                               hidden_size,
                                               hidden_size,
                                               dropout)
        self.ffn = layers.ffn_layer(hidden_size,
                                    filter_size,
                                    hidden_size,
                                    dropout)
        self.ma_prenorm = layers.LayerNorm(hidden_size)
        self.ffn_prenorm = layers.LayerNorm(hidden_size)
        self.ma_postdropout = nn.Dropout(dropout)
        self.ffn_postdropout = nn.Dropout(dropout)

    def forward(self, x, bias):
        """
        Args:
            x: current input Tensor with shape [batch, length, channels]
            bias: self attention bias with shape [batch, 1, length]
            pv: previous value
        Returns:
            the transformed result with shape as x
        """
        # multihead attention
        y, _ = self.ma(self.ma_prenorm(x), None,
                    self.num_heads, bias)
        x = self.ma_postdropout(y) + x
        y = self.ffn(self.ffn_prenorm(x))
        ans = self.ffn_postdropout(y) + x
        return ans

class DecoderLayer(EncoderBase):

    def __init__(self, hidden_size, dropout,
                 num_heads=8, filter_size=2048):
        super(DecoderLayer, self).__init__()
        self.num_heads = num_heads
        self.ma_l1 = attention.MultiheadAttention(hidden_size,
                                                  hidden_size,
                                                  hidden_size,
                                                  dropout)
        self.ma_l2 = attention.MultiheadAttention(hidden_size,
                                                  hidden_size,
                                                  hidden_size,
                                                  dropout)
        self.ffn = layers.ffn_layer(hidden_size,
                                    filter_size,
                                    hidden_size,
                                    dropout)
        self.ma_l1_prenorm = layers.LayerNorm(hidden_size)
        self.ma_l2_prenorm = layers.LayerNorm(hidden_size)
        self.ffn_prenorm = layers.LayerNorm(hidden_size)
        self.ma_l1_postdropout = nn.Dropout(dropout)
        self.ma_l2_postdropout = nn.Dropout(dropout)
        self.ffn_postdropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, self_attention_bias,
                encoder_decoder_bias, previous_input=None):
 
        # self multihead attention
        norm_x = self.ma_l1_prenorm(x)
        all_inputs = norm_x
        if previous_input is not None:
            all_inputs = torch.cat((previous_input, norm_x), dim=1)
            self_attention_bias = None
        y, _ = self.ma_l1(norm_x, all_inputs, self.num_heads, self_attention_bias)
        x = self.ma_l1_postdropout(y) + x
        # encoder decoder multihead attention
        y, attn = self.ma_l2(self.ma_l2_prenorm(x), encoder_output,
                                    self.num_heads, encoder_decoder_bias)
        x = self.ma_l2_postdropout(y) + x
        # ffn layer
        y = self.ffn(self.ffn_prenorm(x))
        ans = self.ffn_postdropout(y) + x
        return ans, attn, all_inputs


class Unk_DecoderLayer(EncoderBase):

    def __init__(self, hidden_size, dropout,
                 num_heads=8, filter_size=2048):
        super(Unk_DecoderLayer, self).__init__()
        self.num_heads = num_heads
        self.ma_l1 = attention.MultiheadAttention(hidden_size,
                                                  hidden_size,
                                                  hidden_size,
                                                  dropout)
        self.ma_l2 = attention.MultiheadAttention(hidden_size,
                                                  hidden_size,
                                                  hidden_size,
                                                  dropout)
        self.ffn = layers.ffn_layer(hidden_size,
                                    filter_size,
                                    hidden_size,
                                    dropout)
        self.ma_l1_prenorm = layers.LayerNorm(hidden_size)
        self.ma_l2_prenorm = layers.LayerNorm(hidden_size)
        self.ffn_prenorm = layers.LayerNorm(hidden_size)
        self.ma_l1_postdropout = nn.Dropout(dropout)
        self.ma_l2_postdropout = nn.Dropout(dropout)
        self.ffn_postdropout = nn.Dropout(dropout)

    def forward(self, x, pre_layer_hiden, encoder_output, self_attention_bias,
                encoder_decoder_bias, previous_input=None):
        # self multihead attention
        norm_x = self.ma_l1_prenorm(x)
        all_inputs = norm_x
        previous_input = None
        if previous_input is not None:
            all_inputs = torch.cat((previous_input, norm_x), dim=1)
            self_attention_bias = None
        y, _ = self.ma_l1(norm_x, pre_layer_hiden, self.num_heads, self_attention_bias)
        x = self.ma_l1_postdropout(y) + x
        # encoder decoder multihead attention
        y, attn = self.ma_l2(self.ma_l2_prenorm(x), encoder_output,
                             self.num_heads, encoder_decoder_bias)
        x = self.ma_l2_postdropout(y) + x
        # ffn layer
        y = self.ffn(self.ffn_prenorm(x))
        ans = self.ffn_postdropout(y) + x
        return ans, attn, all_inputs


class TransformerEncoder(EncoderBase):
    """
    encoder of transformer
    """

    def __init__(self, num_layers, hidden_size,
                 dropout, embeddings):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.layer_stack = nn.ModuleList([
            EncoderLayer(hidden_size, dropout) for _ in range(num_layers)])
        self.layer_norm = layers.LayerNorm(hidden_size)

    def forward(self, input, lengths=None, hidden=None):
        """ See :obj:`EncoderBase.forward()`"""
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, n_batch, emb_dim = emb.size()

        out = emb.transpose(0, 1).contiguous()
        words = input[:, :, 0].transpose(0, 1)
        # CHECKS
        out_batch, out_len, _ = out.size()
        w_batch, w_len = words.size()
        aeq(out_batch, w_batch)
        aeq(out_len, w_len)
        # END CHECKS

        # Make mask.
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).float()
        bias = Variable(torch.unsqueeze(mask * -1e9, 1))
        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            out = self.layer_stack[i](out, bias)
        out = self.layer_norm(out)

        return Variable(emb.data), out.transpose(0, 1).contiguous()

class TransformerDecoder(nn.Module):
    """
    decoder of transformer
    """
    def __init__(self, num_layers, hidden_size, attn_type,
                 copy_attn, dropout, embeddings):
        super(TransformerDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'transformer'
        self.num_layers = num_layers
        self.embeddings = embeddings

        self.layer_stack = nn.ModuleList([
            DecoderLayer(hidden_size, dropout) for _ in range(num_layers)])

        # TransformerDecoder has its own attention mechanism.
        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type)
            self._copy = True
        self.layer_norm = layers.LayerNorm(hidden_size)

    def forward(self, tgt, memory_bank, state, memory_lengths=None):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        """
        # CHECKS
        assert isinstance(state, TransformerDecoderState)
        tgt_len, tgt_batch, _ = tgt.size()
        memory_len, memory_batch, _ = memory_bank.size()
        aeq(tgt_batch, memory_batch)

        src = state.src
        src_words = src[:, :, 0].transpose(0, 1)
        tgt_words = tgt[:, :, 0].transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()
        aeq(tgt_batch, memory_batch, src_batch, tgt_batch)
        aeq(memory_len, src_len)

        if state.previous_input is not None:
            tgt = torch.cat([state.previous_input, tgt], 0)
        # END CHECKS

        # Initialize return variables.
        outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(tgt)
        if state.previous_input is not None:
            emb = emb[state.previous_input.size(0):, ]
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        padding_idx = self.embeddings.word_padding_idx
        src_pad_mask = Variable(src_words.data.eq(padding_idx).float())
        tgt_pad_mask = Variable(tgt_words.data.eq(padding_idx).float().unsqueeze(1))
        tgt_pad_mask = tgt_pad_mask.repeat(1, tgt_len, 1)
        encoder_decoder_bias = torch.unsqueeze(src_pad_mask * -1e9, 1)
        decoder_local_mask = attention.get_local_mask(tgt_len) #[1, length, length]
        decoder_local_mask = decoder_local_mask.repeat(tgt_batch, 1, 1)
        decoder_bias = torch.gt(tgt_pad_mask + decoder_local_mask, 0).float() * -1e9

        saved_inputs = []
        for i in range(self.num_layers):
            prev_layer_input = None
            if state.previous_input is not None:
                prev_layer_input = state.previous_layer_inputs[i]
            output, attn, all_input \
                = self.layer_stack[i](output, src_memory_bank, decoder_bias,
                               encoder_decoder_bias, previous_input=prev_layer_input)
            saved_inputs.append(all_input)

        saved_inputs = torch.stack(saved_inputs)
        output = self.layer_norm(output)

        # Process the result and update the attentions.
        outputs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        attns["std"] = attn
        if self._copy:
            attns["copy"] = attn

        # Update the state.
        state = state.update_state(tgt, saved_inputs)
        return emb, outputs, state, attns

    def init_decoder_state(self, src, memory_bank, enc_hidden):
        return TransformerDecoderState(src)


class Unk_TransformerDecoder(nn.Module):
    """
    decoder of transformer
    """
    def __init__(self, num_layers, hidden_size, attn_type,
                 copy_attn, dropout, embeddings):
        super(Unk_TransformerDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'transformer'
        self.num_layers = 1
        self.embeddings = embeddings

        self.layer_stack = nn.ModuleList([
            Unk_DecoderLayer(hidden_size, dropout) for _ in range(num_layers)])

        # TransformerDecoder has its own attention mechanism.
        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type)
            self._copy = True
        self.layer_norm = layers.LayerNorm(hidden_size)

    def forward(self, tgt, pre_layer_emb, pre_layer_hiden, memory_bank, state, memory_lengths=None):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        """
        # CHECKS
        src = state.src
        assert isinstance(state, TransformerDecoderState)
        tgt_len, tgt_batch, _ = tgt.size()
        memory_len, memory_batch, _ = memory_bank.size()
        aeq(tgt_batch, memory_batch)

        emb = self.embeddings(tgt)
        _, __, hiden_len = emb.size()
        state.previous_input = None
        if state.previous_input is not None:
            emb = emb[state.previous_input.size(0):, ]
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()

        src_words = src[:, :, 0].transpose(0, 1)
        tgt_words = tgt[:, :, 0].transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()
        aeq(tgt_batch, memory_batch, src_batch, tgt_batch)
        aeq(memory_len, src_len)

        if state.previous_input is not None:
            tgt = torch.cat([state.previous_input, tgt], 0)
        # END CHECKS

        # Initialize return variables.
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []

        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        padding_idx = self.embeddings.word_padding_idx
        unk_idx = self.embeddings.word_unk_idx    ########
        src_pad_mask = Variable(src_words.data.eq(padding_idx).float())
        tgt_pad_mask = Variable(tgt_words.data.eq(padding_idx).float().unsqueeze(1))
        tgt_pad_mask = tgt_pad_mask.repeat(1, tgt_len, 1)
        tgt_unk_mask = Variable(tgt_words.data.eq(unk_idx).float().unsqueeze(1))   ########
        tgt_no_unk_mask = Variable(tgt_words.data.ne(unk_idx).float().unsqueeze(1))  ########

        # unk_mask = tgt_unk_mask.view(tgt_batch,tgt_len).transpose(0,1).contiguous()
        # out = []
        # for i in range(tgt_batch):
        #     out.append(output[i].masked_select(unk_mask[i]))
        # unk_bias = Variable(-torch.ones(tgt_batch, tgt_len))


        output = output
                 #*(tgt_unk_mask.repeat(1, hiden_len, 1).transpose(1,2)) + pre_layer_emb.transpose(0,1)*(tgt_no_unk_mask.repeat(1, hiden_len, 1))

        tgt_unk_mask = tgt_unk_mask.repeat(1, tgt_len, 1)   ########（2*11*11）
        # unk_bias = Variable(tgt_words.data.eq(unk_idx).float().unsqueeze(1))      ########
        # tgt = tgt.masked_select(unk_bias)
        # unk_bias = torch.unsqueeze(unk_bias * -1e9, 1)   ########
        encoder_decoder_bias = torch.unsqueeze(src_pad_mask * -1e9, 1)
        decoder_bias = torch.gt(tgt_pad_mask + tgt_unk_mask, 0).float() * -1e9

        saved_inputs = []
        for i in range(self.num_layers):
            prev_layer_input = None
            if state.previous_input is not None:
                prev_layer_input = state.previous_layer_inputs[i]
            output, attn, all_input \
                = self.layer_stack[i](output, pre_layer_hiden.transpose(0,1), src_memory_bank, decoder_bias,
                               encoder_decoder_bias, previous_input=prev_layer_input)
            saved_inputs.append(all_input)

        saved_inputs = torch.stack(saved_inputs)
        output = self.layer_norm(output)

        # Process the result and update the attentions.
        outputs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        attns["std"] = attn
        if self._copy:
            attns["copy"] = attn


        # Update the state.
        state = state.update_state(tgt, saved_inputs)
        return outputs, state, attns

    def init_decoder_state(self, src, memory_bank, enc_hidden):
        return TransformerDecoderState(src)


class TransformerDecoderState(DecoderState):
    def __init__(self, src):
        """
        Args:
            src (FloatTensor): a sequence of source words tensors
                    with optional feature tensors, of size (len x batch).
        """
        self.src = src
        self.previous_input = None
        self.previous_layer_inputs = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        return (self.previous_input, self.previous_layer_inputs, self.src)

    def update_state(self, input, previous_layer_inputs):
        """ Called for every decoder forward pass. """
        state = TransformerDecoderState(self.src)
        state.previous_input = input
        state.previous_layer_inputs = previous_layer_inputs
        return state

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = Variable(self.src.data.repeat(1, beam_size, 1),
                            volatile=True)
