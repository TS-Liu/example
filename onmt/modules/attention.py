import math
from onmt.modules import layers

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

"""
__AUTHOR__: Alanili
__EMAIL__: waajoenglei@gmail.com
"""

def get_local_mask(length, diagonal=1, cuda=True):
    """
    Args:
        length: a int number
    Returns:
        a Tensor with shape [1, length, length]
    """
    ans = Variable(torch.ones(length, length))
    ans = torch.triu(ans, diagonal).unsqueeze(0)
    if cuda:
        ans = ans.cuda()
    return ans

def get_local_bias(length, diagonal=1, cuda=True):
    """
    Args:
        length: a int number
    Returns:
        a Tensor with shape [1, length, length]
    """
    ans = get_local_bias(length, diagonal, cuda)
    return -1e9 * ans

def embedding_to_padding(embed):
    """
    find padding embedding, which are all zeros, from embed Tensor
    Args:
        embed: [..., depth]
    Returns:
        a float Tensor consist of ones(at padding position) and zeros(at other position)
    """
    embed_sum = torch.sum(embed, -1)
    return torch.eq(embed_sum, 0.0).float()

def get_padding_bias(padding):
    """
    Args:
        padding: a float Tensor with shape [batch, length]
    Returns:
        return a Tensor which will be added to attention logits(before it passed to softmax),
        the shape of this Tensor is [batch, 1, length]
    """
    ans = padding * -1e9
    return torch.unsqueeze(ans, 1)

def get_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4, cuda=True):
    """
    Args:
        length: a int, length of sequence
        channels: a int, size of timing embeddings to create. The number of
        different timescales is equal to channels / 2.
        min_timescale: a float
        max_timescale: a float
    Returns:
        a Tensor with shape [1, length, channels]
    """
    position = Variable(torch.Tensor(range(length)))
    num_timescales = channels // 2
    assert num_timescales * 2 == channels, "embedding channels should be devided by 2"
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (float(num_timescales) - 1))
    inv_timescales = Variable(min_timescale * torch.exp(
        torch.Tensor(range(num_timescales)) * -log_timescale_increment))
    scaled_time = torch.unsqueeze(position, 1) * torch.unsqueeze(inv_timescales, 0)
    # the shape of pos_encoding is [length, channels]
    pos_encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 1)
    pos_encoding = torch.unsqueeze(pos_encoding, 0)
    if cuda:
        pos_encoding = pos_encoding.cuda()
    return pos_encoding

def add_timing(x, min_timescale=1.0, max_timescale=1.0e4):
    """
    add position encoding to input
    Args:
        x: a Tensor with shape [batch, length, channels]
        min_timescale: a float
        max_timescale: a float
    """
    _, length, channels = x.size()
    pos_encoding = get_timing_signal(length, channels, min_timescale, max_timescale)
    return x + pos_encoding

class MultiheadAttention(nn.Module):

    def __init__(self,
                 total_key_depth,
                 total_value_depth,
                 channels,
                 attention_dropout=0.0):
        super(MultiheadAttention, self).__init__()
        self.total_key_depth = total_key_depth
        self.input_query_transform = nn.Linear(channels, total_key_depth)
        self.input_key_transform = nn.Linear(channels, total_key_depth)
        self.input_value_transform = nn.Linear(channels, total_value_depth)
        self.attention_softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_transform = nn.Linear(total_value_depth, channels)

    def split_heads(self, x, num_heads):
        """
        Args:
            x: a Tensor with shape [batch, length, channels]
            num_heads: a int number, channels of x should be devided by num_heads
        Returns:
            ans: a Tensor with shape [batch * num_heads, length, channels // num_heads]
        """
        batch, length, channels = x.size()
        assert channels % num_heads == 0, (
               "channels of the input should be devided by num_heads")
        new_dim = channels // num_heads
        ans = x.view(batch, length, num_heads, new_dim).transpose(1, 2)
        return ans

    def combie_heads(self, x, num_heads):
        """
        A reverse process of split_heads function
        Args:
            x: a Tensor with shape [batch * num_heads, length, last_dim]
            num_heads: a int number
        Returns:
            ans: a Tensor with shape [batch, length, last_dim * num_heads]
        """
        batch, _, length, new_dim = x.size()
        ans = x.transpose(1, 2).contiguous().view(batch,
                                    length, num_heads * new_dim)
        return ans

    def forward(self,
                query_antecedent,
                memroy_antecedent,
                num_heads,
                bias):
        """
        Args:
            query_antecedent: a Tensor with shape [batch, length_q, channels]
            memroy_antecedent: a Tensor with shape [batch, length_kv, channels]
            bias: bias Tensor with shape [batch, 1, length_kv] 
                  or [batch, length_q, length_kv]
            num_heads: a int number
        Returns:
            the result of the attention transformation, shape is [batch, length_q, channels]
        """
        if memroy_antecedent is None:
            memroy_antecedent = query_antecedent
        batch_size, query_len, _ = query_antecedent.size()
        _, key_len, _ = memroy_antecedent.size()
        q = self.input_query_transform(query_antecedent)
        k = self.input_key_transform(memroy_antecedent)
        v = self.input_value_transform(memroy_antecedent)
        q = self.split_heads(q, num_heads)
        k = self.split_heads(k, num_heads)
        v = self.split_heads(v, num_heads)
        key_depth_per_head = self.total_key_depth // num_heads
        q = q / math.sqrt(key_depth_per_head)
        logits = torch.matmul(q, k.transpose(2, 3))
        if bias is not None:
            bias = bias.unsqueeze(1).expand_as(logits)
            logits += bias
        attn = self.attention_softmax(logits)
        drop_attn = self.attention_dropout(attn)
        x = torch.matmul(drop_attn, v)
        top_attn = attn.view(batch_size, num_heads,
                    query_len, key_len)[:, 0, :, :].contiguous()
        x = self.combie_heads(x, num_heads)
        return self.output_transform(x), top_attn
