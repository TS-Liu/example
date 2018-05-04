import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

"""
__AUTHOR__: Alanili
__EMAIL__: waajoenglei@gmail.com
"""

class ffn_layer(nn.Module):

    def __init__(self,
                 input_size,
                 filter_size,
                 output_size,
                 relu_dropout=0.0):
        super(ffn_layer, self).__init__()
        self.mid_layer = nn.Linear(input_size, filter_size)
        self.out_layer = nn.Linear(filter_size, output_size)
        self.relu = nn.ReLU()
        self.relu_dropout = nn.Dropout(relu_dropout)

    def forward(self, x):
        t = self.relu(self.mid_layer(x))
        o = self.out_layer(self.relu_dropout(t))
        return o

class Bottle(nn.Module):
        def forward(self, input):
            if len(input.size()) <= 2:
                return super().forward(input)
            size = input.size()[:2]
            out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
            return out.contiguous().view(size[0], size[1], -1)

class LayerNorm(nn.Module):
    """
    Layer normalization module 
    """ 

    def __init__(self, depth, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.eps = eps
        self.scale = nn.Parameter(torch.ones(depth), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(depth), requires_grad=True)

    def forward(self, x):
        """
        apply layer norm
        """
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        norm_x = (x - mean) / (std + self.eps)
        return norm_x * self.scale + self.bias

class BottleSoftmax(Bottle, nn.Softmax):
    """ Perform the reshape routine before and after a softmax operation """
    pass
