import torch.nn as nn
from utils import clones, SublayerConnection

class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below)

    Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and 
    the second is a simple, position-wise fully connected feed-forward network
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
