import torch.nn as nn
from utils import clones, SublayerConnection

class DecoderLayer(nn.Module):
    """
    In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, 
    which performs multi-head attention over the output of the encoder stack. 
    Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization.
    
    Decoder is made of self-attn, src-attn, and feed forward (defined below)
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)