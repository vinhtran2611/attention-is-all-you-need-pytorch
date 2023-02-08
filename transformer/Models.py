''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import copy

from utils import clones, Generator
from transformer.layers.layer_norm import LayerNorm
from transformer.layers.position_wise_feed_forward import PositionwiseFeedForward
from transformer.blocks.encode_layer import EncoderLayer
from transformer.blocks.decode_layer import DecoderLayer
from transformer.layers.multi_head_attention import MultiHeadedAttention
from embedding.position_encoding import PositionalEncoding
from embedding.token_embedding import Embeddings



class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N) # return nn.ModuleList
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        enc = self.encode(src, src_mask)
        return self.decode(enc, src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.dropout = dropout
        

    def forward(self, src_vocab, tgt_vocab):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.h, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        model = EncoderDecoder(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.N),
            Decoder(DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout), self.N),
            nn.Sequential(Embeddings(self.d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)),
            Generator(self.d_model, tgt_vocab),
        )

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model