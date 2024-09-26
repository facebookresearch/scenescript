# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import IntEnum

import torch
import torch.nn as nn
from einops import repeat


class HELPER_TOKEN(IntEnum):
    PAD = 0
    START = 1
    PART = 2
    STOP = 3
    NOT_USED = 4
    NOT_USED_1 = 5
    NUM = 6


def make_autoregressive_mask(size, device=None):
    # Generates an upper-triangular matrix of -inf, with zeros on diag.
    # Example size=5:
    # [[0., -inf, -inf, -inf, -inf],
    #  [0.,   0., -inf, -inf, -inf],
    #  [0.,   0.,   0., -inf, -inf],
    #  [0.,   0.,   0.,   0., -inf],
    #  [0.,   0.,   0.,   0.,   0.]]
    return torch.triu(torch.ones(size, size, device=device) * float("-inf"), diagonal=1)


class SceneScriptDecoder(nn.Module):
    def __init__(
        self,
        d_model,
        num_attn_heads,
        dim_feedforward,
        num_bins,
        max_num_tokens,
        max_num_type_tokens,
        num_decoder_layers,
    ):
        """
        Args:
            d_model: int. Dimension of model.
            num_attn_heads: int. Number of attention heads.
            dim_feedforward: int. Dimension of feedforward network.
            num_bins: int. Number of discretized bins.
            max_num_tokens: int. Maximum number of tokens.
            max_num_type_tokens: int. Maximum number of type tokens.
            num_decoder_layers: int. Number of decoder layers.
        """
        super().__init__()
        self.d_model = d_model
        self.max_num_tokens = max_num_tokens

        # Embeddings
        self.position_embedding = nn.Embedding(max_num_tokens, d_model)
        self.value_embedding = nn.Embedding(num_bins + HELPER_TOKEN.NUM, d_model)
        self.type_embedding = nn.Embedding(max_num_type_tokens, d_model)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            num_attn_heads,
            dim_feedforward,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_decoder_layers, nn.LayerNorm(d_model)
        )

        # Decoding to bins
        self.tail = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_bins + HELPER_TOKEN.NUM),
        )

        self.register_buffer("causal_mask", make_autoregressive_mask(max_num_tokens))

    def embed_position(self, seq_value):
        """Apply positional embedding.

        Args:
            seq_value: [B, T] torch.LongTensor. In range [0, num_bins + HELPER_TOKEN.NUM).

        Returns:
            pos_emb: [B, T, d_model] torch.FloatTensor.
        """
        B, T = seq_value.shape
        device = seq_value.device

        # Target embedding
        t = torch.arange(T, device=device)
        pos_emb = repeat(self.position_embedding(t), "t d -> b t d", b=B)

        return pos_emb

    def forward(self, context, context_mask, seq_value, seq_type):
        """
        Args:
            context: [B, context_length, d_model] torch.FloatTensor.
            context_mask: [B, context_length] torch.BoolTensor. True means ignore.
            seq_value: [B, T] torch.LongTensor. In range [0, num_bins + HELPER_TOKEN.NUM).
            seq_type: [B, T] torch.LongTensor. In range [0, max_num_type_tokens)

        Returns:
            [B, T, num_bins + HELPER_TOKEN.NUM] torch.FloatTensor.
        """
        B, T = seq_value.shape[:2]

        decoder_input = (
            self.embed_position(seq_value)
            + self.value_embedding(seq_value)
            + self.type_embedding(seq_type)
        )

        # Get causal_mask
        assert T <= self.max_num_tokens
        causal_mask = repeat(self.causal_mask[:T, :T], "T Y -> B T Y", B=B)

        # transformer
        decoder_out = self.transformer_decoder(
            tgt=decoder_input,
            tgt_mask=causal_mask,
            tgt_is_causal=True,
            memory=context,
            memory_mask=None,
            memory_key_padding_mask=context_mask,
        )  # [B, T, d_model]
        logits = self.tail(decoder_out)  # [B, T, num_bins + HELPER_TOKEN.NUM]

        return logits
