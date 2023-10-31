# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Any, Dict, List, Optional, Tuple, NamedTuple
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from scipy.spatial import transform

from esm.data import Alphabet

from features import DihedralFeatures
from gvp_encoder import GVPEncoder
from gvp_utils import unflatten_graph
from gvp_transformer_encoder import GVPTransformerEncoder
from transformer_decoder import TransformerDecoder
from util import rotate, CoordBatchConverter 


class GVPTransformerModel(nn.Module):
    """
    GVP-Transformer inverse folding model.

    Architecture: Geometric GVP-GNN as initial layers, followed by
    sequence-to-sequence Transformer encoder and decoder.
    """

    def __init__(self, args, alphabet,alphabet_go):
        super().__init__()
        encoder_embed_tokens = self.build_embedding(
            args, alphabet, args.encoder_embed_dim,
        )
        self.decoder_embed_tokens = self.build_embedding(
            args, alphabet_go, args.decoder_embed_dim, 
        )
        self.decoder_embed_tokens_mask = torch.tensor(['GO:' in tok for tok in alphabet_go.all_toks]).long()
        self.encoder = self.build_encoder(args, alphabet, encoder_embed_tokens)
        self.decoder = self.build_decoder(args, alphabet_go, self.decoder_embed_tokens)
        self.args = args

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = GVPTransformerEncoder(args, src_dict, embed_tokens)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
        )
        return decoder

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.padding_idx
        emb = nn.Embedding(num_embeddings, embed_dim, padding_idx)
        nn.init.normal_(emb.weight, mean=0, std=embed_dim ** -0.5)
        nn.init.constant_(emb.weight[padding_idx], 0)
        return emb

    def forward(
        self,
        esms,
        coords,
        seqs,
        padding_mask,
        confidence,
        return_all_hiddens: bool = False,
        features_only: bool = False,
    ):
        encoder_out = self.encoder(esms,coords,seqs, padding_mask, confidence,
            return_all_hiddens=return_all_hiddens)
        
        batch_ = esms.size()[0]
        batch_all_tokens = torch.tensor([list(range(len(self.decoder.dictionary))) for i in range(batch_)]).long().to(esms.device)
        logits, extra = self.decoder(
            batch_all_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            return_all_hiddens=return_all_hiddens,
        )
        return logits, extra

