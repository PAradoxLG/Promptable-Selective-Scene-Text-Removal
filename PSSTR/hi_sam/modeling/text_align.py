import editdistance
import random
import math
import numpy as np
import torch
from torch import nn
import string
from typing import Union


class AbstractEmbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_trainable = None
        self._ucg_rate = None
        self._input_key = None
        self._emb_key = None

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @property
    def ucg_rate(self) -> Union[float, torch.Tensor]:
        return self._ucg_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @property
    def emb_key(self) -> str:
        return self._emb_key

    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    @ucg_rate.setter
    def ucg_rate(self, value: Union[float, torch.Tensor]):
        self._ucg_rate = value

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @emb_key.setter
    def emb_key(self, value: str):
        self._emb_key = value

    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    @ucg_rate.deleter
    def ucg_rate(self):
        del self._ucg_rate

    @input_key.deleter
    def input_key(self):
        del self._input_key

    @emb_key.deleter
    def emb_key(self):
        del self._emb_key


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.tile(self.pe[None, ...].to(x.device), (x.shape[0], 1, 1))
        return self.dropout(x)


class WordEmbedding(AbstractEmbModel):

    # def __init__(self, max_len, emb_dim, out_emb_dim, n_heads=8, n_trans_layers=6, trainable=False,
    #              lr=1e-4, lambda_cls=0.1, lambda_pos=0.1, clip_dim=1024, visual_len=197, visual_dim=768,
    #              visual_config=None, *args, **kwargs):
    def __init__(self, max_len, emb_dim, n_heads=8, n_trans_layers=12, trainable=False,
                 lr=1e-4, lambda_cls=0.1, lambda_pos=0.1, clip_dim=1024, visual_len=197, visual_dim=768,
                 visual_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_len = max_len
        self.emd_dim = emb_dim
        # self.out_emb_dim = out_emb_dim
        self.n_heads = n_heads
        self.n_trans_layers = n_trans_layers
        # lg20250305
        # self.character = string.printable[:62]
        self.character = string.printable[:-6]
        self.num_cls = len(self.character) + 1

        self.label_embedding = nn.Embedding(self.num_cls, self.emd_dim)
        self.pos_embedding = PositionalEncoding(d_model=self.emd_dim, max_len=self.max_len)
        transformer_block = nn.TransformerEncoderLayer(d_model=self.emd_dim, nhead=self.n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(transformer_block, num_layers=self.n_trans_layers)
        # self.out_project = nn.Linear(emb_dim, out_emb_dim, bias=False)

        # if ckpt_path is not None:
        #     self.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)  


    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def get_index(self, labels):

        indexes = []
        for label in labels:
            if len(label) > self.max_len:
                label = label[:self.max_len]
            index = [self.character.find(c) + 1 for c in label]
            index = index + [0] * (self.max_len - len(index))
            indexes.append(index)

        return torch.tensor(indexes, device=next(self.parameters()).device)

    def get_embeddings(self, x):

        emb = self.label_embedding(x)
        emb = self.pos_embedding(emb)
        out = self.encoder(emb)
        # out = self.out_project(out)

        return out

    def forward(self, labels):

        idx = self.get_index(labels)
        out = self.get_embeddings(idx)

        return out

    def encode(self, text):
        return self(text)

    

def main():
    from thop import profile

    out_channels = 256
    embedder = WordEmbedding(
        max_len=28,
        emb_dim=out_channels,
    ).cuda()
    # 计算总参数数量
    total_params = sum(p.numel() for p in embedder.parameters())

    print(f"Total parameters: {total_params/1e6:.2f} M")

    texts = ['hello', 'world', 'nihao']
    idx = embedder.get_index(texts)
    print("idx.shape >> ", idx.shape)
    word_embedding = embedder(texts)

    flops, params = profile(embedder, inputs=(texts,))
    print(f"FLOPs: {flops:.2f}G")
    print(f"Parameters: {params:.2f}M")

    print(word_embedding.shape)


if __name__ == '__main__':
    main()