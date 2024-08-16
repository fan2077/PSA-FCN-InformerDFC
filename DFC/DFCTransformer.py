import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import numpy as np

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
valid_len=torch.tensor([30]).to(device)


#@save
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


#@save
class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

#@save
class EncoderBlock(nn.Module):
    """transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens=valid_len):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


def _init_weights(m):
    if isinstance(m,nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    if isinstance(m,nn.Linear):
        nn.init.trunc_normal_(m.weight,std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
#@save
class TransformerEncoder(d2l.Encoder):
    """transformer编码器"""
    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        # self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        # self.pos_embed=self.pos_encoding(torch.zeros(1,num_hiddens,num_hiddens))
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))
        self.mlp=Mlp(in_features=num_hiddens,hidden_features=1024)
        self.linear=nn.Linear(in_features=160080,out_features=2)

        self.apply(_init_weights)

    def forward(self, X, valid_lens=valid_len, *args):
        # a=self.pos_embed
        # X = (X + a)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        self.mlp(X)
        X=torch.flatten(X,start_dim=1)
        X=self.linear(X)
        return X

# import scipy.io as sio
# data=sio.loadmat("./data27.mat")
# A=data['data']
# print(A.dtype)
#
# B=torch.Tensor(A)
# print(B.dtype)
# # A=torch.Tensor(A)
# B=B.reshape([1,B.shape[0],B.shape[1]])
# print(B)

def create_model2():
    count=24
    encoder=TransformerEncoder(key_size=count,query_size=count,value_size=count,num_hiddens=count,
                           norm_shape=[6670,count],ffn_num_input=count,ffn_num_hiddens=count*4,num_heads=2,num_layers=4,dropout=0.6
                           )

    return encoder
#
input_matrix = torch.ones((1,6670,24))
encoder=create_model2()
print(encoder)
import thop
flops,params=thop.profile(encoder,inputs=(input_matrix,))
print(f"Total FLOPs: {flops}")
print(f"Total parameters: {params}")
x=encoder(input_matrix)

