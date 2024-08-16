import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
from scipy.stats import zscore


from d2l import torch as d2l

count = 6670
W=60
S=3
numwidows=(170-W)//S+1

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
valid_len=torch.tensor([count]).to(device)



class ProbAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.2, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape


        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / np.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ConvLayer(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(ConvLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


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
        X=X.to(device)
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class EncoderBlock2(nn.Module):
    """transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock2, self).__init__(**kwargs)

        self.attention = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.conv=ConvLayer(d_model=count,dropout=0.5)

    def forward(self, X, valid_lens=valid_len):
        X=X.to(device)
        X = self.conv(self.attention(X, X, X, valid_lens))
        return X

def _init_weights(m):
    if isinstance(m,nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    if isinstance(m,nn.Linear):
        nn.init.trunc_normal_(m.weight,std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
#@save

def sliding_window(data, w, s):
    """使用滑动窗法切割数据"""
    num_windows = (data.shape[1] - w) // s + 1  # 计算切割出的窗口数
    windows = torch.zeros((num_windows, 116, data.shape[2]), device=data.device)  # 初始化窗口数组
    for i in range(num_windows):
        window_data = data[: , i * s:i * s + w , :]  # 切割窗口
        pearson_coef=torch.corrcoef(window_data.squeeze().T)
        pearson_coef=torch.clamp(pearson_coef,-0.999,0.999)
        z_matrix=torch.atanh(pearson_coef)
        windows[i]=z_matrix
    return windows

def flatten_upper_triangular(windows):
    num_windows, w, _ = windows.shape
    flattened = []

    for i in range(num_windows):
        upper_triangular = torch.triu(windows[i], diagonal=1)  # 取上三角部分
        mask = torch.triu(torch.ones_like(upper_triangular), diagonal=1).bool()  # 创建布尔掩码，对角线以下为False
        upper_triangular = upper_triangular[mask]  # 使用掩码筛选非对角线元素
        flattened.append(upper_triangular)

    flattened = torch.stack(flattened)  # 将列表转换为张量
    return flattened



import torch.nn.functional as F

class MaxpoolLayer(nn.Module):
    def __init__(self, c_in):
        super(MaxpoolLayer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, ))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x

class TransformerEncoder2(d2l.Encoder):
    """transformer编码器"""
    def __init__(self,  key_size, query_size, value_size,
                 num_hiddens, num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder2, self).__init__(**kwargs)

        self.Probattention=ProbAttention(mask_flag=False,factor=5,scale=None,attention_dropout=0.2,output_attention=False)
        self.attn_layer=AttentionLayer(attention=self.Probattention,d_model=116,n_heads=4)

        self.num_hiddens = num_hiddens
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock2(key_size, query_size, value_size, num_hiddens,
                             num_heads, dropout, use_bias))
        self.Maxpool=MaxpoolLayer(c_in=count)
        self.linear=nn.Linear(in_features=126730,out_features=2)

        self.apply(_init_weights)

    def forward(self, X, valid_lens=valid_len, *args):
        X=X.to(device)
        X=sliding_window(X,W,S)
        X,attn = self.attn_layer(X,X,X,attn_mask=None)
        X = flatten_upper_triangular(X)
        X=X.unsqueeze(0)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        X=self.Maxpool(X)
        X=torch.flatten(X,start_dim=1)
        X=self.linear(X)
        return X

class TransformerEncoder(d2l.Encoder):
    """transformer编码器"""
    def __init__(self,  key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)

        self.Probattention=ProbAttention(mask_flag=False,factor=5,scale=None,attention_dropout=0.2,output_attention=False)
        self.attn_layer=AttentionLayer(attention=self.Probattention,d_model=116,n_heads=4)

        self.num_hiddens = num_hiddens
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))
        self.mlp=Mlp(in_features=num_hiddens,hidden_features=2*count)
        self.linear=nn.Linear(in_features=numwidows*count,out_features=2)

        self.apply(_init_weights)

    def forward(self, X, valid_lens=valid_len, *args):
        X=X.to(device)
        X=sliding_window(X,W,S)
        X,attn = self.attn_layer(X,X,X,attn_mask=None)
        X = flatten_upper_triangular(X)
        print(X.shape)
        X=X.unsqueeze(0)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        self.mlp(X)
        X=torch.flatten(X,start_dim=1)
        X=self.linear(X)
        return X



# def create_model():
#     encoder=TransformerEncoder(key_size=count,query_size=count,value_size=count,num_hiddens=count,
#                            norm_shape=[numwidows,count],ffn_num_input=count,ffn_num_hiddens=13340,num_heads=5,num_layers=2,dropout=0.6
#                            )
#
#     return encoder


def create_model2():
    encoder2=TransformerEncoder2(key_size=count,query_size=count,value_size=count,num_hiddens=count,num_heads=5,num_layers=2,dropout=0.6
                           )

    return encoder2
#
# encoder=create_model()
# print(encoder)

encoder2=create_model2()
print(encoder2)
#
x=torch.rand((1,170,116))
print(encoder2(x))

# print(encoder(flattened))