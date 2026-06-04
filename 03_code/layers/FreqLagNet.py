import math
from math import sqrt
import torch
import torch.nn as nn

from layers.Embed import DataEmbedding_wo_pos
from utils.tools import masked_softmax


class Dila_conv(nn.Module):
    def __init__(self, seq_len, pred_len, period_len,channel,d_model,dropout):
        super(Dila_conv,self).__init__()
        self.period_len = period_len
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = channel
        self.d_model = d_model
        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                                stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)

        self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)

        self.mlp2 = nn.Sequential(
            nn.Linear(self.pred_len, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )


    def forward(self,x):
        batch_size = x.shape[0]
        # 1D convolution aggregation
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x
        # downsampling: b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        # sparse forecasting
        y = self.linear(x)  # bc,w,m
        # upsampling: bc,w,m -> bc,m,w -> b,c,s
        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)
        y = self.mlp2(y)
        return y


class SCALayer(nn.Module):
    def __init__(self, input_dim, d_model, pred_len, patch_num ,dropout, enc_in,period_len, bias=True):
        super(SCALayer, self).__init__()
        self.channel = enc_in
        self.norm1 = nn.LayerNorm(pred_len)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.patch_len = period_len
        self.pred_len = pred_len

        # Feed-Forward
        self.ff = nn.Sequential(nn.Linear(pred_len, pred_len, bias=bias),
                                nn.Dropout(dropout)
                                )

        self.projection = nn.Linear(input_dim, pred_len)

        self.Relationship_learning = nn.Sequential(
            nn.Linear(patch_num, self.pred_len // self.patch_len, bias=True),
            nn.GELU(),
            nn.Dropout(dropout)
        )


    def forward(self, x_enc, x_tra,History_A):
        # x_tra [256,7,pre_len]
        B,C,D = x_tra.shape

        # 1. 分块处理
        x_enc = x_tra.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)

        batchsize, patch_num,channel,_ =History_A.shape

        # # 变量间的关系之间的学习 Future Relationship Inference
        History_A = History_A.view(batchsize, patch_num,channel * channel).permute(0, 2, 1)

        Future_A = self.Relationship_learning(History_A)

        Future_A = Future_A.permute(0,2,1).view(batchsize, self.pred_len//self.patch_len, channel, channel)

        GLOBAL_MATRIX = torch.mean(Future_A, 1)

        Future_A = masked_softmax(Future_A)

        GLOBAL_MATRIX = masked_softmax(GLOBAL_MATRIX)

        # Multivariate Synergistic Prerdiction
        x_enc = torch.matmul(Future_A, x_enc.permute(0,2,1,3))

        x_enc = x_enc.reshape(B,C,-1)+torch.matmul(GLOBAL_MATRIX,x_tra)

        # # 残差连接
        x = x_tra + x_enc

        # 5. FFN处理（带归一化和残差）
        x = self.norm1(x)  # FFN前归一化
        x = x_tra + self.ff(x)

        return x

# 直接使用历史关系矩阵进行预测
# class SCALayer(nn.Module):
#     def __init__(self, input_dim, d_model, pred_len, patch_num ,dropout, enc_in,period_len, bias=True):
#         super(SCALayer, self).__init__()
#         self.channel = enc_in
#         self.norm1 = nn.LayerNorm(pred_len)
#         self.dropout = nn.Dropout(dropout)
#         self.d_model = d_model
#         self.patch_len = period_len
#
#         # Feed-Forward
#         self.ff = nn.Sequential(nn.Linear(pred_len, pred_len, bias=bias),
#                                 nn.Dropout(dropout)
#                                 )
#
#         self.projection = nn.Sequential(nn.Linear(input_dim, pred_len),
#                                 nn.GELU(),
#                                 nn.Linear(pred_len, pred_len)
#                                 )
#
#         # 关系学习模块
#         self.Relationship_learning = nn.Sequential(
#             nn.Linear(patch_num, patch_num, bias=True),
#             nn.GELU(),
#             nn.Dropout(dropout)
#         )
#
#
#     def forward(self, x_enc, x_tra,History_A):
#         # x_tra [256,7,pre_len]
#         B,C,D = x_enc.shape
#
#         # 1. 分块处理
#         x_enc = x_enc.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
#
#         batchsize, patch_num,channel,_ =History_A.shape
#
#         # Multivariate Synergistic Prerdiction
#         x_enc = torch.matmul(History_A, x_enc.permute(0,2,1,3))
#
#         x_enc = x_enc.reshape(B,C,-1)
#
#         x_enc = self.projection(x_enc)
#
#         # 残差连接
#         x = x_tra + x_enc
#
#         # 5. FFN处理（带归一化和残差）
#         x = self.norm1(x)  # FFN前归一化
#         x = x_tra + self.ff(x)
#
#         return x

class RecursiveRelationship(nn.Module):
    def __init__(self, patch_num, pred_len, patch_len, channel, dropout):
        super().__init__()
        self.pred_steps = pred_len // patch_len
        self.channel = channel

        # 递归单元设计
        self.cell = nn.LSTMCell(
            input_size=patch_num,
            hidden_size=channel * channel
        )

        # 时间步间参数共享
        self.transform = nn.Sequential(
            nn.Linear(channel * channel, channel * channel),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 初始化隐藏状态
        self.h_init = nn.Parameter(torch.randn(1, channel * channel))
        self.c_init = nn.Parameter(torch.randn(1, channel * channel))

    def forward(self, History_A):
        batch_size, patch_num, _ = History_A.shape

        # 初始化递归状态
        h = self.h_init.repeat(batch_size, 1)
        c = self.c_init.repeat(batch_size, 1)

        future_relations = []
        for _ in range(self.pred_steps):
            # 递归计算（参考网页6[6](@ref)的序列生成思想）
            h, c = self.cell(History_A[:, -1, :], (h, c))  # 取最新关系作为输入
            transformed = self.transform(h)

            # 保留时间维度（参考网页4[4](@ref)的分块处理）
            future_relations.append(transformed.view(
                batch_size, self.channel, self.channel
            ))

            # 更新历史关系（类似网页5[5](@ref)的迭代预测）
            History_A = torch.cat([History_A[:, :, 1:],transformed.unsqueeze(-1)], dim=-1)

        return torch.stack(future_relations, dim=1)
class SCALayer(nn.Module):
    def __init__(self, input_dim, d_model, pred_len, patch_num ,dropout, enc_in,period_len, bias=True):
        super(SCALayer, self).__init__()
        self.channel = enc_in
        self.norm1 = nn.LayerNorm(pred_len)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.patch_len = period_len
        self.pred_len = pred_len

        # Feed-Forward
        self.ff = nn.Sequential(nn.Linear(pred_len, pred_len, bias=bias),
                                nn.Dropout(dropout)
                                )

        self.projection = nn.Linear(input_dim, pred_len)

        self.Relationship_learning = RecursiveRelationship(
            patch_num=patch_num,
            pred_len=pred_len,
            patch_len=period_len,
            channel=enc_in,
            dropout=dropout
        )

    def forward(self, x_enc, x_tra,History_A):
        # x_tra [256,7,pre_len]
        B,C,D = x_tra.shape

        # 1. 分块处理
        x_enc = x_tra.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)

        batchsize, patch_num,channel,_ =History_A.shape

        # # 变量间的关系之间的学习 Future Relationship Inference
        History_A = History_A.view(batchsize, patch_num,channel * channel).permute(0, 2, 1)

        Future_A = self.Relationship_learning(History_A)

        Future_A = masked_softmax(Future_A)

        # Multivariate Synergistic Prerdiction
        x_enc = torch.matmul(Future_A, x_enc.permute(0,2,1,3))

        x_enc = x_enc.reshape(B,C,-1)

        # # 残差连接
        x = x_tra + x_enc

        # 5. FFN处理（带归一化和残差）
        x = self.norm1(x)  # FFN前归一化
        x = x_tra + self.ff(x)

        return x

class SCA(nn.Module):
    def __init__(self, sca_layers, input_dim, d_model, norm_layer=None):
        super(SCA, self).__init__()
        self.sca_layers = nn.ModuleList(sca_layers)
        self.norm = norm_layer

    def forward(self,ex_lag, x,History_A):
        for sca_layers in self.sca_layers:
            x = sca_layers(ex_lag, x,History_A)
        return x

