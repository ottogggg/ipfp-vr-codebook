import torch
import torch.nn as nn


import torch.nn.functional as F

from models.AMD.FFT import FrequencyCorrelationMatrix
from models.AMD.common import RevIN, MDM, SparseTSFModel
from models.AMD.tsmoe import AMS


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.model = AMD(configs)

    def forward(self, x):
        return self.model(x)


class AMD(nn.Module):
    """Implementation of AMD."""

    def __init__(self, configs):
        super(AMD, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        input_shape = (self.seq_len, self.enc_in)
        
        self.target_slice = None
        self.norm = getattr(configs, 'norm', True)
        
        n_block = getattr(configs, 'n_block', 2)
        dropout = getattr(configs, 'dropout', 0.1)
        patch = getattr(configs, 'patch', 24)
        k = getattr(configs, 'mdm_k', 3)
        c = getattr(configs, 'mdm_c', 2)
        layernorm = getattr(configs, 'layernorm', True)

        if self.norm:
            self.rev_norm = RevIN(self.enc_in)

        self.pastmixing = MDM(input_shape, k=k, c=c, layernorm=layernorm)

        self.fc_blocks = nn.ModuleList([SparseTSFModel(input_shape, pred_len=input_shape[0], patch=patch)
                                        for _ in range(n_block)])

        ff_dim = getattr(configs, 'ff_dim', 512)
        num_experts = getattr(configs, 'num_experts', 4)
        top_k = getattr(configs, 'top_k', 2)
        self.moe = AMS(input_shape, self.pred_len, ff_dim=ff_dim, dropout=dropout, num_experts=num_experts, top_k=top_k)

        self.freq_corr = FrequencyCorrelationMatrix(top_k_ratio=0.2, distance_type="cosine")

    def forward(self, x):
        # [batch_size, seq_len, feature_num]

        # layer norm
        if self.norm:
            x = self.rev_norm(x, 'norm')
        # [batch_size, seq_len, feature_num]

        # [batch_size, seq_len, feature_num]
        x = torch.transpose(x, 1, 2)
        # [batch_size, feature_num, seq_len]

        time_embedding = self.pastmixing(x)

        # 优化：使用预初始化的频域模块
        dist_matrix, channel_mask = self.freq_corr(time_embedding)

        for fc_block in self.fc_blocks:
            # SparseTSFModel expects [batch_size, seq_len, feature_num]
            x_sfts = torch.transpose(x, 1, 2)
            x_sfts = fc_block(x_sfts)
            # Transpose back to [batch_size, feature_num, seq_len]
            x = torch.transpose(x_sfts, 1, 2)

        # MOE
        x, moe_loss = self.moe(x, time_embedding)  # seq_len -> pred_len

        # [batch_size, feature_num, pred_len]
        x = torch.transpose(x, 1, 2)
        # [batch_size, pred_len, feature_num]

        # 优化：简化权重矩阵计算
        weight_matrix = F.softmax(1 / (1 + dist_matrix), dim=-1)  # 合并归一化和softmax

        # 加权融合 + 残差连接
        x = torch.bmm(x, weight_matrix) + x

        if self.norm:
            x = self.rev_norm(x, 'denorm', self.target_slice)
        # [batch_size, pred_len, feature_num]

        if self.target_slice:
            x = x[:, :, self.target_slice]

        return x, moe_loss