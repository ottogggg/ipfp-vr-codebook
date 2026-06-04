import torch
import torch.nn as nn
import warnings
import random
from layers.Embed import NRI_Embedding
import numpy as np
from utils.tools import *

# warnings.filterwarnings("ignore")
# fix_seed = 2023
# random.seed(fix_seed)
# torch.manual_seed(fix_seed)
# np.random.seed(fix_seed)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim, device='cuda:0')
        # self.dropout = nn.Dropout(0.1)
        # self.gelu = nn.GELU()

    def forward(self, inputs):
        x = self.fc1(inputs)
        # x = self.gelu(x)
        # x = self.dropout(x)
        return x

class Encoder(nn.Module):
    def __init__(self, num_nodes, node_features, egde_dim):
        super(Encoder, self).__init__()
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.egde_dim = egde_dim
        self.mlp_edge = MLP(input_dim=2 * self.node_features, output_dim=self.egde_dim)
        self.mlp_node = MLP(input_dim=self.egde_dim, output_dim=self.node_features)
        self.out = MLP(input_dim=self.egde_dim, output_dim=1)

        # 生成所有节点对的索引
        self.node_indices = torch.arange(self.num_nodes, device='cuda:0')
        self.idx_i, self.idx_j = torch.meshgrid(self.node_indices, self.node_indices, indexing='ij')
        self.mask = self.idx_i < self.idx_j

    def node_edge(self, x):
        batch_size = x.shape[0]  # 128

        # 提取节点对的特征
        features_i = x[:, self.idx_i[self.mask], :]
        features_j = x[:, self.idx_j[self.mask], :]
        features_ij = torch.cat([features_i, features_j], dim=-1)

        # 通过MLP得到边的表示
        edge_representations = self.mlp_edge(features_ij)

        # 创建一个全零的边表示张量
        full_edge_representations = torch.zeros(batch_size, self.num_nodes, self.num_nodes, self.egde_dim, device='cuda:0')

        # 用计算得到的边表示填充全零张量
        full_edge_representations[:, self.idx_i[self.mask], self.idx_j[self.mask], :] = edge_representations
        full_edge_representations[:, self.idx_j[self.mask], self.idx_i[self.mask], :] = edge_representations  # 无向边
        return full_edge_representations

    def edge_node(self, x):
        # 聚合所有节点的边表示
        node_edge_representation = torch.sum(x, dim=2)
        node_representations = self.mlp_node(node_edge_representation)

        return node_representations

    def forward(self, data):
        # Step 1: 计算两两节点间边的表示
        x = self.node_edge(data)
        # Step 2: 根据边的表示计算节点的表示
        x = self.edge_node(x)
        # # # step 3 根据节点表示获得边的表示
        x = self.node_edge(x)
        # 输出层
        x = self.out(x)
        return x


class NRI_Inference(nn.Module):
    def __init__(self, num_nodes, node_features, egde_dim,num_samples):
        super(NRI_Inference, self).__init__()
        self.encoder = Encoder(num_nodes, node_features, egde_dim)
        self.num_samples = num_samples

    def forward(self, data):
        # 编码器获取特征之间概率
        edge_representations = self.encoder(data)
        edge_representations = edge_representations.squeeze(-1)

        batch_size, num_nodes, _ = edge_representations.shape
        diag_indices = torch.arange(num_nodes, device=edge_representations.device)
        edge_representations[:, diag_indices, diag_indices] = float('-inf')

        adj = torch.softmax(edge_representations,dim=-1)
        # adj = torch.nn.functional.gumbel_softmax(edge_representations)

        # 离散采样
        adj = self.inverse_transform_sampling(adj,self.num_samples)

        return adj

    def inverse_transform_sampling(self, tensor, num_samples):
        # 归一化概率分布
        probs = tensor / tensor.sum(dim=-1, keepdim=True)

        # 计算累积分布函数
        cdf = torch.cumsum(probs, dim=-1)

        # 生成均匀随机数并扩展维度
        u = torch.rand(*probs.shape[:-1], num_samples, device='cuda:0')

        # 逆变换获取采样索引
        indices = torch.searchsorted(cdf, u)  # (B, S, K)

        # 构造三维坐标索引
        batch_coords = torch.arange(probs.size(0), device='cuda:0')[:, None, None]
        seq_coords = torch.arange(probs.size(1), device='cuda:0')[None, :, None]
        indices_3d = torch.stack([batch_coords.expand_as(indices),
                                  seq_coords.expand_as(indices),
                                  indices], dim=-1)

        # 去重并向量化赋值
        unique_coords = torch.unique(indices_3d.view(-1, 3), dim=0).unbind(-1)
        new_matrix = torch.zeros_like(probs)
        new_matrix[unique_coords] = probs[unique_coords]

        new_matrix = masked_softmax(new_matrix)
        return new_matrix



class NRI(nn.Module):
    def __init__(self, num_nodes,seq_len, node_features, egde_dim, patch_len,num_samples):
        super(NRI, self).__init__()
        self.patch_len = patch_len
        self.model = NRI_Inference(num_nodes, node_features, egde_dim,num_samples)

    def forward(self, data):

        # b,v,l = data.shape
        # x = data.reshape(-1, b,v,self.patch_len)
        #
        # patch_num, batch, v, l = x.shape
        # corr_list = []
        # with torch.no_grad():
        #     for i in range(patch_num):
        #         input = x[i, :, :, :]
        #         adj = self.model(input)
        #         corr_list.append(adj)
        # corr_list = torch.stack(corr_list, dim=0)
        # corr_list = corr_list.permute(1,0,2,3)
        # return corr_list

        b,v,l = data.shape
        x = data.reshape(-1, b,v,self.patch_len)
        patch_num, batch, v, l = x.shape
        corr_list = []
        with torch.no_grad():
            for i in range(patch_num):
                input = x[i, :, :, :]
                x_centered = input - input.mean(dim=2, keepdim=True)  # 形状保持 [256,7,24]
                # 计算协方差矩阵
                cov_matrix = torch.bmm(x_centered, x_centered.transpose(1, 2))  # 批矩阵乘法 [256,7,7]
                cov_matrix = cov_matrix / (input.shape[2] - 1)  # 无偏估计，分母为 n-1
                # 计算标准差
                std_dev = x_centered.std(dim=2, unbiased=True, keepdim=False)  # 形状 [256,7]
                # 计算标准差外积矩阵
                std_outer = torch.bmm(std_dev.unsqueeze(2), std_dev.unsqueeze(1))  # [256,7,7]
                # 计算 Pearson 相关系数矩阵，避免分母为零
                corr_matrix = cov_matrix / (std_outer + 1e-8)
                corr_matrix = torch.softmax(corr_matrix, dim=-1)
                # 添加极小值保证数值稳定性
                corr_list.append(corr_matrix)
        corr_list = torch.stack(corr_list, dim=0)
        corr_list = corr_list.permute(1,0,2,3)
        return corr_list