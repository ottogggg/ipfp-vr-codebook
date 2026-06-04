"""
@FileName：test.py\n
@Description：\n  
@Author：zlq\n 
@Time：2025/6/16 13:44\n
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
class multi_adaptive_hypergraoh(nn.Module):
    def __init__(self, configs):
        super(multi_adaptive_hypergraoh, self).__init__()
        # 基础参数
        self.seq_len = configs.seq_len  # 时间序列长度
        self.window_size = configs.window_size  # 各层窗口大小列表
        self.inner_size = configs.inner_size  # 内部维度
        self.dim = configs.d_model  # 嵌入维度
        self.hyper_num = configs.hyper_num  # 各层超边数量列表
        self.alpha = 3  # 激活系数
        self.k = configs.k  # 每个节点的top-k连接

        # 动态构建的模块
        self.embedhy = nn.ModuleList()  # 超边嵌入层列表
        self.embednod = nn.ModuleList()  # 节点嵌入层列表
        self.linhy = nn.ModuleList()  # 超边线性变换
        self.linnod = nn.ModuleList()  # 节点线性变换

        for i in range(len(self.hyper_num)):
            # 为每层添加超边嵌入
            self.embedhy.append(nn.Embedding(self.hyper_num[i], self.dim))
            self.linhy.append(nn.Linear(self.dim, self.dim))
            self.linnod.append(nn.Linear(self.dim, self.dim))

            # 为每层添加节点嵌入（首层特殊处理）
            if i == 0:
                self.embednod.append(nn.Embedding(self.seq_len, self.dim))
            else:
                product = math.prod(self.window_size[:i])
                layer_size = math.floor(self.seq_len / product)
                self.embednod.append(nn.Embedding(int(layer_size), self.dim))

        self.dropout = nn.Dropout(p=0.1)  # 随机失活


    def forward(self,x):
        # 1. 计算各层节点数量
        node_num = [self.seq_len]  # 首层节点数=序列长度
        for ws in self.window_size:
            node_num.append(math.floor(node_num[-1] / ws))

        hyperedge_all = []  # 存储所有层的超图

        # 2. 逐层构建超图
        for i in range(len(self.hyper_num)):
            # 获取当前层的超边和节点索引
            hyp_idx = torch.arange(self.hyper_num[i]).to(x.device)
            node_idx = torch.arange(node_num[i]).to(x.device)

            # 3. 获取嵌入表示
            hyper_embed = self.embedhy[i](hyp_idx)  # 超边嵌入 [num_hyperedges, dim]
            node_embed = self.embednod[i](node_idx)  # 节点嵌入 [num_nodes, dim]

            # 4. 计算关联矩阵（公式对应图片中的E1E2^T）
            adj = torch.mm(node_embed, hyper_embed.transpose(1, 0))  # [num_nodes, num_hyperedges]
            adj = F.softmax(F.relu(self.alpha * adj))  # 归一化

            # 5. 构建稀疏连接（top-k选择）
            mask = torch.zeros_like(adj)
            values, indices = adj.topk(min(adj.size(1), self.k), dim=1)
            mask.scatter_(1, indices, values.fill_(1))
            adj = adj * mask

            # 6. 二值化处理
            adj = torch.where(adj > 0.5, torch.tensor(1).to(x.device),
                              torch.tensor(0).to(x.device))

            # 7. 移除空超边
            adj = adj[:, (adj != 0).any(dim=0)]

            # 8. 转换为超图表示
            # 获取每个超边连接的节点列表
            result_list = [
                list(torch.nonzero(adj[:, col]).flatten().tolist())
                for col in range(adj.shape[1])
            ]

            # 构建超图的双列表表示：
            # node_list: 所有被连接的节点ID
            # hyperedge_list: 对应的超边ID
            node_list = torch.cat([
                torch.tensor(sublist) for sublist in result_list if sublist
            ]).tolist()

            count_list = torch.sum(adj, dim=0).tolist()  # 每个超边的节点数
            hyperedge_list = torch.cat([
                torch.full((count,), idx)
                for idx, count in enumerate(count_list, start=0)
            ]).tolist()

            # 9. 堆叠成超图矩阵
            hypergraph = np.vstack((node_list, hyperedge_list))
            hyperedge_all.append(hypergraph)

        return hyperedge_all
class Config:
    def __init__(self):
        self.seq_len = 7       # 初始序列长度
        self.window_size = [4,4]  # 分层降采样窗口
        self.inner_size = 5    # 未使用，保留字段
        self.d_model = 512      # 嵌入维度
        self.hyper_num = [10, 5] # 每层超边数量
        self.k = 3              # 超边连接的稀疏性控制

if __name__ == '__main__':
    configs = Config()
    model = multi_adaptive_hypergraoh(configs)
    x = torch.randn(32, 96, 7)  # [batch_size, seq_len, num_variables]
    print(model(x))