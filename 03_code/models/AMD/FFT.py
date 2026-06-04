import torch
import torch.nn.functional as F


class FrequencyCorrelationMatrix(torch.nn.Module):
    """
    频域相关矩阵计算 + 稀疏化模块
    输入：[batch_size, feature_num, seq_len] 时域序列
    输出：
        - 频域距离矩阵: [batch_size, feature_num, feature_num]
        - 稀疏化通道掩码: [batch_size, feature_num, feature_num]
    """
    def __init__(self, top_k_ratio=0.3, distance_type="cosine"):
        """
        Args:
            top_k_ratio: 稀疏化保留的Top-K比例（0~1），如0.3表示保留每个特征前30%的关联
            distance_type: 距离计算类型，支持 "cosine"（余弦距离）、"euclidean"（欧式距离）
        """
        super().__init__()
        self.top_k_ratio = top_k_ratio
        self.distance_type = distance_type

    def forward(self, x):
        """
        前向计算：时域→频域幅值→距离矩阵→概率矩阵→稀疏掩码
        Args:
            x: torch.Tensor, 形状 [batch_size, feature_num, seq_len]
        Returns:
            dist_matrix: 频域距离矩阵
            channel_mask: 稀疏化通道掩码
        """
        # 步骤1：时域序列 → 频域幅值（对最后一维做FFT，使用rfft优化实数输入）
        x_fft = torch.fft.rfft(x, dim=-1)  # [batch, feat, seq//2+1] 实数FFT优化
        x_amp = torch.abs(x_fft)

        # 步骤2：计算频域特征的距离矩阵（批量优化，无循环）
        dist_matrix = self._compute_batch_distance(x_amp)

        # 步骤3：距离矩阵 → 概率矩阵（Softmax归一化）
        prob_matrix = F.softmax(-dist_matrix, dim=-1)  # [batch, feat, feat]

        # 步骤4：概率矩阵 → 稀疏化通道掩码（Top-K保留）
        channel_mask = self._batch_sparsify(prob_matrix)

        return dist_matrix, channel_mask

    def _compute_batch_distance(self, x_amp):
        """
        批量计算特征间的距离矩阵（优化版，无for循环）
        Args:
            x_amp: 频域幅值，形状 [batch_size, feature_num, freq_num]
        Returns:
            距离矩阵，形状 [batch_size, feature_num, feature_num]
        """
        if self.distance_type == "cosine":
            # 余弦距离优化：直接归一化后矩阵乘法
            x_norm = F.normalize(x_amp, p=2, dim=-1)
            similarity = torch.bmm(x_norm, x_norm.transpose(1, 2))
            dist_matrix = 1 - similarity

        elif self.distance_type == "euclidean":
            # 欧式距离优化：使用平方展开避免广播
            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
            x_sq = (x_amp ** 2).sum(dim=-1, keepdim=True)  # [batch, feat, 1]
            dist_matrix = x_sq + x_sq.transpose(1, 2) - 2 * torch.bmm(x_amp, x_amp.transpose(1, 2))
            dist_matrix = torch.sqrt(torch.clamp(dist_matrix, min=1e-8))  # 数值稳定性

        else:
            raise ValueError(f"不支持的距离类型：{self.distance_type}，仅支持 cosine/euclidean")

        return dist_matrix

    def _batch_sparsify(self, prob_matrix):
        """
        批量稀疏化：根据Top-K比例生成二进制掩码（优化版）
        Args:
            prob_matrix: 概率矩阵，形状 [batch_size, feature_num, feature_num]
        Returns:
            通道掩码，形状 [batch_size, feature_num, feature_num]（1表示保留，0表示稀疏）
        """
        feat_num = prob_matrix.size(1)
        k = max(1, int(feat_num * self.top_k_ratio))

        # 批量取Top-K的索引
        _, top_indices = torch.topk(prob_matrix, k=k, dim=-1)

        # 优化：使用scatter_直接生成掩码
        mask = torch.zeros_like(prob_matrix)
        mask.scatter_(dim=-1, index=top_indices, value=1.0)

        return mask


# -------------------------- 测试代码 --------------------------
if __name__ == "__main__":
    # 1. 构造测试输入：batch=2, 特征数=8, 序列长度=16
    batch_size, feature_num, seq_len = 2, 8, 16
    x = torch.randn(batch_size, feature_num, seq_len)  # 模拟时域序列

    # 2. 初始化模块（保留Top-30%关联，使用余弦距离）
    freq_corr = FrequencyCorrelationMatrix(top_k_ratio=0.3, distance_type="cosine")

    # 3. 前向计算
    dist_matrix, channel_mask = freq_corr(x)

    # 4. 输出结果信息
    print("=" * 50)
    print(f"输入形状: {x.shape}")
    print(f"频域距离矩阵形状: {dist_matrix.shape}")

    print(f"稀疏化通道掩码形状: {channel_mask.shape}")
    print("=" * 50)
    print("掩码示例（第一个样本）:\n", channel_mask[0].int())
    print("=" * 50)
    # 验证掩码的稀疏性：每个特征保留的关联数
    mask_sum = channel_mask[0].sum(dim=-1)
    print(f"第一个样本每个特征保留的关联数: {mask_sum.tolist()}")
