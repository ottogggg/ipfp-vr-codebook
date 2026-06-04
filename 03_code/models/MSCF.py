import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDecompose(nn.Module):
    def __init__(self, num_scales=4, pool_stride=2):
        super().__init__()
        self.num_scales = num_scales
        self.pool_stride = pool_stride
        self.pool = nn.AvgPool1d(kernel_size=pool_stride, stride=pool_stride, ceil_mode=False)

    def forward(self, x):
        # x: [B, C, L]
        scales = [x]
        cur = x
        for _ in range(1, self.num_scales):
            if cur.size(-1) < self.pool_stride:
                scales.append(cur)
                continue
            cur = self.pool(cur)
            scales.append(cur)
        return scales


class SingleScaleSparsePredictor(nn.Module):
    def __init__(self, scale_seq_len, pred_len, enc_in, period_len, d_model, model_type):
        super().__init__()
        self.seq_len = int(scale_seq_len)
        self.pred_len = int(pred_len)
        self.enc_in = int(enc_in)
        self.period_len = max(1, int(period_len))
        self.d_model = int(d_model)
        self.model_type = model_type

        self.seg_num_x = max(1, self.seq_len // self.period_len)
        self.seg_num_y = max(1, self.pred_len // self.period_len)
        self.eff_in_len = self.seg_num_x * self.period_len
        self.eff_out_len = self.seg_num_y * self.period_len

        kernel_size = 1 + 2 * (self.period_len // 2)
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=self.period_len // 2,
            padding_mode="zeros",
            bias=False,
        )

        if self.model_type == "linear":
            self.mapper = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
        else:
            self.mapper = nn.Sequential(
                nn.Linear(self.seg_num_x, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.seg_num_y),
            )

    def forward(self, x):
        # x: [B, C, Ls]
        bsz, channels, cur_len = x.shape
        if cur_len < self.eff_in_len:
            x = F.pad(x, (0, self.eff_in_len - cur_len))
        elif cur_len > self.eff_in_len:
            x = x[:, :, : self.eff_in_len]

        x = self.conv1d(x.reshape(-1, 1, self.eff_in_len)).reshape(bsz, channels, self.eff_in_len) + x
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        y = self.mapper(x)
        y = y.permute(0, 2, 1).reshape(bsz, channels, self.eff_out_len)

        if self.eff_out_len != self.pred_len:
            y = F.interpolate(y, size=self.pred_len, mode="linear", align_corners=False)
        return y


class AdaptiveChannelSoftMask(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, x):
        # x: [B, C, L]
        x_freq = torch.fft.rfft(x, dim=-1)
        feat = torch.cat([x_freq.real, x_freq.imag], dim=-1)  # [B, C, F*2]
        feat = F.normalize(feat, p=2, dim=-1)
        sim = torch.matmul(feat, feat.transpose(1, 2)) / max(self.temperature, 1e-6)
        mask = torch.softmax(sim, dim=-1)
        return mask


class IntraScaleFusion(nn.Module):
    def __init__(self, pred_len, dropout, attn_dim):
        super().__init__()
        self.attn_dim = attn_dim
        self.w_q = nn.Linear(pred_len, attn_dim, bias=False)
        self.w_k = nn.Linear(pred_len, attn_dim, bias=False)
        self.w_v = nn.Linear(pred_len, pred_len, bias=False)
        self.norm = nn.LayerNorm(pred_len)
        self.ff = nn.Sequential(
            nn.Linear(pred_len, pred_len, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, y, mask):
        # y: [B, C, P], mask: [B, C, C]
        q = self.w_q(y)  # [B, C, D]
        k = self.w_k(y)  # [B, C, D]
        v = self.w_v(y)  # [B, C, P]

        logits = torch.matmul(q, k.transpose(1, 2)) / (self.attn_dim ** 0.5)  # [B, C, C]
        # Non-core channels are suppressed before softmax, following Eq.(4.11).
        attn_input = logits * mask + (1.0 - mask) * (-1e9)
        score = torch.softmax(attn_input, dim=-1)

        rel_y = torch.matmul(score, v)  # [B, C, P]
        out = self.norm(y + rel_y)
        out = y + self.ff(out)
        return out


class AdaptiveScaleFusion(nn.Module):
    def __init__(self, num_scales, pred_len):
        super().__init__()
        self.num_scales = num_scales
        self.scale_router = nn.Linear(pred_len, num_scales, bias=True)
        self.noise_proj = nn.Linear(pred_len, num_scales, bias=True)
        self.softplus = nn.Softplus()

    def forward(self, scale_preds):
        # scale_preds: list of [B, C, P]
        x = torch.stack(scale_preds, dim=1)  # [B, S, C, P]
        # Build per-variable, per-scale weights from temporal summary.
        feat = x.mean(dim=1)  # [B, C, P]

        clean_logits = self.scale_router(feat)  # [B, C, S]
        noise_scale = self.softplus(self.noise_proj(feat))
        noise = torch.randn_like(noise_scale)
        noisy_logits = clean_logits + noise * noise_scale
        weights = torch.softmax(noisy_logits, dim=-1)  # [B, C, S]

        fused = torch.einsum("bscp,bcs->bcp", x, weights)
        return fused


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.period_len = max(1, configs.period_len)
        self.model_type = configs.model_type
        self.dropout = configs.dropout

        self.ms_num_scales = int(getattr(configs, "ms_num_scales", 4))
        self.ms_pool_stride = int(getattr(configs, "ms_pool_stride", 2))
        assert self.model_type in ["linear", "mlp"]

        self.ms_decompose = MultiScaleDecompose(
            num_scales=self.ms_num_scales, pool_stride=self.ms_pool_stride
        )

        self.scale_predictors = nn.ModuleList()
        self.scale_masks = nn.ModuleList()
        self.scale_fusions = nn.ModuleList()
        attn_dim = max(8, self.d_model // 4)
        for i in range(self.ms_num_scales):
            scale_len = max(2, self.seq_len // (self.ms_pool_stride ** i))
            predictor = SingleScaleSparsePredictor(
                scale_seq_len=scale_len,
                pred_len=self.pred_len,
                enc_in=self.enc_in,
                period_len=max(1, min(self.period_len, scale_len)),
                d_model=self.d_model,
                model_type=self.model_type,
            )
            self.scale_predictors.append(predictor)
            self.scale_masks.append(AdaptiveChannelSoftMask())
            self.scale_fusions.append(IntraScaleFusion(self.pred_len, self.dropout, attn_dim))

        self.scale_fusion = AdaptiveScaleFusion(self.ms_num_scales, self.pred_len)

    def forward(self, x):
        # x: [B, L, C]
        seq_mean = torch.mean(x, dim=1, keepdim=True)
        x = (x - seq_mean).permute(0, 2, 1)  # [B, C, L]

        multi_scale_x = self.ms_decompose(x)
        scale_preds = []
        for i in range(self.ms_num_scales):
            x_s = multi_scale_x[i]
            y_s = self.scale_predictors[i](x_s)
            mask_s = self.scale_masks[i](x_s)
            y_s = self.scale_fusions[i](y_s, mask_s)
            scale_preds.append(y_s)

        y = self.scale_fusion(scale_preds)

        y = y.permute(0, 2, 1) + seq_mean
        return y
