import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
from mamba_ssm import Mamba

def reorder_points_by_hilbert(xyz, features, p=10):
    """
    使用 Hilbert 和 Trans-Hilbert 双曲线对点云重排序，并生成来源指示符：
    Args:
        xyz: Tensor [B, N, 3] 原始坐标
        features: Tensor [B, C, N] 对应特征
        p: 曲线阶数 (2^p 分辨率)
    Returns:
        reordered_xyz: Tensor [B, 2*N, 3] 重排序后坐标
        reordered_features: Tensor [B, C, 2*N] 重排序后特征
        order_indicator: Tensor [2*N] (0 表示 Hilbert, 1 表示 Trans-Hilbert)
    """
    B, N, _ = xyz.shape
    hilbert = HilbertCurve(p, 3)
    reordered_xyz_list = []
    reordered_feat_list = []
    for b in range(B):
        coords = xyz[b].cpu().numpy()
        mins = coords.min(0)
        maxs = coords.max(0)
        norm = ((coords - mins) / (maxs - mins + 1e-6) * (2**p - 1)).astype(int)

        idx_h = np.array(hilbert.distances_from_points(norm.tolist()))
        order_h = np.argsort(idx_h)

        order_t = order_h[::-1].copy()

        order = np.concatenate([order_h, order_t])

        coords_h = coords[order_h]
        coords_t = coords[order_t]
        feats = features[b].cpu()
        feat_h = feats[:, order_h]
        feat_t = feats[:, order_t]
        reordered_xyz_list.append(torch.from_numpy(np.concatenate([coords_h, coords_t], axis=0)).to(xyz.device))
        reordered_feat_list.append(torch.cat([feat_h, feat_t], dim=1).to(features.device))
    reordered_xyz = torch.stack(reordered_xyz_list, dim=0)
    reordered_features = torch.stack(reordered_feat_list, dim=0)

    order_indicator = torch.cat([torch.zeros(N, dtype=torch.long), torch.ones(N, dtype=torch.long)], dim=0)
    return reordered_xyz, reordered_features, order_indicator

class WRBlock(nn.Module):

    def __init__(self, channels, d_state=64, d_conv=4, expand=2, prior_channels=6):
        super().__init__()
        c = channels + prior_channels
        self.prior_channels = prior_channels

        self.ln_main    = nn.LayerNorm(c)

        self.fc_main1   = nn.Linear(c, c)

        self.dwconv_main= nn.Conv1d(c, c, kernel_size=3, padding=1, groups=c)

        self.mamba      = Mamba(d_model=c, d_state=d_state, d_conv=d_conv, expand=expand)

        self.fc_main2   = nn.Linear(c, c)

        self.fc_aux     = nn.Linear(c, c)
        self.proj_main = nn.Conv1d(c, c, kernel_size=1, bias=False)
        self.proj_aux  = nn.Conv1d(c, c, kernel_size=1, bias=False)
        self.bn_out    = nn.BatchNorm1d(c)

    def forward(self, x, xyz, order_h, order_t, wavelet_prior=None):
        B, C, N = x.shape
        if wavelet_prior is None:
            wavelet_prior = x.new_zeros((B, N, self.prior_channels))
        if wavelet_prior.shape != (B, N, self.prior_channels):
            raise ValueError('Invalid wavelet prior shape')
        x = torch.cat([x, wavelet_prior.transpose(1, 2)], dim=1)
        C = x.shape[1]

        coords_h = torch.gather(
            xyz, 1, order_h.unsqueeze(-1).expand(-1, -1, 3)
        )
        coords_t = torch.gather(
            xyz, 1, order_t.unsqueeze(-1).expand(-1, -1, 3)
        )
        xyz_seq  = torch.cat([coords_h, coords_t], dim=1)

        feat_h = torch.gather(
            x, 2, order_h.unsqueeze(1).expand(-1, C, -1)
        )
        feat_t = torch.gather(
            x, 2, order_t.unsqueeze(1).expand(-1, C, -1)
        )
        x_seq  = torch.cat([feat_h, feat_t], dim=2)

        skip = x_seq

        seq = x_seq.permute(0, 2, 1)
        seq = self.ln_main(seq)
        aux = seq.clone()

        seq = F.silu(self.fc_main1(seq))
        seq = seq.permute(0, 2, 1)
        seq = F.silu(self.dwconv_main(seq))
        seq = seq.permute(0, 2, 1)

        ssm_fwd = self.mamba(seq)

        seq_rev = torch.flip(seq, dims=[1])
        ssm_rev = self.mamba(seq_rev)
        ssm_rev = torch.flip(ssm_rev, dims=[1])

        seq_ssm = 0.5 * (ssm_fwd + ssm_rev)

        seq2     = F.silu(self.fc_main2(seq_ssm))
        main_out = seq2.permute(0, 2, 1)

        aux2    = torch.sigmoid(self.fc_aux(aux))
        aux_out = aux2 * seq_ssm

        aux_out = aux_out.permute(0, 2, 1)

        skip_f = skip
        main_f = self.proj_main(main_out)
        aux_f  = self.proj_aux(aux_out)

        fused = skip_f + main_f + aux_f
        fused = self.bn_out(fused)
        fused = F.silu(fused)

        return fused.permute(0, 2, 1)
