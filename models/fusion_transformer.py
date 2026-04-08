# models/fusion_transformer.py
# Cross-Modal Attention Transformer：视差特征 × BEV特征 双向融合

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import cfg


# ─────────────────────────────────────────────────────────
# 1. 视差特征 → 融合尺寸 的投影模块
# ─────────────────────────────────────────────────────────
class StereoProjector(nn.Module):
    """
    将视差网络输出的特征 [B, 32, H/4, W/4]
    投影到融合空间 [B, 256, FUSE_BEV_H, FUSE_BEV_W]
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(cfg.DISP_FEAT_CH, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, cfg.FEAT_CH, 3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.FEAT_CH),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat):
        feat = self.proj(feat)
        feat = F.interpolate(
            feat,
            size=(cfg.FUSE_BEV_H, cfg.FUSE_BEV_W),
            mode="bilinear", align_corners=False
        )
        return feat   # [B, 256, FUSE_BEV_H, FUSE_BEV_W]


# ─────────────────────────────────────────────────────────
# 2. 单层 Cross-Modal Attention
# ─────────────────────────────────────────────────────────
class CrossModalAttentionLayer(nn.Module):
    """
    双向交叉注意力：
      - 视差特征作为Query，BEV特征作为Key/Value → 视差特征吸收激光雷达信息
      - BEV特征作为Query，视差特征作为Key/Value → BEV特征吸收视觉信息
    两路结果相加后经FFN输出
    """
    def __init__(self, d_model=cfg.FEAT_CH, nhead=cfg.TF_HEADS,
                 dim_ff=cfg.TF_DIM_FF, dropout=cfg.TF_DROPOUT):
        super().__init__()
        # 视差→BEV 方向
        self.attn_s2b = nn.MultiheadAttention(d_model, nhead, batch_first=True,
                                              dropout=dropout)
        # BEV→视差 方向
        self.attn_b2s = nn.MultiheadAttention(d_model, nhead, batch_first=True,
                                              dropout=dropout)

        # 各自的FFN
        self.ffn_s = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.ffn_b = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )

        self.norm_s1 = nn.LayerNorm(d_model)
        self.norm_s2 = nn.LayerNorm(d_model)
        self.norm_b1 = nn.LayerNorm(d_model)
        self.norm_b2 = nn.LayerNorm(d_model)
        self.drop    = nn.Dropout(dropout)

    def forward(self, feat_s, feat_b):
        """
        feat_s: [B, N, C]  视差特征序列（N = H*W）
        feat_b: [B, N, C]  BEV特征序列
        """
        # ── 视差特征 cross-attend BEV ──
        s2b, _ = self.attn_s2b(query=feat_s, key=feat_b, value=feat_b)
        feat_s = self.norm_s1(feat_s + self.drop(s2b))
        feat_s = self.norm_s2(feat_s + self.drop(self.ffn_s(feat_s)))

        # ── BEV特征 cross-attend 视差 ──
        b2s, _ = self.attn_b2s(query=feat_b, key=feat_s, value=feat_s)
        feat_b = self.norm_b1(feat_b + self.drop(b2s))
        feat_b = self.norm_b2(feat_b + self.drop(self.ffn_b(feat_b)))

        return feat_s, feat_b


# ─────────────────────────────────────────────────────────
# 3. 完整融合 Transformer（3层堆叠）
# ─────────────────────────────────────────────────────────
class FusionTransformer(nn.Module):
    """
    输入:
        disp_feat: [B, 32, H/4, W/4]    视差特征（来自LightStereoNet）
        bev_feat:  [B, 256, BEV_H, BEV_W] BEV特征（来自BEVEncoder）
    输出:
        fused_feat: [B, 256, FUSE_H, FUSE_W]  融合后特征（送入检测头）
    """
    def __init__(self):
        super().__init__()
        # 视差特征投影到统一维度
        self.stereo_proj = StereoProjector()

        # 3层 Cross-Modal Attention
        self.layers = nn.ModuleList([
            CrossModalAttentionLayer() for _ in range(cfg.TF_LAYERS)
        ])

        # 最终融合：拼接后1×1卷积降维
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(cfg.FEAT_CH * 2, cfg.FEAT_CH, 1, bias=False),
            nn.BatchNorm2d(cfg.FEAT_CH),
            nn.ReLU(inplace=True),
        )

    def forward(self, disp_feat, bev_feat):
        B = disp_feat.shape[0]
        H, W = cfg.FUSE_BEV_H, cfg.FUSE_BEV_W

        # ① 投影到统一空间
        feat_s = self.stereo_proj(disp_feat)  # [B, 256, H, W]
        feat_b = bev_feat                      # [B, 256, H, W]

        # ② 展平为序列 [B, H*W, C]
        feat_s = feat_s.flatten(2).permute(0, 2, 1)  # [B, N, 256]
        feat_b = feat_b.flatten(2).permute(0, 2, 1)  # [B, N, 256]

        # ③ 逐层 Cross-Modal Attention
        for layer in self.layers:
            feat_s, feat_b = layer(feat_s, feat_b)

        # ④ 恢复空间维度
        feat_s = feat_s.permute(0, 2, 1).reshape(B, cfg.FEAT_CH, H, W)
        feat_b = feat_b.permute(0, 2, 1).reshape(B, cfg.FEAT_CH, H, W)

        # ⑤ 拼接 + 1×1卷积融合
        fused = self.fuse_conv(torch.cat([feat_s, feat_b], dim=1))
        return fused   # [B, 256, FUSE_H, FUSE_W]