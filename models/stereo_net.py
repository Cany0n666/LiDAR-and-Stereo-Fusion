# models/stereo_net.py
# 轻量双目视差网络：共享编码器 + 相关体 + Soft-Argmin 可微视差回归

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import cfg


# ─────────────────────────────────────────────────────────
# 1. 共享特征编码器（左右目共用同一套权重）
# ─────────────────────────────────────────────────────────
class FeatureEncoder(nn.Module):
    """
    输入:  [B, 3, H, W]  RGB 图像
    输出:  [B, 32, H/4, W/4]  特征图（下采样4倍）
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1：stride=2，下采样 ×2
            nn.Conv2d(3,  16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Block 2：stride=2，下采样 ×2（总计 ×4）
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)   # [B, 32, H/4, W/4]


# ─────────────────────────────────────────────────────────
# 2. 相关体构建（Cost Volume）
# ─────────────────────────────────────────────────────────
def build_correlation_volume(feat_l, feat_r, max_disp):
    """
    在特征空间构建视差相关体。
    对右目特征图沿水平方向逐步平移 0~max_disp 个像素，
    与左目特征图做内积，得到每个视差假设下的匹配代价。

    feat_l, feat_r: [B, C, H, W]
    max_disp      : 最大视差（在特征图尺度，= cfg.MAX_DISP // 4）
    返回: cost_volume [B, max_disp, H, W]
    """
    B, C, H, W = feat_l.shape
    cost_volume = torch.zeros(B, max_disp, H, W,
                              device=feat_l.device, dtype=feat_l.dtype)
    for d in range(max_disp):
        if d == 0:
            cost_volume[:, d, :, :] = (feat_l * feat_r).mean(dim=1)
        else:
            # 右目特征向右平移 d 个像素（左目看到的点在右目中偏左）
            cost_volume[:, d, :, d:] = (
                feat_l[:, :, :, d:] * feat_r[:, :, :, :-d]
            ).mean(dim=1)
    return cost_volume   # [B, max_disp, H, W]


# ─────────────────────────────────────────────────────────
# 3. 代价聚合网络（3D 卷积平滑代价体）
# ─────────────────────────────────────────────────────────
class CostAggregation(nn.Module):
    """
    输入: cost_volume [B, max_disp, H, W]
    将其视为 [B, 1, D, H, W] 做 3D 卷积，聚合视差维度上的上下文信息
    输出: [B, max_disp, H, W]
    """
    def __init__(self, max_disp):
        super().__init__()
        self.agg = nn.Sequential(
            nn.Conv3d(1, 8,  3, padding=1, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, 3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, 3, padding=1, bias=False),
        )

    def forward(self, cost):
        x = cost.unsqueeze(1)          # [B, 1, D, H, W]
        x = self.agg(x).squeeze(1)    # [B, D, H, W]
        return x


# ─────────────────────────────────────────────────────────
# 4. Soft-Argmin 可微视差回归
# ─────────────────────────────────────────────────────────
def soft_argmin(cost_volume):
    """
    对代价体在视差维度做 Softmax，再对视差索引加权求和，
    得到连续可微的视差预测值。

    cost_volume: [B, D, H, W]
    返回: disp   [B, 1, H, W]

    可微性说明：
      - Softmax 处处可导
      - 加权求和是线性操作
      - 整个过程对输入 cost_volume 的梯度可以反向传播
      - 相比 argmax（不可微），Soft-Argmin 允许端到端训练
    """
    D = cost_volume.shape[1]
    # 视差索引 [0, 1, ..., D-1]，形状 [1, D, 1, 1]
    disp_idx = torch.arange(D, device=cost_volume.device,
                             dtype=cost_volume.dtype)
    disp_idx = disp_idx.view(1, D, 1, 1)

    # Softmax：将代价转为概率分布（代价越小匹配越好，取负号）
    prob = F.softmax(-cost_volume, dim=1)   # [B, D, H, W]

    # 加权求和得到亚像素级视差
    disp = (prob * disp_idx).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    return disp


# ─────────────────────────────────────────────────────────
# 5. 视差上采样模块（将特征尺度视差恢复到原图尺度）
# ─────────────────────────────────────────────────────────
class DispRefine(nn.Module):
    """
    将 H/4 × W/4 的视差图上采样回 H × W，
    并用轻量卷积做边缘细化。
    """
    def __init__(self):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),   # 视差值非负
        )

    def forward(self, disp_low, target_h, target_w):
        # 双线性上采样
        disp_up = F.interpolate(
            disp_low,
            size=(target_h, target_w),
            mode='bilinear', align_corners=False
        )
        # 乘以下采样倍数（4）还原到像素单位
        disp_up = disp_up * 4.0
        # 轻量细化
        disp_up = self.refine(disp_up)
        return disp_up   # [B, 1, H, W]


# ─────────────────────────────────────────────────────────
# 6. 完整视差网络
# ─────────────────────────────────────────────────────────
class LightStereoNet(nn.Module):
    """
    输入:
        img_l: [B, 3, H, W]  左目图像
        img_r: [B, 3, H, W]  右目图像
    输出:
        disp:      [B, 1, H, W]    全分辨率视差图
        disp_feat: [B, 32, H/4, W/4]  中间特征（送入 FusionTransformer）
    """
    def __init__(self):
        super().__init__()
        self.max_disp_feat = cfg.MAX_DISP // 4   # 特征图尺度最大视差 = 48

        self.encoder  = FeatureEncoder()
        self.cost_agg = CostAggregation(self.max_disp_feat)
        self.refine   = DispRefine()

    def forward(self, img_l, img_r):
        H, W = img_l.shape[2], img_l.shape[3]

        # ① 共享编码器提取特征
        feat_l = self.encoder(img_l)   # [B, 32, H/4, W/4]
        feat_r = self.encoder(img_r)   # [B, 32, H/4, W/4]

        # ② 构建相关体
        cost = build_correlation_volume(
            feat_l, feat_r, self.max_disp_feat)   # [B, 48, H/4, W/4]

        # ③ 代价聚合
        cost = self.cost_agg(cost)                 # [B, 48, H/4, W/4]

        # ④ Soft-Argmin 回归视差（特征尺度）
        disp_low = soft_argmin(cost)               # [B, 1, H/4, W/4]

        # ⑤ 上采样 + 细化到原图尺度
        disp = self.refine(disp_low, H, W)         # [B, 1, H, W]

        # feat_l 作为视觉特征送入融合模块
        return disp, feat_l