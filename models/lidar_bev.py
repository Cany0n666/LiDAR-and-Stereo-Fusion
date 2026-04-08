# models/lidar_bev.py
# 点云 → BEV 投影 + CNN 特征编码器

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import cfg


# ─────────────────────────────────────────────────────────
# 1. 点云 → BEV 伪图像（numpy，在 DataLoader worker 中执行）
# ─────────────────────────────────────────────────────────
def points_to_bev(points):
    """
    将激光雷达点云投影为 BEV 伪图像（3通道）。

    points: np.ndarray [N, 4]  (x, y, z, intensity)
            x: 前向，y: 左向，z: 上向（KITTI 坐标系）

    返回: bev  np.ndarray [3, BEV_H, BEV_W]  float32
          通道0：最大高度图（归一化到 [0,1]）
          通道1：平均反射强度图（归一化到 [0,1]）
          通道2：点云密度图（log 归一化）
    """
    x_min, x_max = cfg.PC_X_MIN, cfg.PC_X_MAX
    y_min, y_max = cfg.PC_Y_MIN, cfg.PC_Y_MAX
    z_min, z_max = cfg.PC_Z_MIN, cfg.PC_Z_MAX
    voxel         = cfg.BEV_VOXEL_SIZE
    H, W          = cfg.BEV_H, cfg.BEV_W

    # ── 范围过滤 ──────────────────────────────────────────
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] < z_max)
    )
    pts = points[mask]   # [M, 4]

    # ── 计算体素索引 ──────────────────────────────────────
    # BEV 的行对应 x（前向），列对应 y（横向）
    row = np.floor((pts[:, 0] - x_min) / voxel).astype(np.int32)
    col = np.floor((pts[:, 1] - y_min) / voxel).astype(np.int32)
    row = np.clip(row, 0, H - 1)
    col = np.clip(col, 0, W - 1)

    # ── 初始化三个通道 ────────────────────────────────────
    height_map    = np.full((H, W), z_min, dtype=np.float32)
    intensity_map = np.zeros((H, W), dtype=np.float32)
    density_map   = np.zeros((H, W), dtype=np.float32)

    # ── 填充：遍历每个点 ──────────────────────────────────
    # 使用 np.maximum.at 实现最大高度聚合
    np.maximum.at(height_map,    (row, col), pts[:, 2])
    np.add.at(intensity_map,     (row, col), pts[:, 3])
    np.add.at(density_map,       (row, col), 1.0)

    # ── 强度图：除以点数得到均值 ──────────────────────────
    nonzero = density_map > 0
    intensity_map[nonzero] /= density_map[nonzero]

    # ── 归一化 ────────────────────────────────────────────
    height_map    = (height_map - z_min) / (z_max - z_min + 1e-6)
    height_map    = np.clip(height_map, 0.0, 1.0)
    intensity_map = np.clip(intensity_map, 0.0, 1.0)
    density_map   = np.log1p(density_map) / np.log1p(
        density_map.max() + 1e-6)   # log 压缩

    bev = np.stack([height_map, intensity_map, density_map], axis=0)
    return bev.astype(np.float32)   # [3, H, W]


# ─────────────────────────────────────────────────────────
# 2. BEV CNN 编码器
# ─────────────────────────────────────────────────────────
class BEVEncoder(nn.Module):
    """
    输入:  bev [B, 3, BEV_H, BEV_W]   = [B, 3, 704, 800]
    输出:  feat [B, 256, FUSE_BEV_H, FUSE_BEV_W]  = [B, 256, 88, 100]

    网络结构：
      3 个 stride=2 的下采样块（总计 ×8 下采样）
      + 残差连接保留细节
    """
    def __init__(self):
        super().__init__()

        # ── 下采样骨干 ────────────────────────────────────
        self.block1 = self._make_block(cfg.BEV_C, 64,  stride=2)   # /2
        self.block2 = self._make_block(64,        128, stride=2)   # /4
        self.block3 = self._make_block(128,       256, stride=2)   # /8

        # ── 残差捷径（1×1 卷积对齐通道）────────────────────
        self.skip1 = nn.Sequential(
            nn.Conv2d(cfg.BEV_C, 64,  1, stride=2, bias=False),
            nn.BatchNorm2d(64))
        self.skip2 = nn.Sequential(
            nn.Conv2d(64,  128, 1, stride=2, bias=False),
            nn.BatchNorm2d(128))
        self.skip3 = nn.Sequential(
            nn.Conv2d(128, 256, 1, stride=2, bias=False),
            nn.BatchNorm2d(256))

        # ── 输出尺寸对齐（自适应池化到 FUSE_BEV_H × FUSE_BEV_W）──
        self.out_pool = nn.AdaptiveAvgPool2d(
            (cfg.FUSE_BEV_H, cfg.FUSE_BEV_W))

        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _make_block(in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, bev):
        # Block 1 + 残差
        x = self.relu(self.block1(bev) + self.skip1(bev))    # [B, 64,  H/2, W/2]
        # Block 2 + 残差
        x = self.relu(self.block2(x)   + self.skip2(x))      # [B, 128, H/4, W/4]
        # Block 3 + 残差
        x = self.relu(self.block3(x)   + self.skip3(x))      # [B, 256, H/8, W/8]
        # 自适应池化对齐融合尺寸
        x = self.out_pool(x)                                  # [B, 256, 88, 100]
        return x