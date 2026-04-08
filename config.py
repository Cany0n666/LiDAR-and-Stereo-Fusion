# config.py  ── 全局超参数配置

import os
import numpy as np


class Config:

    # ══════════════════════════════════════════════
    # 路径配置
    # ══════════════════════════════════════════════
    KITTI_ROOT  = "./data/KITTI"          # KITTI 根目录
    CKPT_DIR    = "./checkpoints"         # 模型保存目录
    LOG_DIR     = "./logs"                # 日志目录

    # ══════════════════════════════════════════════
    # 数据配置
    # ══════════════════════════════════════════════
    IMG_H       = 370                     # 输入图像高度
    IMG_W       = 1224                    # 输入图像宽度
    VAL_RATIO   = 0.2                     # 验证集比例（8:2 划分）

    # 图像归一化（ImageNet 均值/标准差）
    IMG_MEAN    = [0.485, 0.456, 0.406]
    IMG_STD     = [0.229, 0.224, 0.225]

    # ══════════════════════════════════════════════
    # 点云 / BEV 配置
    # ══════════════════════════════════════════════
    PC_X_MIN, PC_X_MAX =  0.0,  70.4     # 前向范围 (m)
    PC_Y_MIN, PC_Y_MAX = -40.0, 40.0     # 横向范围 (m)
    PC_Z_MIN, PC_Z_MAX = -3.0,  1.0      # 垂直范围 (m)
    BEV_VOXEL_SIZE     = 0.1             # 体素大小 (m)

    # BEV 特征图尺寸（由范围和体素大小决定）
    BEV_H = int((PC_X_MAX - PC_X_MIN) / BEV_VOXEL_SIZE)   # 704
    BEV_W = int((PC_Y_MAX - PC_Y_MIN) / BEV_VOXEL_SIZE)   # 800
    BEV_C = 3    # BEV 通道数：高度图 + 强度图 + 密度图

    # ══════════════════════════════════════════════
    # 视差网络配置
    # ══════════════════════════════════════════════
    MAX_DISP      = 192                  # 最大视差搜索范围（像素）
    DISP_FEAT_CH  = 32                   # 视差特征通道数

    # ══════════════════════════════════════════════
    # 融合 Transformer 配置
    # ══════════════════════════════════════════════
    FEAT_CH       = 256                  # 统一特征维度
    FUSE_BEV_H    = 44                   # 融合特征图高度
    FUSE_BEV_W    = 50                  # 融合特征图宽度
    TF_LAYERS     = 3                    # Cross-Modal Attention 层数
    TF_HEADS      = 8                    # 多头注意力头数
    TF_DIM_FF     = 1024                 # FFN 隐层维度
    TF_DROPOUT    = 0.1                  # Dropout 比例

    # ══════════════════════════════════════════════
    # 检测头 / Anchor 配置
    # ══════════════════════════════════════════════
    NUM_CLASSES   = 3                    # Car / Pedestrian / Cyclist

    # 每类 Anchor 的 [长, 宽, 高]（单位：米，KITTI 统计均值）
    # 每类 Anchor 的 [w, l, h]（宽, 长, 高）
    ANCHORS = {
        "Car"        : [1.6,  3.9,  1.56],
        "Pedestrian" : [0.6,  0.8,  1.73],
        "Cyclist"    : [0.6,  1.76, 1.73],
    }
    # Anchor z中心（地面以上，由KITTI统计得到）
    ANCHOR_Z_CENTERS = {
        "Car"        : -0.9,
        "Pedestrian" : -0.6,
        "Cyclist"    : -0.6,
    }
    ANCHOR_ROTATIONS      = [0, np.pi / 2]   # 两个旋转方向
    NUM_ANCHORS_PER_LOC   = len(ANCHORS) * len(ANCHOR_ROTATIONS)  # 6

    # Anchor 匹配阈值
    POS_IOU_THRESH = 0.35
    NEG_IOU_THRESH = 0.25

    # ══════════════════════════════════════════════
    # 损失函数配置
    # ══════════════════════════════════════════════
    FOCAL_ALPHA   = 0.75
    FOCAL_GAMMA   = 2.0
    LOSS_CLS_W    = 5.0                  # 分类损失权重
    LOSS_REG_W    = 2.0                  # 回归损失权重

    # ══════════════════════════════════════════════
    # 训练配置
    # ══════════════════════════════════════════════
    BATCH_SIZE      = 2
    NUM_WORKERS     = 4                  # Windows 调试时改为 0
    MAX_EPOCHS      = 80
    LR              = 1e-4
    WEIGHT_DECAY    = 1e-4
    GRAD_CLIP       = 10.0
    PHOTO_LOSS_WEIGHT = 0.1                # 光度损失权重
    LR_DECAY_EPOCH  = [50, 70]           # MultiStepLR 衰减节点
    LR_DECAY_GAMMA  = 0.1
    SAVE_FREQ       = 10                 # 每隔多少 epoch 保存一次

    # ══════════════════════════════════════════════
    # 推理配置
    # ══════════════════════════════════════════════
    SCORE_THRESH  = 0.1                  # 置信度阈值
    NMS_THRESH    = 0.25                  # NMS 中心距离阈值（米）
    MAX_BOXES     = 50                   # 每帧最多保留框数


cfg = Config()