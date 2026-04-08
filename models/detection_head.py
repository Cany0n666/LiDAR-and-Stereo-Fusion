# models/detection_head.py

# 基于Anchor的3D检测头 + Focal Loss + SmoothL1 Loss



import torch

import torch.nn as nn

import torch.nn.functional as F

import numpy as np

import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import cfg

from utils.box_utils import encode_boxes





# ─────────────────────────────────────────────────────────

# 1. Anchor 生成

# ─────────────────────────────────────────────────────────

def generate_anchors(feat_h, feat_w):

    """

    在BEV特征图每个位置生成 NUM_ANCHORS_PER_LOC 个Anchor

    返回: anchors [feat_h * feat_w * NUM_ANCHORS_PER_LOC, 7]

          格式: [x, y, z, w, l, h, rot]   z 为中心坐标

    """

    x_res = (cfg.PC_X_MAX - cfg.PC_X_MIN) / feat_h   # X轴(前向) → H方向

    y_res = (cfg.PC_Y_MAX - cfg.PC_Y_MIN) / feat_w   # Y轴(横向) → W方向



    xs = np.arange(feat_h) * x_res + cfg.PC_X_MIN + x_res / 2   # [feat_h]

    ys = np.arange(feat_w) * y_res + cfg.PC_Y_MIN + y_res / 2   # [feat_w]

    xx, yy = np.meshgrid(xs, ys, indexing="ij")   # [feat_h, feat_w]



    cls_names   = list(cfg.ANCHORS.keys())

    all_anchors = []



    for cls_name in cls_names:

        w, l, h = cfg.ANCHORS[cls_name]

        z_ctr   = cfg.ANCHOR_Z_CENTERS[cls_name]

        for rot in cfg.ANCHOR_ROTATIONS:

            anchor = np.stack([

                xx,

                yy,

                np.full_like(xx, z_ctr),

                np.full_like(xx, w),

                np.full_like(xx, l),

                np.full_like(xx, h),

                np.full_like(xx, rot),

            ], axis=-1)   # [feat_h, feat_w, 7]

            all_anchors.append(anchor.reshape(-1, 7))



    anchors = np.concatenate(all_anchors, axis=0)   # [N_anchors, 7]

    return torch.from_numpy(anchors.astype(np.float32))





# ─────────────────────────────────────────────────────────

# 2. 检测头网络

# ─────────────────────────────────────────────────────────

class DetectionHead(nn.Module):

    """

    输入: fused_feat [B, 256, H, W]

    输出:

        cls_pred: [B, NUM_ANCHORS_PER_LOC * NUM_CLASSES, H, W]

        reg_pred: [B, NUM_ANCHORS_PER_LOC * 7, H, W]

    """

    def __init__(self):

        super().__init__()

        mid_ch = 256

        self.shared = nn.Sequential(

            nn.Conv2d(cfg.FEAT_CH, mid_ch, 3, padding=1, bias=False),

            nn.BatchNorm2d(mid_ch),

            nn.ReLU(inplace=True),

            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),

            nn.BatchNorm2d(mid_ch),

            nn.ReLU(inplace=True),

        )

        self.cls_head = nn.Conv2d(

            mid_ch, cfg.NUM_ANCHORS_PER_LOC * cfg.NUM_CLASSES, 1)

        self.reg_head = nn.Conv2d(

            mid_ch, cfg.NUM_ANCHORS_PER_LOC * 7, 1)



        # Focal Loss 推荐初始化：让初始预测概率≈0.01，避免早期loss爆炸

        nn.init.constant_(self.cls_head.bias,

                          -np.log((1 - 0.01) / 0.01))



    def forward(self, fused_feat):

        x        = self.shared(fused_feat)

        cls_pred = self.cls_head(x)   # [B, A*C, H, W]

        reg_pred = self.reg_head(x)   # [B, A*7, H, W]

        return cls_pred, reg_pred





# ─────────────────────────────────────────────────────────

# 3. Focal Loss

# ─────────────────────────────────────────────────────────

class FocalLoss(nn.Module):

    def __init__(self, alpha=cfg.FOCAL_ALPHA, gamma=cfg.FOCAL_GAMMA):

        super().__init__()

        self.alpha = alpha

        self.gamma = gamma

    def forward(self, pred, target):

        p   = torch.sigmoid(pred)

        pt  = torch.where(target == 1, p, 1 - p)

        at  = torch.where(

            target == 1,

            torch.full_like(pred, self.alpha),

            torch.full_like(pred, 1 - self.alpha)

        )

        loss = -at * (1 - pt) ** self.gamma * torch.log(pt + 1e-8)

        return loss.mean()





# ─────────────────────────────────────────────────────────

# 4. BEV 轴对齐 IoU（修复：替换高斯近似）

# ─────────────────────────────────────────────────────────

def _bev_iou(anchors, gt_boxes):

    """

    计算 BEV 平面上轴对齐矩形的 IoU（忽略旋转角，快速近似）

    anchors:  [A, 7]  格式 [x, y, z, w, l, h, rot]

    gt_boxes: [M, 7]

    返回:     [A, M]

    """

    # 取 x, y, w, l

    a_x  = anchors[:, 0].unsqueeze(1)   # [A, 1]

    a_y  = anchors[:, 1].unsqueeze(1)

    a_w  = anchors[:, 3].unsqueeze(1)

    a_l  = anchors[:, 4].unsqueeze(1)



    g_x  = gt_boxes[:, 0].unsqueeze(0)  # [1, M]

    g_y  = gt_boxes[:, 1].unsqueeze(0)

    g_w  = gt_boxes[:, 3].unsqueeze(0)

    g_l  = gt_boxes[:, 4].unsqueeze(0)



    # 转换为角点格式

    a_x1 = a_x - a_w / 2;  a_x2 = a_x + a_w / 2

    a_y1 = a_y - a_l / 2;  a_y2 = a_y + a_l / 2

    g_x1 = g_x - g_w / 2;  g_x2 = g_x + g_w / 2

    g_y1 = g_y - g_l / 2;  g_y2 = g_y + g_l / 2



    # 交集

    inter_x1 = torch.max(a_x1, g_x1)

    inter_y1 = torch.max(a_y1, g_y1)

    inter_x2 = torch.min(a_x2, g_x2)

    inter_y2 = torch.min(a_y2, g_y2)

    inter_w  = (inter_x2 - inter_x1).clamp(min=0)

    inter_h  = (inter_y2 - inter_y1).clamp(min=0)

    inter    = inter_w * inter_h   # [A, M]



    # 各自面积

    area_a = a_w * a_l             # [A, 1]

    area_g = g_w * g_l             # [1, M]

    union  = area_a + area_g - inter + 1e-6



    return inter / union           # [A, M]





# ─────────────────────────────────────────────────────────

# 5. 损失计算（修复：多类独立二分类 + 真实IoU匹配）

# ─────────────────────────────────────────────────────────

def compute_loss(cls_pred, reg_pred, gt_boxes, gt_cls, anchors):

    """

    cls_pred: [B, A*C, H, W]

    reg_pred: [B, A*7, H, W]

    gt_boxes: list of [M, 7]   z 为中心坐标

    gt_cls:   list of [M]

    anchors:  [A_total, 7]

    """

    B              = cls_pred.shape[0]

    focal_loss_fn  = FocalLoss()

    total_cls_loss = 0.0

    total_reg_loss = 0.0



    C = cfg.NUM_CLASSES

    # 展平预测

    cls_flat = cls_pred.permute(0, 2, 3, 1).reshape(B, -1, C)  # [B, N, C]

    reg_flat = reg_pred.permute(0, 2, 3, 1).reshape(B, -1, 7)  # [B, N, 7]



    for b in range(B):

        boxes = gt_boxes[b]   # [M, 7]

        cls   = gt_cls[b]     # [M]



        if boxes.shape[0] == 0:

            # 无GT：所有 anchor 全部作为负样本，每类独立计算

            cls_loss_b = 0.0

            for c in range(C):

                dummy = torch.zeros(

                    cls_flat.shape[1], device=cls_pred.device)

                cls_loss_b += focal_loss_fn(cls_flat[b, :, c], dummy)

            total_cls_loss += cls_loss_b / C

            continue



        # ── 修复：使用真实 BEV IoU 匹配 ─────────────────────

        iou = _bev_iou(

            anchors.to(cls_pred.device),

            boxes.to(cls_pred.device)

        )                                        # [A, M]

        max_iou, max_idx = iou.max(dim=1)        # [A]

        pos_mask = max_iou >= cfg.POS_IOU_THRESH

        neg_mask = max_iou <  cfg.NEG_IOU_THRESH

        if b == 0: print(f"[DEBUG] max_iou max={max_iou.max():.4f} mean={max_iou.mean():.4f} pos={pos_mask.sum().item()} neg={neg_mask.sum().item()}")



        # 强制匹配：每个 GT 至少对应一个 anchor（IoU最大的那个）

        best_anchor_per_gt = iou.argmax(dim=0)   # [M]

        pos_mask[best_anchor_per_gt] = True

        neg_mask[best_anchor_per_gt] = False



        # Hard negative mining：负样本数量限制为正样本的10倍
        num_pos = pos_mask.sum().item()
        num_neg_keep = max(num_pos * 10, 100)
        neg_indices = neg_mask.nonzero(as_tuple=True)[0]
        if len(neg_indices) > num_neg_keep:
            perm = torch.randperm(len(neg_indices), device=cls_pred.device)
            neg_mask_new = torch.zeros_like(neg_mask)
            neg_mask_new[neg_indices[perm[:num_neg_keep]]] = True
            neg_mask = neg_mask_new
        valid = pos_mask | neg_mask



        # ── 修复：每类独立二分类，不再用 .max(-1).values ────

        cls_loss_b = 0.0

        for c in range(C):

            cls_target_c = torch.zeros(

                anchors.shape[0], device=cls_pred.device)

            # 正样本中属于类别c的anchor置1

            if pos_mask.sum() > 0:

                pos_gt_cls = cls[max_idx[pos_mask]]          # [P]

                cls_target_c[pos_mask] = (pos_gt_cls == c).float()

            cls_loss_b += focal_loss_fn(

                cls_flat[b, :, c][valid],

                cls_target_c[valid]

            )

        total_cls_loss += cls_loss_b / C



        # ── 回归损失（仅正样本）──────────────────────────────

        if pos_mask.sum() > 0:

            pos_anchors  = anchors[pos_mask].to(cls_pred.device)

            pos_gt       = boxes[max_idx[pos_mask]]

            reg_target   = encode_boxes(pos_gt, pos_anchors)

            reg_pred_pos = reg_flat[b][pos_mask]

            total_reg_loss += F.smooth_l1_loss(

                reg_pred_pos, reg_target)



    cls_loss = total_cls_loss / B

    reg_loss = total_reg_loss / B

    total    = cfg.LOSS_CLS_W * cls_loss + cfg.LOSS_REG_W * reg_loss

    return total, cls_loss, reg_loss

