"""
边界框工具函数
统一全项目的编解码、IoU、NMS逻辑，避免重复定义
"""
import torch
import numpy as np
from typing import Optional


def limit_period(val: torch.Tensor,
                 offset: float = 0.5,
                 period: float = np.pi) -> torch.Tensor:
    """将角度限制在 [-period*offset, period*(1-offset)] 范围内"""
    return val - torch.floor(val / period + offset) * period


def boxes_to_corners_3d(boxes: torch.Tensor) -> torch.Tensor:
    """
    将3D边界框转换为8个角点
    Args:
        boxes: [N, 7]  (x, y, z, w, l, h, θ)
                x/y/z 均为中心点坐标（与 config.py 一致）
    Returns:
        corners: [N, 8, 3]
    """
    x, y, z = boxes[:, 0], boxes[:, 1], boxes[:, 2]
    w, l, h = boxes[:, 3], boxes[:, 4], boxes[:, 5]
    theta   = boxes[:, 6]

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    dx = w / 2
    dy = l / 2
    dz = h / 2   # ← 修复：z 是中心，上下各 h/2

    corners = []
    for z_sign in [-1, 1]:                                    # 底面 / 顶面
        for sx, sy in [(-1, -1), (-1, 1), (1, 1), (1, -1)]:  # 四个角
            cx = x + sx * dx * cos_t - sy * dy * sin_t
            cy = y + sx * dx * sin_t + sy * dy * cos_t
            cz = z + z_sign * dz
            corners.append(torch.stack([cx, cy, cz], dim=-1))

    return torch.stack(corners, dim=1)   # [N, 8, 3]


def encode_boxes(boxes: torch.Tensor,
                 anchors: torch.Tensor) -> torch.Tensor:
    """
    编码边界框（SECOND 编码方式，与 detection_head.py 统一）
    Args:
        boxes:   [N, 7]  GT框，z 为中心
        anchors: [N, 7]  Anchor，z 为中心
    Returns:
        encoded: [N, 7]
    """
    diagonal = torch.sqrt(
        anchors[:, 3] ** 2 + anchors[:, 4] ** 2).clamp(min=1e-6)

    return torch.stack([
        (boxes[:, 0] - anchors[:, 0]) / diagonal,
        (boxes[:, 1] - anchors[:, 1]) / diagonal,
        (boxes[:, 2] - anchors[:, 2]) / anchors[:, 5].clamp(min=1e-6),
        torch.log(boxes[:, 3] / anchors[:, 3].clamp(min=1e-6)),
        torch.log(boxes[:, 4] / anchors[:, 4].clamp(min=1e-6)),
        torch.log(boxes[:, 5] / anchors[:, 5].clamp(min=1e-6)),
        boxes[:, 6] - anchors[:, 6],
    ], dim=-1)


def decode_boxes(encoded: torch.Tensor,
                 anchors: torch.Tensor) -> torch.Tensor:
    """
    解码边界框（与 encode_boxes 互为逆操作）
    Args:
        encoded: [N, 7]
        anchors: [N, 7]
    Returns:
        boxes: [N, 7]  z 为中心
    """
    diagonal = torch.sqrt(
        anchors[:, 3] ** 2 + anchors[:, 4] ** 2).clamp(min=1e-6)

    return torch.stack([
        encoded[:, 0] * diagonal + anchors[:, 0],
        encoded[:, 1] * diagonal + anchors[:, 1],
        encoded[:, 2] * anchors[:, 5] + anchors[:, 2],
        torch.exp(encoded[:, 3].clamp(max=10)) * anchors[:, 3],
        torch.exp(encoded[:, 4].clamp(max=10)) * anchors[:, 4],
        torch.exp(encoded[:, 5].clamp(max=10)) * anchors[:, 5],
        encoded[:, 6] + anchors[:, 6],
    ], dim=-1)


def boxes_iou_bev(boxes1: torch.Tensor,
                  boxes2: torch.Tensor) -> torch.Tensor:
    """
    BEV视角下的 IoU（AABB 近似，不考虑旋转）
    Args:
        boxes1: [N, 7]
        boxes2: [M, 7]
    Returns:
        iou: [N, M]
    """
    b1_x1 = boxes1[:, 0] - boxes1[:, 3] / 2
    b1_x2 = boxes1[:, 0] + boxes1[:, 3] / 2
    b1_y1 = boxes1[:, 1] - boxes1[:, 4] / 2
    b1_y2 = boxes1[:, 1] + boxes1[:, 4] / 2

    b2_x1 = boxes2[:, 0] - boxes2[:, 3] / 2
    b2_x2 = boxes2[:, 0] + boxes2[:, 3] / 2
    b2_y1 = boxes2[:, 1] - boxes2[:, 4] / 2
    b2_y2 = boxes2[:, 1] + boxes2[:, 4] / 2

    inter_x = (torch.min(b1_x2[:, None], b2_x2)
               - torch.max(b1_x1[:, None], b2_x1)).clamp(min=0)
    inter_y = (torch.min(b1_y2[:, None], b2_y2)
               - torch.max(b1_y1[:, None], b2_y1)).clamp(min=0)
    inter_area = inter_x * inter_y

    area1 = boxes1[:, 3] * boxes1[:, 4]
    area2 = boxes2[:, 3] * boxes2[:, 4]

    return inter_area / (area1[:, None] + area2 - inter_area + 1e-6)


def boxes_iou_3d(boxes1: torch.Tensor,
                 boxes2: torch.Tensor) -> torch.Tensor:
    """
    3D IoU（BEV面积重叠 × 高度重叠）
    ⚠️ z 为中心坐标，与 config.py / detection_head.py 保持一致
    Args:
        boxes1: [N, 7]  (x, y, z_center, w, l, h, θ)
        boxes2: [M, 7]
    Returns:
        iou: [N, M]
    """
    # BEV 交集
    bev_iou   = boxes_iou_bev(boxes1, boxes2)           # [N, M]
    bev_area1 = boxes1[:, 3] * boxes1[:, 4]             # [N]
    bev_area2 = boxes2[:, 3] * boxes2[:, 4]             # [M]
    # 反推 BEV 交集面积
    bev_inter = bev_iou * (bev_area1[:, None]
                           + bev_area2 - bev_iou
                           * (bev_area1[:, None] + bev_area2))
    # 修复：用更直接的方式算 BEV 交集面积
    b1_x1 = boxes1[:, 0] - boxes1[:, 3] / 2
    b1_x2 = boxes1[:, 0] + boxes1[:, 3] / 2
    b1_y1 = boxes1[:, 1] - boxes1[:, 4] / 2
    b1_y2 = boxes1[:, 1] + boxes1[:, 4] / 2
    b2_x1 = boxes2[:, 0] - boxes2[:, 3] / 2
    b2_x2 = boxes2[:, 0] + boxes2[:, 3] / 2
    b2_y1 = boxes2[:, 1] - boxes2[:, 4] / 2
    b2_y2 = boxes2[:, 1] + boxes2[:, 4] / 2

    inter_x = (torch.min(b1_x2[:, None], b2_x2)
               - torch.max(b1_x1[:, None], b2_x1)).clamp(min=0)
    inter_y = (torch.min(b1_y2[:, None], b2_y2)
               - torch.max(b1_y1[:, None], b2_y1)).clamp(min=0)
    inter_area = inter_x * inter_y                       # [N, M]

    # ── 高度重叠（z 是中心，修复原版的底部语义错误）────────────
    b1_z1 = boxes1[:, 2] - boxes1[:, 5] / 2             # 底部 [N]
    b1_z2 = boxes1[:, 2] + boxes1[:, 5] / 2             # 顶部 [N]
    b2_z1 = boxes2[:, 2] - boxes2[:, 5] / 2             # 底部 [M]
    b2_z2 = boxes2[:, 2] + boxes2[:, 5] / 2             # 顶部 [M]

    inter_z = (torch.min(b1_z2[:, None], b2_z2)
               - torch.max(b1_z1[:, None], b2_z1)).clamp(min=0)  # [N, M]

    inter_vol = inter_area * inter_z                     # [N, M]

    vol1 = boxes1[:, 3] * boxes1[:, 4] * boxes1[:, 5]   # [N]
    vol2 = boxes2[:, 3] * boxes2[:, 4] * boxes2[:, 5]   # [M]

    return inter_vol / (vol1[:, None] + vol2 - inter_vol + 1e-6)


def nms_3d(boxes: torch.Tensor,
           scores: torch.Tensor,
           threshold: float = 0.25) -> torch.Tensor:
    """
    基于 BEV IoU 的 3D NMS
    Args:
        boxes:     [N, 7]
        scores:    [N]
        threshold: IoU 阈值（默认 0.25，比 2D 更严格）
    Returns:
        keep: [K]  保留框的索引
    """
    if boxes.shape[0] == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep  = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        iou  = boxes_iou_bev(boxes[i:i+1], boxes[order[1:]])[0]
        mask = iou <= threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)