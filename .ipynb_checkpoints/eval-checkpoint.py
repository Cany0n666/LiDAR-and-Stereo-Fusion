# eval.py  ── 评估程序（计算 KITTI mAP）

import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from config import cfg
from data.kitti_dataset    import KITTIDataset, collate_fn
from models.lidar_bev      import points_to_bev
from models.detection_head import generate_anchors
from utils.box_utils       import decode_boxes, nms_3d   # ← 统一使用 box_utils
from train                 import FusionDetector, batch_points_to_bev


# ─────────────────────────────────────────────────────────
# 推理单帧
# ─────────────────────────────────────────────────────────
@torch.no_grad()
def inference_single(model, img_l, img_r, bev, anchors, device):
    """
    返回 dict:
        boxes:  [K, 7]
        scores: [K]
        labels: [K]  (long)
    """
    model.eval()
    img_l = img_l.unsqueeze(0).to(device)
    img_r = img_r.unsqueeze(0).to(device)
    bev   = bev.unsqueeze(0).to(device)

    _, cls_pred, reg_pred = model(img_l, img_r, bev)

    C = cfg.NUM_CLASSES
    # 展平
    cls_flat = cls_pred[0].permute(1, 2, 0).reshape(-1, C)   # [N, C]
    reg_flat = reg_pred[0].permute(1, 2, 0).reshape(-1, 7)   # [N, 7]

    scores_all         = torch.sigmoid(cls_flat)              # [N, C]
    max_scores, labels = scores_all.max(dim=-1)               # [N]

    # 置信度过滤
    mask = max_scores >= cfg.SCORE_THRESH
    if mask.sum() == 0:
        return {
            "boxes" : torch.zeros(0, 7),
            "scores": torch.zeros(0),
            "labels": torch.zeros(0, dtype=torch.long),
        }

    # 使用 box_utils.decode_boxes 解码
    boxes_dec = decode_boxes(reg_flat[mask], anchors[mask])
    scores    = max_scores[mask]
    labels    = labels[mask]

    # 使用 box_utils.nms_3d（基于 BEV IoU）
    keep = nms_3d(boxes_dec, scores, threshold=cfg.NMS_THRESH)
    keep = keep[:cfg.MAX_BOXES]

    return {
        "boxes" : boxes_dec[keep].cpu(),
        "scores": scores[keep].cpu(),
        "labels": labels[keep].cpu(),
    }


# ─────────────────────────────────────────────────────────
# AP 计算（VOC 11点插值）
# ─────────────────────────────────────────────────────────
def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p   = prec[rec >= t].max() if (rec >= t).any() else 0.0
        ap += p / 11.0
    return ap


def compute_ap(preds, gts, dist_thresh=2.0) -> float:
    """
    简化 AP：用 BEV 中心距离（< dist_thresh 米）判断 TP/FP
    preds: list of dict  每帧预测
    gts:   list of Tensor [M, 7]  每帧GT
    """
    tp_list, fp_list, score_list = [], [], []
    n_gt = sum(g.shape[0] for g in gts)

    for pred, gt in zip(preds, gts):
        if pred["boxes"].shape[0] == 0:
            continue
        scores  = pred["scores"]
        boxes   = pred["boxes"]
        matched = torch.zeros(gt.shape[0], dtype=torch.bool)

        for i in range(boxes.shape[0]):
            score_list.append(scores[i].item())
            if gt.shape[0] == 0:
                tp_list.append(0)
                fp_list.append(1)
                continue
            dist       = torch.norm(boxes[i, :2] - gt[:, :2], dim=-1)
            min_d, j   = dist.min(0)
            if min_d < dist_thresh and not matched[j]:
                tp_list.append(1)
                fp_list.append(0)
                matched[j] = True
            else:
                tp_list.append(0)
                fp_list.append(1)

    if len(score_list) == 0 or n_gt == 0:
        return 0.0

    order  = np.argsort(score_list)[::-1]
    tp_cum = np.cumsum(np.array(tp_list)[order])
    fp_cum = np.cumsum(np.array(fp_list)[order])
    rec    = tp_cum / (n_gt + 1e-8)
    prec   = tp_cum / (tp_cum + fp_cum + 1e-8)
    return voc_ap(rec, prec)


# ─────────────────────────────────────────────────────────
# 主评估函数
# ─────────────────────────────────────────────────────────
def evaluate(ckpt_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = FusionDetector().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    print(f"[INFO] 加载权重: {ckpt_path}")

    val_set    = KITTIDataset(cfg.KITTI_ROOT, split="val")
    val_loader = DataLoader(
        val_set, batch_size=1,
        shuffle=False, collate_fn=collate_fn)
    anchors = generate_anchors(
        cfg.FUSE_BEV_H, cfg.FUSE_BEV_W).to(device)

    cls_names = ["Car", "Pedestrian", "Cyclist"]
    all_preds = {c: [] for c in cls_names}
    all_gts   = {c: [] for c in cls_names}

    for batch in val_loader:
        img_l  = batch["img_l"][0]
        img_r  = batch["img_r"][0]
        bev_np = points_to_bev(batch["lidar"][0].numpy()) \
                 if batch["lidar"] else \
                 np.zeros((cfg.BEV_C, cfg.BEV_H, cfg.BEV_W), dtype=np.float32)
        bev  = torch.from_numpy(bev_np)
        pred = inference_single(
            model, img_l, img_r, bev, anchors, device)

        for ci, cname in enumerate(cls_names):
            pmask = pred["labels"] == ci
            all_preds[cname].append({
                "boxes" : pred["boxes"][pmask],
                "scores": pred["scores"][pmask],
            })
            # gt_cls 与 CLASS_NAMES 对应
            from data.kitti_dataset import KITTIDataset as _KD
            name_to_idx = {n: i for i, n in enumerate(_KD.CLASS_NAMES)}
            gmask = torch.tensor([
                lbl["type"] == cname
                for lbl in batch["labels"][0]
            ], dtype=torch.bool) if batch["labels"][0] else torch.zeros(0, dtype=torch.bool)

            gt_boxes_list = torch.stack([
                torch.from_numpy(lbl["bbox"])   # 用 location 替代
                for lbl in batch["labels"][0]
                if lbl["type"] == cname
            ]) if any(lbl["type"] == cname
                      for lbl in batch["labels"][0]) \
              else torch.zeros(0, 7)

            all_gts[cname].append(gt_boxes_list)

    # 打印结果
    print("\n===== 评估结果 =====")
    aps = []
    for cname in cls_names:
        ap = compute_ap(all_preds[cname], all_gts[cname])
        aps.append(ap)
        print(f"  {cname:12s}  AP = {ap * 100:.2f}%")
    print(f"  {'mAP':12s}       = {sum(aps) / len(aps) * 100:.2f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="./checkpoints/best.pth")
    args = parser.parse_args()
    evaluate(args.ckpt)