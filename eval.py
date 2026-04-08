# eval.py  ── 评估程序（计算 KITTI mAP）

import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from config import cfg
from data.kitti_dataset    import KITTIDataset, collate_fn
from models.lidar_bev      import points_to_bev
from models.detection_head import generate_anchors
from utils.box_utils       import decode_boxes, nms_3d
from train                 import FusionDetector


# ─────────────────────────────────────────────────────────
# 推理单帧
# ─────────────────────────────────────────────────────────
@torch.no_grad()
def inference_single(model, img_l, img_r, bev, anchors, device):
    model.eval()
    img_l = img_l.unsqueeze(0).to(device)
    img_r = img_r.unsqueeze(0).to(device)
    bev   = bev.unsqueeze(0).to(device)

    _, cls_pred, reg_pred = model(img_l, img_r, bev)

    C        = cfg.NUM_CLASSES
    cls_flat = cls_pred[0].permute(1, 2, 0).reshape(-1, C)   # [N, C]
    reg_flat = reg_pred[0].permute(1, 2, 0).reshape(-1, 7)   # [N, 7]

    scores_all         = torch.sigmoid(cls_flat)              # [N, C]
    max_scores, labels = scores_all.max(dim=-1)               # [N]

    mask = max_scores >= cfg.SCORE_THRESH
    if mask.sum() == 0:
        return {
            "boxes" : torch.zeros(0, 7),
            "scores": torch.zeros(0),
            "labels": torch.zeros(0, dtype=torch.long),
        }

    boxes_dec = decode_boxes(reg_flat[mask], anchors[mask])
    scores    = max_scores[mask]
    labels    = labels[mask]

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
            dist     = torch.norm(boxes[i, :2] - gt[:, :2], dim=-1)
            min_d, j = dist.min(0)
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
    all_preds  = {c: [] for c in cls_names}
    all_gts    = {c: [] for c in cls_names}

    for batch in val_loader:
        img_l = batch["img_l"][0]
        img_r = batch["img_r"][0]

        # ✅ points → bev
        pts = batch["points"][0]          # Tensor[N, 4]
        if pts.shape[0] > 0:
            bev_np = points_to_bev(pts.numpy())
        else:
            bev_np = np.zeros(
                (cfg.BEV_C, cfg.BEV_H, cfg.BEV_W), dtype=np.float32)
        bev = torch.from_numpy(bev_np)

        pred = inference_single(
            model, img_l, img_r, bev, anchors, device)

        # ✅ gt_boxes / gt_cls 直接用 Tensor，不用字典
        gt_boxes = batch["gt_boxes"][0]   # Tensor[M, 7]
        gt_cls   = batch["gt_cls"][0]     # Tensor[M]

        for ci, cname in enumerate(cls_names):
            # 预测：按类别过滤
            pmask = pred["labels"] == ci
            all_preds[cname].append({
                "boxes" : pred["boxes"][pmask],
                "scores": pred["scores"][pmask],
            })

            # GT：按类别过滤
            if gt_boxes.shape[0] > 0:
                gmask = gt_cls == ci
                all_gts[cname].append(gt_boxes[gmask])   # Tensor[K, 7]
            else:
                all_gts[cname].append(torch.zeros(0, 7))

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
