# visualize.py  ── 可视化推理结果（图像投影 + BEV俯视图）

import os
import argparse
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader

from config              import cfg
from data.kitti_dataset  import KITTIDataset, collate_fn
from models.lidar_bev    import points_to_bev
from models.detection_head import generate_anchors
from utils.box_utils     import decode_boxes, nms_3d
from train               import FusionDetector

# ──────────────────────────────────────────────
# 类别颜色  BGR (OpenCV) / RGB (matplotlib)
# ──────────────────────────────────────────────
CLS_NAMES  = ["Car", "Pedestrian", "Cyclist"]
CLS_COLORS_BGR = {
    0: (0,   255,   0),   # Car        绿
    1: (0,   128, 255),   # Pedestrian 橙
    2: (255,   0, 255),   # Cyclist    紫
}
CLS_COLORS_RGB = {
    0: (0.0,  1.0,  0.0),
    1: (1.0,  0.5,  0.0),
    2: (0.8,  0.0,  1.0),
}

# ──────────────────────────────────────────────
# 几何工具
# ──────────────────────────────────────────────
def get_3d_box_corners(x, y, z, w, l, h, theta):
    """
    LiDAR 坐标系下的 3D 框 8 个角点
    box: (x, y, z, w, l, h, theta)  z 为底面中心
    返回 [8, 3]
    """
    corners_x = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2])
    corners_y = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2])
    corners_z = np.array([ 0,    0,    0,    0,    h,    h,    h,    h  ])

    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [0,              0,             1]])
    corners = R @ np.stack([corners_x, corners_y, corners_z], axis=0)  # [3,8]
    corners[0] += x
    corners[1] += y
    corners[2] += z
    return corners.T  # [8, 3]


def lidar_to_camera(corners_lidar, Tr_velo_to_cam):
    """LiDAR [8,3] → Camera [8,3]"""
    ones   = np.ones((corners_lidar.shape[0], 1))
    pts_h  = np.hstack([corners_lidar, ones])          # [8,4]
    pts_cam = (Tr_velo_to_cam @ pts_h.T).T             # [8,4]
    return pts_cam[:, :3]


def project_to_image(corners_cam, P2):
    """Camera [8,3] → Image [8,2]"""
    ones   = np.ones((corners_cam.shape[0], 1))
    pts_h  = np.hstack([corners_cam, ones])            # [8,4]
    pts_img = (P2 @ pts_h.T).T                         # [8,3]
    pts_img[:, :2] /= (pts_img[:, 2:3] + 1e-8)
    return pts_img[:, :2]                              # [8,2]


def draw_box_3d_on_image(img, corners_2d, color_bgr, thickness=2):
    """在图像上画 3D 框的 12 条棱"""
    edges = [
        (0,1),(1,2),(2,3),(3,0),   # 底面
        (4,5),(5,6),(6,7),(7,4),   # 顶面
        (0,4),(1,5),(2,6),(3,7),   # 侧棱
    ]
    pts = corners_2d.astype(np.int32)
    for i, j in edges:
        p1 = tuple(pts[i])
        p2 = tuple(pts[j])
        cv2.line(img, p1, p2, color_bgr, thickness, cv2.LINE_AA)


# ──────────────────────────────────────────────
# 推理（复用 eval.py 逻辑）
# ──────────────────────────────────────────────
@torch.no_grad()
def inference_single(model, img_l, img_r, bev, anchors, device):
    model.eval()
    img_l = img_l.unsqueeze(0).to(device)
    img_r = img_r.unsqueeze(0).to(device)
    bev   = bev.unsqueeze(0).to(device)

    _, cls_pred, reg_pred = model(img_l, img_r, bev)

    C        = cfg.NUM_CLASSES
    cls_flat = cls_pred[0].permute(1, 2, 0).reshape(-1, C)
    reg_flat = reg_pred[0].permute(1, 2, 0).reshape(-1, 7)

    scores_all         = torch.sigmoid(cls_flat)
    max_scores, labels = scores_all.max(dim=-1)

    mask = max_scores >= cfg.SCORE_THRESH
    if mask.sum() == 0:
        return {"boxes": torch.zeros(0, 7),
                "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.long)}

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


# ──────────────────────────────────────────────
# 单帧可视化
# ──────────────────────────────────────────────
def visualize_sample(sample, pred, save_path):
    """
    生成左右拼接图：
      左：图像 + 3D 框投影
      右：BEV 点云 + 框俯视
    """
    # ---------- 解包 calib ----------
    calib = sample["calib"]
    P2             = calib["P2"]              # [3,4]
    Tr_velo_to_cam = calib["Tr_velo_to_cam"]  # [4,4]

    # ---------- 左图 ----------
    img_np = sample["img_l"].permute(1, 2, 0).numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    H_img, W_img = img_bgr.shape[:2]

    boxes  = pred["boxes"].numpy()    # [N, 7]
    scores = pred["scores"].numpy()
    labels = pred["labels"].numpy()

    for i in range(len(boxes)):
        x, y, z, w, l, h, theta = boxes[i]
        cls   = int(labels[i])
        score = scores[i]
        color_bgr = CLS_COLORS_BGR.get(cls, (255, 255, 255))

        corners_lidar = get_3d_box_corners(x, y, z, w, l, h, theta)
        corners_cam   = lidar_to_camera(corners_lidar, Tr_velo_to_cam)

        # 过滤相机后方的框
        if corners_cam[:, 2].max() < 0:
            continue

        corners_2d = project_to_image(corners_cam, P2)
        draw_box_3d_on_image(img_bgr, corners_2d, color_bgr)

        # 标签文字
        u, v = int(corners_2d[:, 0].mean()), int(corners_2d[:, 1].min()) - 5
        label_txt = f"{CLS_NAMES[cls]} {score:.2f}"
        cv2.putText(img_bgr, label_txt, (u, max(v, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1, cv2.LINE_AA)

    # ---------- BEV 图 ----------
    fig_bev, ax_bev = plt.subplots(1, 1, figsize=(6, 10))
    ax_bev.set_facecolor("#1a1a2e")
    ax_bev.set_title("BEV View", color="white")
    ax_bev.tick_params(colors="white")
    for spine in ax_bev.spines.values():
        spine.set_edgecolor("white")

    # 点云散点（取 x-y 平面）
    pts = sample["points"].numpy()   # [N, 4]
    if pts.shape[0] > 0:
        # 只显示前方区域
        mask_pts = (pts[:, 0] > cfg.X_MIN) & (pts[:, 0] < cfg.X_MAX) & \
                   (pts[:, 1] > cfg.Y_MIN) & (pts[:, 1] < cfg.Y_MAX)
        pts_vis = pts[mask_pts]
        if pts_vis.shape[0] > 0:
            intensity = pts_vis[:, 3] if pts_vis.shape[1] >= 4 else np.ones(pts_vis.shape[0])
            ax_bev.scatter(pts_vis[:, 1], pts_vis[:, 0],
                           s=0.3, c=intensity, cmap="plasma",
                           vmin=0, vmax=1, alpha=0.6)

    # BEV 框（x-y 平面的矩形）
    for i in range(len(boxes)):
        x, y, z, w, l, h, theta = boxes[i]
        cls   = int(labels[i])
        score = scores[i]
        color_rgb = CLS_COLORS_RGB.get(cls, (1, 1, 1))

        # 旋转矩形 4 个角点（BEV: x=前, y=左）
        corners_lidar = get_3d_box_corners(x, y, z, w, l, h, theta)[:4, :2]  # 底面4点 [4,2]
        poly = plt.Polygon(corners_lidar[:, [1, 0]],  # (y, x) → (横, 纵)
                           fill=False, edgecolor=color_rgb, linewidth=1.5)
        ax_bev.add_patch(poly)
        ax_bev.text(y, x, f"{CLS_NAMES[cls]}\n{score:.2f}",
                    color=color_rgb, fontsize=6, ha="center")

    ax_bev.set_xlim(cfg.Y_MIN, cfg.Y_MAX)
    ax_bev.set_ylim(cfg.X_MIN, cfg.X_MAX)
    ax_bev.set_xlabel("Y (m)", color="white")
    ax_bev.set_ylabel("X (m)", color="white")
    ax_bev.set_aspect("equal")
    fig_bev.patch.set_facecolor("#1a1a2e")

    # 把 BEV 图转成 numpy
    fig_bev.tight_layout()
    fig_bev.canvas.draw()
    bev_rgb = np.frombuffer(fig_bev.canvas.tostring_rgb(), dtype=np.uint8)
    bev_rgb = bev_rgb.reshape(fig_bev.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig_bev)
    bev_bgr = cv2.cvtColor(bev_rgb, cv2.COLOR_RGB2BGR)

    # ---------- 拼接 ----------
    # 统一高度
    target_h = max(H_img, bev_bgr.shape[0])
    if img_bgr.shape[0] != target_h:
        img_bgr = cv2.resize(img_bgr, (int(W_img * target_h / H_img), target_h))
    if bev_bgr.shape[0] != target_h:
        bev_bgr = cv2.resize(bev_bgr,
                             (int(bev_bgr.shape[1] * target_h / bev_bgr.shape[0]), target_h))

    combined = np.hstack([img_bgr, bev_bgr])
    cv2.imwrite(save_path, combined)
    print(f"  Saved → {save_path}")


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="KITTI 3D Detection Visualizer")
    parser.add_argument("--ckpt",        default="checkpoints/best.pth")
    parser.add_argument("--split",       default="val", choices=["train", "val"])
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--save_dir",    default="vis_results")
    parser.add_argument("--score_thresh",type=float, default=None,
                        help="覆盖 config.SCORE_THRESH，默认用 config 值")
    args = parser.parse_args()

    if args.score_thresh is not None:
        cfg.SCORE_THRESH = args.score_thresh

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Ckpt   : {args.ckpt}")
    print(f"Split  : {args.split}  |  samples : {args.num_samples}")
    print(f"ScoreTh: {cfg.SCORE_THRESH}")

    # ---------- 模型 ----------
    model = FusionDetector().to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print("✅ Checkpoint loaded!")

    # ---------- 数据 ----------
    dataset = KITTIDataset(cfg.KITTI_ROOT, split=args.split)
    anchors = generate_anchors(cfg.FUSE_BEV_H, cfg.FUSE_BEV_W).to(device)

    # ---------- 推理 & 可视化 ----------
    indices = np.random.choice(len(dataset),
                               min(args.num_samples, len(dataset)),
                               replace=False)
    for rank, idx in enumerate(indices):
        sample = dataset[int(idx)]

        # points → bev
        pts = sample["points"]
        if pts.shape[0] > 0:
            bev_np = points_to_bev(pts.numpy())
        else:
            bev_np = np.zeros((cfg.BEV_C, cfg.BEV_H, cfg.BEV_W), dtype=np.float32)
        bev = torch.from_numpy(bev_np)

        pred = inference_single(
            model,
            sample["img_l"], sample["img_r"], bev,
            anchors, device
        )

        n_det = len(pred["boxes"])
        print(f"[{rank+1:02d}/{args.num_samples}] idx={idx:04d}  detections={n_det}")

        save_path = os.path.join(args.save_dir, f"vis_{rank+1:02d}_idx{idx:04d}.jpg")
        visualize_sample(sample, pred, save_path)

    print(f"\n✅ 全部完成，结果保存在 ./{args.save_dir}/")


if __name__ == "__main__":
    main()