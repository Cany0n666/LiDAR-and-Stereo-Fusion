# train.py  ── 训练主程序

import os, sys, time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import cfg
from data.kitti_dataset import KITTIDataset, collate_fn
from models.stereo_net   import LightStereoNet
from models.lidar_bev    import BEVEncoder, points_to_bev
from models.fusion_transformer import FusionTransformer
from models.detection_head import DetectionHead, generate_anchors, compute_loss


# ─────────────────────────────────────────────────────────
# 完整模型（把4个子模块组合起来）
# ─────────────────────────────────────────────────────────
class FusionDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stereo_net  = LightStereoNet()
        self.bev_encoder = BEVEncoder()
        self.fusion      = FusionTransformer()
        self.det_head    = DetectionHead()

    def forward(self, img_l, img_r, bev):
        # ① 视差估计 + 特征提取
        disp, disp_feat = self.stereo_net(img_l, img_r)

        # ② BEV特征编码
        bev_feat = self.bev_encoder(bev)

        # ③ 跨模态融合
        fused = self.fusion(disp_feat, bev_feat)

        # ④ 检测头
        cls_pred, reg_pred = self.det_head(fused)

        return disp, cls_pred, reg_pred


# ─────────────────────────────────────────────────────────
# 工具：点云batch → BEV tensor
# ─────────────────────────────────────────────────────────
def batch_points_to_bev(points_list, device):
    import numpy as np
    bevs = []
    for pts in points_list:
        bev = points_to_bev(pts.numpy())   # [3, H, W]
        bevs.append(torch.from_numpy(bev))
    return torch.stack(bevs).to(device)    # [B, 3, H, W]

# ─────────────────────────────────────────────────────────
# 自监督光度损失（左图 warp 重建误差）
# ─────────────────────────────────────────────────────────
def photometric_loss(img_l, img_r, disp, alpha=0.85):
    """
    用预测视差把右图 warp 到左图，计算 SSIM+L1 光度损失。
    img_l / img_r : [B, 3, H, W]  已归一化
    disp          : [B, 1, H, W]  预测视差（像素单位）
    """
    import torch.nn.functional as F
    B, _, H, W = img_l.shape
    disp_r = disp.squeeze(1)

    xs = torch.linspace(0, W-1, W, device=img_l.device)
    ys = torch.linspace(0, H-1, H, device=img_l.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

    sample_x = grid_x - disp_r
    sample_x_norm = 2.0 * sample_x / (W - 1) - 1.0
    sample_y_norm = 2.0 * grid_y  / (H - 1) - 1.0
    grid = torch.stack([sample_x_norm, sample_y_norm], dim=-1)

    img_r_warp = F.grid_sample(
        img_r, grid, mode="bilinear",
        padding_mode="border", align_corners=True)

    l1 = (img_l - img_r_warp).abs().mean()

    def _ssim(x, y, c1=1e-4, c2=9e-4):
        mu_x   = F.avg_pool2d(x,   3, 1, 1)
        mu_y   = F.avg_pool2d(y,   3, 1, 1)
        sig_x  = F.avg_pool2d(x*x, 3, 1, 1) - mu_x**2
        sig_y  = F.avg_pool2d(y*y, 3, 1, 1) - mu_y**2
        sig_xy = F.avg_pool2d(x*y, 3, 1, 1) - mu_x*mu_y
        s = ((2*mu_x*mu_y + c1)*(2*sig_xy + c2)) /             ((mu_x**2 + mu_y**2 + c1)*(sig_x + sig_y + c2))
        return s.clamp(0, 1)

    ssim_loss = (1.0 - _ssim(img_l, img_r_warp)).mean() * 0.5
    return alpha * ssim_loss + (1.0 - alpha) * l1



# ─────────────────────────────────────────────────────────
# 主训练循环
# ─────────────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # ── 数据集 ──
    train_set = KITTIDataset(cfg.KITTI_ROOT, split="train")
    val_set   = KITTIDataset(cfg.KITTI_ROOT, split="val")
    train_loader = DataLoader(
        train_set, batch_size=cfg.BATCH_SIZE,
        shuffle=True, num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=1,
        shuffle=False, num_workers=2,
        collate_fn=collate_fn
    )
    print(f"[INFO] 训练集: {len(train_set)} 样本 | 验证集: {len(val_set)} 样本")

    # ── 模型 ──
    model = FusionDetector().to(device)

    # ── 优化器 ──
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg.LR_DECAY_EPOCH,
        gamma=cfg.LR_DECAY_GAMMA
    )

    # ── 预生成Anchor（固定，不需要梯度）──
    anchors = generate_anchors(cfg.FUSE_BEV_H, cfg.FUSE_BEV_W).to(device)

    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    best_val_loss = float("inf")

    # ── Epoch循环 ──
    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        model.train()
        t0 = time.time()
        total_loss_sum = 0.0

        for step, batch in enumerate(train_loader):
            img_l = batch["img_l"].to(device)
            img_r = batch["img_r"].to(device)
            bev   = batch_points_to_bev(batch["points"], device)
            gt_boxes = [b.to(device) for b in batch["gt_boxes"]]
            gt_cls   = [c.to(device) for c in batch["gt_cls"]]

            optimizer.zero_grad()

            # 前向传播
            disp, cls_pred, reg_pred = model(img_l, img_r, bev)

            # 计算损失
            loss, cls_loss, reg_loss = compute_loss(
                cls_pred, reg_pred, gt_boxes, gt_cls, anchors
            )
            photo_loss = photometric_loss(img_l, img_r, disp)
            loss = loss + cfg.PHOTO_LOSS_WEIGHT * photo_loss

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()

            total_loss_sum += loss.item()

            if step % 50 == 0:
                print(f"  Epoch {epoch:03d} | Step {step:04d} | "
                      f"Loss: {float(loss):.4f} "
                      f"(cls={float(cls_loss):.4f}, "
                      f"reg={float(reg_loss):.4f})")

        scheduler.step()
        avg_loss = total_loss_sum / len(train_loader)
        elapsed  = time.time() - t0
        print(f"[Epoch {epoch:03d}] avg_loss={avg_loss:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.6f} | "
              f"time={elapsed:.1f}s")

        # ── 验证 ──
        if epoch % 5 == 0:
            val_loss = validate(model, val_loader, anchors, device)
            print(f"  >> Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(),
                           os.path.join(cfg.CKPT_DIR, "best.pth"))
                print(f"  >> 保存最优模型 (val_loss={val_loss:.4f})")

        # ── 定期保存 ──
        if epoch % cfg.SAVE_FREQ == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, os.path.join(cfg.CKPT_DIR, f"epoch_{epoch:03d}.pth"))


def validate(model, loader, anchors, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            img_l = batch["img_l"].to(device)
            img_r = batch["img_r"].to(device)
            bev   = batch_points_to_bev(batch["points"], device)
            gt_boxes = [b.to(device) for b in batch["gt_boxes"]]
            gt_cls   = [c.to(device) for c in batch["gt_cls"]]

            _, cls_pred, reg_pred = model(img_l, img_r, bev)
            loss, _, _ = compute_loss(
                cls_pred, reg_pred, gt_boxes, gt_cls, anchors)
            total += float(loss)
    return total / len(loader)


if __name__ == "__main__":
    train()