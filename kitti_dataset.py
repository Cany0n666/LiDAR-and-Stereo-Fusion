"""
KITTI数据集加载器 —— 与 train.py / config.py 对齐版
坐标系修复：GT Box 从 Camera 坐标系正确转换到 LiDAR 坐标系
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, List, Tuple


class KITTIDataset(Dataset):
    """KITTI 3D目标检测数据集"""

    CLASSES = ["Car", "Pedestrian", "Cyclist"]

    def __init__(self, root: str, split: str):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from config import cfg

        self.root   = root
        self.split  = split
        self.cfg    = cfg

        self.img_h  = cfg.IMG_H
        self.img_w  = cfg.IMG_W
        self.class_to_idx = {c: i for i, c in enumerate(self.CLASSES)}

        self.pc_range = [
            cfg.PC_X_MIN, cfg.PC_Y_MIN, cfg.PC_Z_MIN,
            cfg.PC_X_MAX, cfg.PC_Y_MAX, cfg.PC_Z_MAX
        ]

        self.image_left_dir  = os.path.join(root, split, 'image_2')
        self.image_right_dir = os.path.join(root, split, 'image_3')
        self.velodyne_dir    = os.path.join(root, split, 'velodyne')
        self.calib_dir       = os.path.join(root, split, 'calib')
        self.label_dir       = os.path.join(root, split, 'label_2')

        self.samples = self._load_samples()
        print(f"[KITTIDataset] split={split}, samples={len(self.samples)}")

    def _load_samples(self) -> List[str]:
        samples = []
        if os.path.exists(self.image_left_dir):
            for f in sorted(os.listdir(self.image_left_dir)):
                if f.endswith('.png'):
                    sid = f.replace('.png', '')
                    r_path = os.path.join(self.image_right_dir, f)
                    if os.path.exists(r_path):
                        samples.append(sid)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sid = self.samples[idx]

        img_l  = self._load_image(os.path.join(self.image_left_dir,  f'{sid}.png'))
        img_r  = self._load_image(os.path.join(self.image_right_dir, f'{sid}.png'))
        points = self._load_point_cloud(os.path.join(self.velodyne_dir, f'{sid}.bin'))
        calib  = self._load_calib(os.path.join(self.calib_dir, f'{sid}.txt'))
        gt_boxes, gt_cls = self._load_labels(
            os.path.join(self.label_dir, f'{sid}.txt'), calib)

        return {
            'sample_id': sid,
            'img_l'    : img_l,
            'img_r'    : img_r,
            'points'   : points,
            'calib'    : calib,
            'gt_boxes' : gt_boxes,
            'gt_cls'   : gt_cls,
        }

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert('RGB')
        img = img.resize((self.img_w, self.img_h), Image.Resampling.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (img - mean) / std

    def _load_point_cloud(self, path: str) -> torch.Tensor:
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        xmin, ymin, zmin, xmax, ymax, zmax = self.pc_range
        mask = (
            (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) &
            (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax) &
            (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
        )
        return torch.from_numpy(pts[mask])

    def _load_calib(self, path: str) -> Dict:
        calib = {}
        with open(path) as f:
            for line in f:
                if ':' in line:
                    k, v = line.split(':', 1)
                    calib[k.strip()] = torch.tensor(
                        [float(x) for x in v.strip().split()],
                        dtype=torch.float32)
        for key, shape in [('P2',             (3, 4)),
                            ('P3',             (3, 4)),
                            ('R0_rect',        (3, 3)),
                            ('Tr_velo_to_cam', (3, 4))]:
            if key in calib:
                calib[key] = calib[key].reshape(shape)
        return calib

    @staticmethod
    def _cam_to_lidar(pts_cam: np.ndarray,
                      R0: np.ndarray,
                      Tr: np.ndarray) -> np.ndarray:
        R0_4x4 = np.eye(4, dtype=np.float64)
        R0_4x4[:3, :3] = R0.astype(np.float64)

        Tr_4x4 = np.eye(4, dtype=np.float64)
        Tr_4x4[:3, :] = Tr.astype(np.float64)

        T = R0_4x4 @ Tr_4x4
        T_inv = np.linalg.inv(T)

        N = pts_cam.shape[0]
        pts_hom = np.concatenate(
            [pts_cam.astype(np.float64),
             np.ones((N, 1), dtype=np.float64)], axis=1)

        pts_lidar = (T_inv @ pts_hom.T).T
        return pts_lidar[:, :3]

    @staticmethod
    def _ry_to_lidar_heading(ry: float) -> float:
        heading = -(ry + np.pi / 2)
        while heading > np.pi:
            heading -= 2 * np.pi
        while heading < -np.pi:
            heading += 2 * np.pi
        return heading

    def _load_labels(self, path: str, calib: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        boxes, cls_ids = [], []

        if not os.path.exists(path):
            return torch.zeros((0, 7)), torch.zeros((0,), dtype=torch.long)

        R0 = calib['R0_rect'].numpy()
        Tr = calib['Tr_velo_to_cam'].numpy()
        xmin, ymin, zmin, xmax, ymax, zmax = self.pc_range

        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue
                name = parts[0]
                if name not in self.class_to_idx:
                    continue

                h   = float(parts[8])
                w   = float(parts[9])
                l   = float(parts[10])
                x_c = float(parts[11])
                y_c = float(parts[12])
                z_c = float(parts[13])
                ry  = float(parts[14])

                y_c_center = y_c - h / 2.0

                pts_cam   = np.array([[x_c, y_c_center, z_c]])
                pts_lidar = self._cam_to_lidar(pts_cam, R0, Tr)
                x_l, y_l, z_l = pts_lidar[0]

                if not (xmin <= x_l <= xmax and
                        ymin <= y_l <= ymax and
                        zmin <= z_l <= zmax):
                    continue

                heading = self._ry_to_lidar_heading(ry)
                boxes.append([x_l, y_l, z_l, w, l, h, heading])
                cls_ids.append(self.class_to_idx[name])

        if not boxes:
            return torch.zeros((0, 7)), torch.zeros((0,), dtype=torch.long)

        return (torch.tensor(boxes,   dtype=torch.float32),
                torch.tensor(cls_ids, dtype=torch.long))


def collate_fn(batch: List[Dict]) -> Dict:
    return {
        'sample_ids': [b['sample_id'] for b in batch],
        'img_l'     : torch.stack([b['img_l']  for b in batch]),
        'img_r'     : torch.stack([b['img_r']  for b in batch]),
        'points'    : [b['points']   for b in batch],
        'calibs'    : [b['calib']    for b in batch],
        'gt_boxes'  : [b['gt_boxes'] for b in batch],
        'gt_cls'    : [b['gt_cls']   for b in batch],
    }
