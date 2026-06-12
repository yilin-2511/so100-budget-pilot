"""轨迹文件加载与解析。

每条 .npz 轨迹为 (N, 20) 数组:
  [time, qpos(6), ctrl(6), ee_x, ee_y, ee_z, ee_qw, ee_qx, ee_qy, ee_qz]
"""

import os
import glob
import numpy as np

JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]


def load_traj(filepath):
    """加载单个 .npz 轨迹文件，返回 dict。"""
    data = np.load(filepath, allow_pickle=True)
    traj = data["trajectory"]
    names = list(data["joint_names"])

    return {
        "filepath": filepath,
        "filename": os.path.basename(filepath),
        "time": traj[:, 0],
        "qpos": traj[:, 1:7],
        "ctrl": traj[:, 7:13],
        "ee_pos": traj[:, 13:16],
        "ee_quat": traj[:, 16:20],
        "joint_names": names,
        "n_frames": traj.shape[0],
        "duration": float(traj[-1, 0]),
    }


def scan_recordings(directory):
    """扫描目录，返回排序后的 .npz 文件路径列表。"""
    pattern = os.path.join(directory, "*.npz")
    files = glob.glob(pattern)
    return sorted(files, key=os.path.getmtime, reverse=True)
