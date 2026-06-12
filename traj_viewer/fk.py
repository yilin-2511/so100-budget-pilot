"""SO-ARM100 正运动学 — 纯 NumPy 实现。

从 so100_fk.py 精简，无需 ikpy 或 MuJoCo。
链式乘 4×4 齐次矩阵计算各关节 3D 坐标。
"""

import numpy as np
import math

# 关节参数：translation (m) + rotation (RPY rad) + axis
# 来源：MuJoCo 模型 body position/quaternion → 已验证 FK 误差 0mm
JOINT_PARAMS = [
    {"t": [0.0, -0.0452, 0.0165],  "rpy": [1.5708, 0.0, 0.0],      "axis": [0, 1, 0]},
    {"t": [0.0,  0.1025, 0.0306],  "rpy": [1.5708, 0.0, 0.0],      "axis": [1, 0, 0]},
    {"t": [0.0,  0.11257, 0.028],  "rpy": [-1.5708, 0.0, 0.0],     "axis": [1, 0, 0]},
    {"t": [0.0,  0.0052, 0.1349],  "rpy": [-1.5708, 0.0, 0.0],     "axis": [1, 0, 0]},
    {"t": [0.0, -0.0601, 0.0],     "rpy": [0.0,  1.5708, 0.0],     "axis": [0, 1, 0]},
    {"t": [-0.0202, -0.0244, 0.0], "rpy": [-3.1416, -0.0, -3.1416], "axis": [0, 0, 1]},
]

LINK_NAMES = ["Base", "Rotation", "Pitch", "Elbow",
              "Wrist_Pitch", "Wrist_Roll", "Gripper"]


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def _axis_rotation(axis, angle):
    """绕指定轴旋转 angle 弧度。"""
    if axis[0] == 1:
        return _rot_x(angle)
    elif axis[1] == 1:
        return _rot_y(angle)
    elif axis[2] == 1:
        return _rot_z(angle)
    return np.eye(3)

def _rpy_to_matrix(rpy):
    """RPY 欧拉角 → 3×3 旋转矩阵（ZYX 顺序）。"""
    return _rot_z(rpy[2]) @ _rot_y(rpy[1]) @ _rot_x(rpy[0])


def _joint_transform(p, angle):
    """计算单个关节的 4×4 齐次变换矩阵。"""
    T = np.eye(4)
    T[:3, :3] = _rpy_to_matrix(p["rpy"])
    T[:3, 3] = p["t"]
    R = np.eye(4)
    R[:3, :3] = _axis_rotation(p["axis"], angle)
    return T @ R


def compute_joint_positions(q):
    """给定 5 个 arm 关节角 (+ 1 jaw)，返回 7 个关节世界坐标。

    Args:
        q: array-like, length 6 — [Rotation, Pitch, Elbow,
           Wrist_Pitch, Wrist_Roll, Jaw]

    Returns:
        (7, 3) ndarray — 基座 + 6 个关节末端的世界坐标 (m)
    """
    positions = np.zeros((7, 3))
    positions[0] = [0.0, 0.0, 0.0]  # 基座（原点）

    T = np.eye(4)
    for i in range(6):
        T = T @ _joint_transform(JOINT_PARAMS[i], q[i])
        positions[i + 1] = T[:3, 3]

    return positions


def compute_ee_pos(q):
    """给定 6 关节角，返回末端 (Fixed_Jaw) 世界坐标。

    仅用前 5 个 arm 关节，不含 Jaw。匹配 demo_cam.py 的 ee_body_id。

    Args:
        q: array-like, length 6

    Returns:
        (3,) ndarray — Fixed_Jaw 位置 (m)
    """
    return compute_joint_positions(q)[5]  # Wrist_Roll 输出 = Fixed_Jaw
