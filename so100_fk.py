"""SO100 Forward Kinematics — pure NumPy, for verification and orientation extraction.

Joint parameters extracted from MuJoCo model body quaternions (NOT URDF).
FK matches MuJoCo data.xpos at 0mm error.
"""

import numpy as np
import math

# Joint parameters from MuJoCo body_pos + body_quat→RPY (verified 0mm FK error)
JOINT_PARAMS = [
    {"translation": [0, -0.0452, 0.0165], "rotation": [ 1.5708, 0, 0],      "axis": [0, 1, 0]},
    {"translation": [0,  0.1025, 0.0306],  "rotation": [ 1.5708, 0, 0],      "axis": [1, 0, 0]},
    {"translation": [0,  0.11257, 0.028],   "rotation": [-1.5708, 0, 0],      "axis": [1, 0, 0]},
    {"translation": [0,  0.0052, 0.1349],   "rotation": [-1.5708, 0, 0],      "axis": [1, 0, 0]},
    {"translation": [0, -0.0601, 0],         "rotation": [0,  1.5708, 0],      "axis": [0, 1, 0]},
    {"translation": [-0.0202, -0.0244, 0],   "rotation": [-3.1416, -0, -3.1416], "axis": [0, 0, 1]},
]

JOINT_LIMITS = [
    [-1.92, 1.92], [-3.32, 0.174], [-0.174, 3.14],
    [-1.66, 1.66], [-2.79, 2.79], [-0.174, 1.75],
]


class So100FK:
    """Pure-numpy FK for SO100 — used by demo to extract current orientation."""

    def __init__(self):
        self.joint_params = JOINT_PARAMS
        self.joint_limits = JOINT_LIMITS

    @staticmethod
    def _rot_x(a): c, s = math.cos(a), math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    @staticmethod
    def _rot_y(a): c, s = math.cos(a), math.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    @staticmethod
    def _rot_z(a): c, s = math.cos(a), math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

    @classmethod
    def _rpy_to_matrix(cls, rpy):
        return cls._rot_z(rpy[2]) @ cls._rot_y(rpy[1]) @ cls._rot_x(rpy[0])

    @classmethod
    def _axis_rotation(cls, axis, angle):
        if axis[0] == 1:   return cls._rot_x(angle)
        elif axis[1] == 1: return cls._rot_y(angle)
        elif axis[2] == 1: return cls._rot_z(angle)
        else:               return np.eye(3)

    @classmethod
    def _joint_to_4x4(cls, t, rpy, axis, angle):
        T = np.eye(4)
        T[:3, :3] = cls._rpy_to_matrix(rpy)
        T[:3, 3] = t
        R = np.eye(4)
        R[:3, :3] = cls._axis_rotation(axis, angle)
        return T @ R

    def forward_kinematics_5dof_matrix(self, q):
        """5-DOF FK → Fixed_Jaw 4×4 transform. Used by demo for orientation lock."""
        T = np.eye(4)
        for i in range(5):
            p = self.joint_params[i]
            T = T @ self._joint_to_4x4(p["translation"], p["rotation"], p["axis"], q[i])
        return T.copy()

    def forward_kinematics_5dof(self, q):
        """5-DOF FK → Fixed_Jaw position [x,y,z]."""
        return self.forward_kinematics_5dof_matrix(q)[:3, 3].copy()
