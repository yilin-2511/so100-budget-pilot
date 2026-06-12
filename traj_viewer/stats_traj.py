"""轨迹统计摘要 — 时长、关节范围、末端行程、FK 误差。"""

import numpy as np
from fk import compute_ee_pos


def traj_stats(traj_data):
    """返回多行统计字符串，覆盖核心指标。

    Args:
        traj_data: load_traj() 返回的 dict

    Returns:
        str — 格式化的统计摘要
    """
    time = traj_data["time"]
    qpos = traj_data["qpos"]
    ctrl = traj_data["ctrl"]
    ee_pos = traj_data["ee_pos"]
    joint_names = traj_data["joint_names"]
    filename = traj_data["filename"]
    n = traj_data["n_frames"]

    # ── 基本 ──
    duration = float(time[-1])
    fps = n / duration if duration > 0 else 0.0

    N_ARM = 5  # 只看 arm 关节，跳过 Jaw
    # ── 关节范围 ──
    q_span = qpos[:, :N_ARM].max(axis=0) - qpos[:, :N_ARM].min(axis=0)

    # ── 控制误差 (|ctrl - qpos|, 仅 arm 关节, 度) ──
    ctrl_err_deg = np.degrees(np.abs(ctrl[:, :N_ARM] - qpos[:, :N_ARM]))

    # ── 末端 ──
    ee_total = _total_travel(ee_pos)

    # ── FK 误差 ──
    ee_fk = np.array([compute_ee_pos(qpos[k]) for k in range(0, n, max(1, n // 2000))])
    ee_sub = ee_pos[::max(1, n // 2000)]
    fk_err = np.linalg.norm(ee_sub - ee_fk, axis=1)
    mean_fk_err = float(np.mean(fk_err))

    # ── 组装 ──
    lines = [
        f"轨迹: {filename}",
        f"{'='*50}",
        f"帧数: {n}    时长: {duration:.2f}s    平均帧率: {fps:.1f} Hz",
        f"",
        f"── Arm 关节运动范围 (不含 Jaw) ──",
    ]
    for i, name in enumerate(joint_names[:N_ARM]):
        lines.append(
            f"  {name:14s}  "
            f"min={np.degrees(qpos[:, i].min()):+7.1f}°  "
            f"max={np.degrees(qpos[:, i].max()):+7.1f}°  "
            f"span={np.degrees(q_span[i]):+7.1f}°"
        )
    lines.append(f"")
    lines.append(f"── 末端执行器 ──")
    lines.append(f"  总行程:     {ee_total*1000:.0f} mm")
    lines.append(f"  平均速度:   {ee_total/duration*1000:.0f} mm/s")
    lines.append(f"")
    lines.append(f"── 控制追踪 (|ctrl - qpos|, 度) ──")
    for i, name in enumerate(joint_names[:N_ARM]):
        lines.append(
            f"  {name:14s}  "
            f"均值={ctrl_err_deg[:, i].mean():.2f}°  "
            f"P95={np.percentile(ctrl_err_deg[:, i], 95):.2f}°  "
            f"最大={ctrl_err_deg[:, i].max():.2f}°"
        )
    lines.append(f"")
    lines.append(f"── FK 精度 ──")
    lines.append(f"  FK 推算误差均值 (|MuJoCo - FK|): {mean_fk_err*1000:.2f} mm")

    return "\n".join(lines)


def _total_travel(points):
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
