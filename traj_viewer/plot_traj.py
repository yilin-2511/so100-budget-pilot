"""轨迹可视化 — matplotlib 图表生成。

四个功能：关节角曲线、末端 3D 轨迹、机械臂 3D 动画、多轨迹对比。
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # 兼容 tkinter 主循环

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from load_traj import load_traj
from fk import compute_joint_positions, LINK_NAMES

# ── 中文字体 ──────────────────────────────────────────────
_FONT = None
for _name in ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "sans-serif"]:
    try:
        matplotlib.font_manager.findfont(_name, fallback_to_default=False)
        _FONT = _name
        break
    except Exception:
        continue
if _FONT:
    plt.rcParams["font.sans-serif"] = [_FONT, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


# ── 轨迹 3D 坐标单位格式化 ───────────────────────────────
def _meter_formatter(x, pos):
    return f"{x*1000:.0f}"  # 米 → 毫米


def _deg_formatter(x, pos):
    return f"{np.degrees(x):.0f}°"  # 弧度 → 度


# ═══════════════════════════════════════════════════════════
# 1. 关节角度曲线
# ═══════════════════════════════════════════════════════════
def plot_joint_angles(traj, title=None):
    """绘制 5 个 arm 关节角度随时间变化曲线（不含 Jaw）。

    蓝实线 = qpos（物理实际）, 橙虚线 = ctrl（控制目标）
    """
    N_ARM = 5  # 只看 arm 关节，跳过 Jaw
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    fig.suptitle(title or f"关节角度 — {traj['filename']}", fontsize=13)

    joint_names = traj["joint_names"][:N_ARM]
    time = traj["time"]
    qpos = traj["qpos"][:, :N_ARM]
    ctrl = traj["ctrl"][:, :N_ARM]

    for i, ax in enumerate(axes.flat):
        if i < N_ARM:
            ax.plot(time, qpos[:, i], "C0", linewidth=1.0, label="qpos (物理)")
            ax.plot(time, ctrl[:, i], "C1--", linewidth=0.8, label="ctrl (目标)")
            ax.set_ylabel("角度")
            ax.set_title(joint_names[i])
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)
        else:
            ax.axis("off")  # 第 6 个格子空着

    axes[-1, 0].set_xlabel("时间 (s)")
    axes[-1, 1].set_xlabel("时间 (s)")
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════
# 2. 末端 3D 轨迹对比
# ═══════════════════════════════════════════════════════════
def plot_ee_trajectory(traj):
    """3D 图：MuJoCo ee_pos vs 纯 numpy FK 推算 ee_pos。

    蓝线 = MuJoCo 真值（来自 .npz）
    红虚线 = FK 推算值（从 qpos 用纯 numpy 计算）
    两条线重合 → FK 参数正确
    """
    from fk import compute_ee_pos

    # FK 推算（每 10 帧取 1 帧，轨迹数据已高密度冗余）
    step = max(1, traj["n_frames"] // 5000)
    ee_mj = traj["ee_pos"][::step]
    ee_fk = np.array([compute_ee_pos(traj["qpos"][k]) for k in range(0, traj["n_frames"], step)])

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ee_mj[:, 0], ee_mj[:, 1], ee_mj[:, 2],
            "C0", linewidth=1.2, label="MuJoCo ee_pos")
    ax.plot(ee_fk[:, 0], ee_fk[:, 1], ee_fk[:, 2],
            "C3--", linewidth=1.0, label="FK 推算 ee_pos")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"末端 3D 轨迹对比 — {traj['filename']}")

    # 等比例坐标轴，避免拉伸失真
    ax.set_box_aspect([1.0, 1.0, 1.0])

    # 计算 FK 误差（用全量数据保证统计准确）
    err = np.linalg.norm(ee_mj - ee_fk, axis=1)
    ax.legend()
    ax.text2D(0.02, 0.98,
              f"FK 平均误差: {np.mean(err)*1000:.2f} mm",
              transform=ax.transAxes, fontsize=10,
              verticalalignment="top")
    stats_text = (
        f"MuJoCo 总行程: {_total_travel(ee_mj)*1000:.0f} mm\n"
        f"FK 推算总行程: {_total_travel(ee_fk)*1000:.0f} mm"
    )
    ax.text2D(0.02, 0.88, stats_text, transform=ax.transAxes,
              fontsize=9, verticalalignment="top")
    plt.show()
    return np.mean(err) * 1000  # mm


# ═══════════════════════════════════════════════════════════
# 3. 机械臂 3D 动画
# ═══════════════════════════════════════════════════════════
def plot_arm_animation(traj, speed=1.0):
    """机械臂 3D stick figure 动画。

    显示 7 个关节连杆，末端轨迹残影，可旋转/缩放视角。
    """
    # 预计算所有帧的关节位置
    all_positions = np.array([
        compute_joint_positions(traj["qpos"][k]) for k in range(traj["n_frames"])
    ])  # (N, 7, 3)

    # 预计算 FK 末端位置用于轨迹残影
    from fk import compute_ee_pos
    ee_trail = np.array([compute_ee_pos(traj["qpos"][k]) for k in range(traj["n_frames"])])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 轴标签
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"机械臂 3D 动画 — {traj['filename']}")

    # 固定视角范围 + 等比例
    ax.set_xlim(-0.15, 0.30)
    ax.set_ylim(-0.55, 0.10)
    ax.set_zlim(0.0, 0.40)
    ax.set_box_aspect([1.0, 1.0, 1.0])

    # ── 交互控制状态 ──
    duration = traj["duration"]
    state = {
        "play_time": 0.0,    # 当前动画时间 (s)，按 recording 时间轴
        "playing": True,
        "speed": speed,
    }

    def draw_frame_at_time(anim_t):
        """根据动画时间找到最近帧并绘制。"""
        k = min(np.searchsorted(traj["time"], anim_t), traj["n_frames"] - 1)
        ax.clear()
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"机械臂 3D 动画 — {traj['filename']}  "
                     f"t={anim_t:.1f}s / {duration:.1f}s  "
                     f"{state['speed']:.0f}×")
        ax.set_xlim(-0.15, 0.30)
        ax.set_ylim(-0.55, 0.10)
        ax.set_zlim(0.0, 0.40)
        ax.set_box_aspect([1.0, 1.0, 1.0])

        pts = all_positions[k]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                "o-", color="#37474F", linewidth=2.5, markersize=5)
        for j in range(7):
            ax.text(pts[j, 0] + 0.005, pts[j, 1] + 0.005, pts[j, 2],
                    LINK_NAMES[j], fontsize=7)
        # 末端轨迹残影（最近 1 秒的路径）
        trail_mask = (traj["time"] >= max(0, anim_t - 1.0)) & (traj["time"] <= anim_t)
        if np.any(trail_mask):
            trail_pts = ee_trail[trail_mask]
            ax.plot(trail_pts[:, 0], trail_pts[:, 1], trail_pts[:, 2],
                    color="#90CAF9", linewidth=0.6, alpha=0.7)
        return k

    def on_key(event):
        if event.key == " ":
            state["playing"] = not state["playing"]
        elif event.key == "right":
            state["play_time"] = min(state["play_time"] + 1.0, duration)
            draw_frame_at_time(state["play_time"])
            fig.canvas.draw_idle()
        elif event.key == "left":
            state["play_time"] = max(state["play_time"] - 1.0, 0)
            draw_frame_at_time(state["play_time"])
            fig.canvas.draw_idle()
        elif event.key in ("+", "="):
            state["speed"] = min(state["speed"] * 1.5, 20.0)
        elif event.key == "-":
            state["speed"] = max(state["speed"] / 1.5, 0.5)

    fig.canvas.mpl_connect("key_press_event", on_key)

    draw_frame_at_time(0)
    fig.text(0.5, 0.02,
             "空格=暂停/播放   ←→=快退1s/快进1s   +/-=变速  鼠标=旋转/缩放",
             ha="center", fontsize=9, color="#616161")
    plt.tight_layout()
    plt.ion()
    plt.show()

    # ── 动画主循环（基于挂钟时间） ──
    import time
    real_start = time.perf_counter()
    sim_offset = state["play_time"]
    last_frame = -1
    while plt.fignum_exists(fig.number):
        fig.canvas.flush_events()
        if state["playing"]:
            now = time.perf_counter()
            state["play_time"] = sim_offset + (now - real_start) * state["speed"]
            if state["play_time"] >= duration:
                state["play_time"] = 0.0
                real_start = now
                sim_offset = 0.0
            k = draw_frame_at_time(state["play_time"])
            if k != last_frame:
                fig.canvas.draw_idle()
                last_frame = k
        else:
            sim_offset = state["play_time"]
            real_start = time.perf_counter()
            last_frame = -1
        plt.pause(0.016)  # ~60 FPS timer
    plt.ioff()


# ═══════════════════════════════════════════════════════════
# 4. 多轨迹对比
# ═══════════════════════════════════════════════════════════
def plot_multi_trajectories(filepaths):
    """多条轨迹的关节角叠合对比。

    每条轨迹用不同颜色，重合度高 → 操作一致性好。
    """
    N_ARM = 5
    trajs = [load_traj(f) for f in filepaths]
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    fig.suptitle("多轨迹关节角对比", fontsize=13)

    joint_names = trajs[0]["joint_names"][:N_ARM]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(trajs), 10)))

    for i, ax in enumerate(axes.flat):
        if i < N_ARM:
            for j, t in enumerate(trajs):
                label = t["filename"].replace(".npz", "")
                ax.plot(t["time"], t["qpos"][:, i],
                        color=colors[j], linewidth=0.8, label=label)
            ax.set_ylabel("角度")
            ax.set_title(joint_names[i])
            ax.legend(fontsize=6, loc="upper right")
            ax.grid(True, alpha=0.3)
        else:
            ax.axis("off")

    axes[-1, 0].set_xlabel("时间 (s)")
    axes[-1, 1].set_xlabel("时间 (s)")
    plt.tight_layout()
    plt.show()


# ── 辅助 ──────────────────────────────────────────────────
def _total_travel(points):
    """计算 3D 轨迹总行程 (m)。"""
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
