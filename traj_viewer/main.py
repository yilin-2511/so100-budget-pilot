"""机械臂轨迹可视化分析工具 — tkinter GUI 入口。

用法:
    python main.py                  # GUI 模式
    python main.py <file.npz>       # 命令行快速查看统计
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox

import matplotlib
matplotlib.use("TkAgg")

from load_traj import scan_recordings, load_traj
from stats_traj import traj_stats
from plot_traj import (
    plot_joint_angles,
    plot_ee_trajectory,
    plot_arm_animation,
    plot_multi_trajectories,
)

REC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "recordings")


# ── 命令行模式 ────────────────────────────────────────────
def _cli_mode(filepath):
    """python main.py <traj.npz>"""
    t = load_traj(filepath)
    print(traj_stats(t))
    print()
    print("生成图表...")
    plot_joint_angles(t)
    plot_ee_trajectory(t)
    plot_arm_animation(t)


# ── GUI ───────────────────────────────────────────────────
class TrajViewerApp:
    def __init__(self, root):
        self.root = root
        root.title("机械臂轨迹可视化分析工具")
        root.geometry("600x480")
        root.resizable(True, True)

        # ── 左侧：轨迹列表 ──
        left = tk.Frame(root, width=280)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=(10, 5), pady=10)
        left.pack_propagate(False)

        tk.Label(left, text="轨迹列表（Ctrl/Shift 多选）",
                 font=("Microsoft YaHei", 10, "bold")).pack(anchor="w")

        list_frame = tk.Frame(left)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=4)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(
            list_frame, selectmode=tk.EXTENDED,
            font=("Consolas", 10), yscrollcommand=scrollbar.set,
            exportselection=False,
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)

        btn_refresh = tk.Button(left, text="刷新列表", command=self._refresh_list,
                                font=("Microsoft YaHei", 9))
        btn_refresh.pack(pady=(4, 0))

        # ── 右侧：功能按钮 ──
        right = tk.Frame(root)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)

        tk.Label(right, text="功能",
                 font=("Microsoft YaHei", 10, "bold")).pack(anchor="w", pady=(0, 8))

        btn_style = {"width": 22, "height": 2, "font": ("Microsoft YaHei", 10)}

        tk.Button(right, text="关节角度曲线", command=self._show_joints,
                  bg="#E3F2FD", **btn_style).pack(pady=3)
        tk.Button(right, text="末端 3D 轨迹", command=self._show_ee,
                  bg="#E8F5E9", **btn_style).pack(pady=3)
        # 动画速度选择
        anim_frame = tk.Frame(right)
        anim_frame.pack(pady=3)
        tk.Button(anim_frame, text="3D 机械臂动画（1×）", command=lambda: self._show_animation(1.0),
                  bg="#FFF3E0", font=("Microsoft YaHei", 10), width=11, height=2).pack(side=tk.LEFT, padx=2)
        tk.Button(anim_frame, text="3×", command=lambda: self._show_animation(3.0),
                  bg="#FFE0B2", font=("Microsoft YaHei", 10), width=4, height=2).pack(side=tk.LEFT, padx=2)
        tk.Button(anim_frame, text="5×", command=lambda: self._show_animation(5.0),
                  bg="#FFCC80", font=("Microsoft YaHei", 10), width=4, height=2).pack(side=tk.LEFT, padx=2)
        tk.Button(right, text="多轨迹对比（≥2 条）",
                  command=self._show_multi,
                  bg="#F3E5F5", **btn_style).pack(pady=3)
        tk.Button(right, text="轨迹统计", command=self._show_stats,
                  bg="#ECEFF1", **btn_style).pack(pady=3)

        # 提示
        tk.Label(right, text='\n选择 2 条以上轨迹后可使用「多轨迹对比」',
                 font=("Microsoft YaHei", 8), fg="#78909C").pack(pady=(10, 0))
        tk.Label(right, text="3D 动画键盘：空格暂停 ←→跳帧 +/-变速",
                 font=("Microsoft YaHei", 8), fg="#78909C").pack()

        # ── 初始加载 ──
        self._refresh_list()

    # ── 列表操作 ──
    def _refresh_list(self):
        self.listbox.delete(0, tk.END)
        files = scan_recordings(REC_DIR)
        for f in files:
            self.listbox.insert(tk.END, os.path.basename(f))
        self._filepaths = files
        if files:
            self.listbox.selection_set(0)

    def _get_selected(self):
        selected = self.listbox.curselection()
        if not selected:
            messagebox.showwarning("提示", "请先选择轨迹")
            return []
        return [self._filepaths[i] for i in selected]

    # ── 功能回调 ──
    def _show_joints(self):
        files = self._get_selected()
        if not files:
            return
        for f in files:
            plot_joint_angles(load_traj(f))

    def _show_ee(self):
        files = self._get_selected()
        if not files:
            return
        for f in files:
            plot_ee_trajectory(load_traj(f))

    def _show_animation(self, speed=1.0):
        files = self._get_selected()
        if not files:
            return
        t = load_traj(files[0])
        if len(files) > 1:
            messagebox.showinfo("提示", f"将播放第一条: {t['filename']}")
        plot_arm_animation(t, speed=speed)

    def _show_multi(self):
        files = self._get_selected()
        if len(files) < 2:
            messagebox.showwarning("提示", "请选择至少 2 条轨迹进行对比（Ctrl/Shift 多选）")
            return
        plot_multi_trajectories(files)

    def _show_stats(self):
        files = self._get_selected()
        if not files:
            return
        for f in files:
            t = load_traj(f)
            s = traj_stats(t)
            # 弹窗展示统计
            top = tk.Toplevel(self.root)
            top.title(f"轨迹统计 — {t['filename']}")
            top.geometry("480x420")
            txt = tk.Text(top, font=("Consolas", 10), wrap=tk.NONE,
                          bg="#FAFAFA", relief=tk.FLAT)
            txt.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            txt.insert("1.0", s)
            txt.config(state=tk.DISABLED)


# ── 入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        _cli_mode(sys.argv[1])
    else:
        root = tk.Tk()
        TrajViewerApp(root)
        root.mainloop()
