"""Replay recorded trajectories with picker UI.

Scans recordings/ folder, shows list of saved trajectories.
Click one → replay in MuJoCo with physics (cube interaction).
"""
import glob
import os
import sys
import time
import tkinter as tk

import numpy as np
import mujoco
import mujoco.viewer

# ---------------------------------------------------------------------------
# MuJoCo scene
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, "model", "so100_pick_place.xml")
TRAJ_DIR = os.path.join(HERE, "recordings")

# ---------------------------------------------------------------------------
# Scan trajectories
# ---------------------------------------------------------------------------
def scan_trajs():
    files = sorted(glob.glob(os.path.join(TRAJ_DIR, "*.npz")),
                   key=os.path.getmtime, reverse=True)
    trajs = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            d = np.load(f, allow_pickle=True)
            frames = len(d["trajectory"])
            joints = list(d.get("joint_names", []))
            trajs.append({"name": name, "path": f, "frames": frames, "joints": joints})
        except Exception:
            trajs.append({"name": name, "path": f, "frames": "?", "joints": []})
    return trajs

# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def _reset_scene(model, data):
    """Reset arm to home, cube to its initial XML position."""
    mujoco.mj_resetData(model, data)
    cube_qpos = data.qpos[6:13].copy()
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[6:13] = cube_qpos
    mujoco.mj_forward(model, data)

def replay_traj(traj_info, speed):
    """speed: "1x" (timestamp), "2x" (double speed), "max" (frame-by-frame)."""
    loaded = np.load(traj_info["path"], allow_pickle=True)
    traj = loaded["trajectory"]

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    model.dof_damping[:] = 5.0
    _reset_scene(model, data)

    # Let cube settle on table before starting replay (hold arm ctrl)
    for _ in range(300):
        data.ctrl[:6] = [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]
        mujoco.mj_step(model, data)

    frame_idx = [0]
    t0_traj = traj[0, 0]
    t0_real = time.perf_counter()
    speed_label = {"1x": "1x", "2x": "2x", "max": "MAX"}[speed]
    print(f"[REPLAY] {traj_info['name']} ({len(traj)} frames, {traj[-1,0]:.1f}s) @ {speed_label}")

    with mujoco.viewer.launch_passive(model, data,
                                       show_left_ui=False,
                                       show_right_ui=False) as viewer:
        while viewer.is_running():
            if frame_idx[0] < len(traj):
                if speed != "max":
                    # Timestamp-based wait
                    target_t = (traj[frame_idx[0], 0] - t0_traj) / (1.0 if speed == "1x" else 2.0)
                    while time.perf_counter() - t0_real < target_t:
                        if not viewer.is_running():
                            break
                        time.sleep(0.001)
                    if not viewer.is_running():
                        break
                data.ctrl[:6] = traj[frame_idx[0], 7:13]
                frame_idx[0] += 1
            else:
                # Loop — reset scene and settle cube
                frame_idx[0] = 0
                t0_real = time.perf_counter()
                _reset_scene(model, data)
                for _ in range(300):
                    data.ctrl[:6] = [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]
                    mujoco.mj_step(model, data)
            mujoco.mj_step(model, data)
            viewer.sync()

    print("[REPLAY] Done. Back to picker.")

# ---------------------------------------------------------------------------
# tkinter picker
# ---------------------------------------------------------------------------
trajs = scan_trajs()
if not trajs:
    print("No trajectories found in", TRAJ_DIR)
    print("Record one first with demo_pick_place.py")
    sys.exit(1)

root = tk.Tk()
root.title("Trajectory Replay")
root.geometry("550x400")
root.attributes("-topmost", True)

tk.Label(root, text="Select a trajectory to replay",
         font=("Arial", 12, "bold")).pack(pady=(8, 4))
tk.Label(root, text=f"{len(trajs)} trajectories in {TRAJ_DIR}",
         font=("Arial", 8), fg="gray").pack()

# Listbox with scrollbar
frame = tk.Frame(root)
frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

scrollbar = tk.Scrollbar(frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

lb = tk.Listbox(frame, font=("Consolas", 10), yscrollcommand=scrollbar.set)
for t in trajs:
    joints_str = " | ".join(t["joints"][:6]) if t["joints"] else ""
    lb.insert(tk.END, f"  {t['name']:30s}  {t['frames']:>5d} frames  {joints_str}")
lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.config(command=lb.yview)

# Speed selector
speed_var = tk.StringVar(value="1x")
sf = tk.Frame(root)
sf.pack(pady=(4, 0))
tk.Label(sf, text="Speed:", font=("Arial", 10)).pack(side=tk.LEFT, padx=(0, 8))
for s in ["1x", "2x", "MAX"]:
    tk.Radiobutton(sf, text=s, variable=speed_var, value=s.lower() if s != "MAX" else "max",
                   font=("Arial", 10)).pack(side=tk.LEFT, padx=4)

# Buttons
bf = tk.Frame(root)
bf.pack(pady=6)

def on_replay():
    sel = lb.curselection()
    if not sel:
        lbl_status.config(text="Select a trajectory first", fg="red")
        return
    t = trajs[sel[0]]
    root.withdraw()
    replay_traj(t, speed_var.get())
    root.deiconify()

def on_refresh():
    global trajs
    trajs = scan_trajs()
    lb.delete(0, tk.END)
    for t in trajs:
        joints_str = " | ".join(t["joints"][:6]) if t["joints"] else ""
        lb.insert(tk.END, f"  {t['name']:30s}  {t['frames']:>5d} frames  {joints_str}")
    lbl_status.config(text=f"Refreshed — {len(trajs)} trajectories", fg="gray")

tk.Button(bf, text="▶ Replay", font=("Arial", 11, "bold"),
          bg="#4CAF50", fg="white", width=12, command=on_replay).pack(side=tk.LEFT, padx=5)
tk.Button(bf, text="↻ Refresh", font=("Arial", 11),
          width=12, command=on_refresh).pack(side=tk.LEFT, padx=5)
tk.Button(bf, text="✕ Quit", font=("Arial", 11),
          width=12, command=root.destroy).pack(side=tk.LEFT, padx=5)

lbl_status = tk.Label(root, text="Select a trajectory and click Replay",
                       font=("Arial", 9), fg="gray")
lbl_status.pack(pady=(0, 5))

print(f"[OK] Found {len(trajs)} trajectories")
root.mainloop()
