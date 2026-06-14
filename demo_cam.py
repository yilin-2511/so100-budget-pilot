"""SO100 Pick & Place — Keyboard Teleop.

Same control engine, redesigned UI:
  - Dark theme (matching demo_lerobot_record style)
  - Hybrid Intuitive Frame IK, rate-limited, time interpolation
  - .npz recording to recordings/
"""
import os
import sys
import time
import tkinter as tk
from tkinter import simpledialog

import numpy as np
import mujoco
import mujoco.viewer
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from so100_ik import So100IK

# ---------------------------------------------------------------------------
# MuJoCo scene
# ---------------------------------------------------------------------------
XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    "model", "so100_pick_place.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
model.dof_damping[:] = 3.0

j_qpos_ids = [model.jnt_qposadr[j] for j in range(6)]
ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")

mujoco.mj_resetData(model, data)
cube_qpos = data.qpos[6:13].copy()
mujoco.mj_resetDataKeyframe(model, data, 0)
data.qpos[6:13] = cube_qpos
mujoco.mj_forward(model, data)
for _ in range(300):
    data.ctrl[:6] = [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]
    mujoco.mj_step(model, data)

# ---------------------------------------------------------------------------
# IK
# ---------------------------------------------------------------------------
ik = So100IK()

POS_SPEED = 0.10
IK_INTERVAL = 0.05
JAW_SPEED = 1.0

target_pos = np.zeros(3)
target_orient = np.eye(3)

_home_q5 = np.array([data.qpos[j_qpos_ids[i]] for i in range(5)])
_home_T = ik.fk.forward_kinematics_5dof_matrix(_home_q5)
target_orient = _home_T[:3, :3].copy()
print(f"[LOCK] Gripper Z = [{target_orient[0,2]:+.4f}, {target_orient[1,2]:+.4f}, {target_orient[2,2]:+.4f}]")

q_ik_start = np.zeros(5)
q_ik_target = np.zeros(5)
ik_progress = 1.0
last_ik_time = 0.0

# ---------------------------------------------------------------------------
# Recording (.npz)
# ---------------------------------------------------------------------------
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
TRAJ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
os.makedirs(TRAJ_DIR, exist_ok=True)

recording = False
record_start_time = 0
record_name = ""
frames = []

def save_traj():
    arr = np.array(frames, dtype=np.float32)
    fname = os.path.join(TRAJ_DIR, f"{record_name}.npz")
    np.savez_compressed(fname, trajectory=arr, joint_names=JOINT_NAMES)
    print(f"[REC] Saved {len(frames)} frames -> {fname}")

# ---------------------------------------------------------------------------
# Hybrid Intuitive Frame
# ---------------------------------------------------------------------------
def _hybrid_directions(R):
    z_tool = R[:3, 2]; fwd = z_tool.copy(); fwd[2] = 0.0
    norm = np.linalg.norm(fwd)
    if norm < 1e-6: fwd = np.array([0.0, 1.0, 0.0])
    else: fwd /= norm
    left = np.array([-fwd[1], fwd[0], 0.0])
    up = np.array([0.0, 0.0, 1.0])
    return fwd, left, up

def reset_target():
    global target_pos, ik_progress
    q_now = np.array([data.qpos[j_qpos_ids[i]] for i in range(5)])
    T = ik.fk.forward_kinematics_5dof_matrix(q_now)
    target_pos = T[:3, 3].copy()
    ik_progress = 1.0

# ---------------------------------------------------------------------------
# Button flags
# ---------------------------------------------------------------------------
ee_active = {k: False for k in ["+X","-X","+Y","-Y","+Z","-Z"]}
jaw_active = {"+": False, "-": False}
joint_active = {f"{d}J{j}": False for d in "+-" for j in range(5)}
current_mode = ["ee"]

# ---------------------------------------------------------------------------
# tkinter panel — dark theme
# ---------------------------------------------------------------------------
BG, CARD, ACCENT = "#1a1a2e", "#16213e", "#0f3460"
GREEN, RED, PURPLE, BLUE, GREY = "#4caf50", "#e53935", "#7c4dff", "#448aff", "#546e7a"
TEXT, TEXT_DIM = "#eceff1", "#90a4ae"

root = tk.Tk()
root.title("SO-ARM100 Teleop")
root.attributes("-topmost", True)
root.configure(bg=BG)
root.geometry("900x700")

top = tk.Frame(root, bg=ACCENT, height=52)
top.pack(fill=tk.X); top.pack_propagate(False)

lbl_mode = tk.Label(top, text="● EE", font=("Arial", 20, "bold"), fg=BLUE, bg=ACCENT)
lbl_mode.pack(side=tk.LEFT, padx=(18, 0))

tk.Label(top, text="↑↓←→ = XY   Shift/Ctrl = Z   ,/. = Jaw",
         font=("Arial", 11), fg=TEXT_DIM, bg=ACCENT).pack(side=tk.RIGHT, padx=18)

body = tk.Frame(root, bg=BG)
body.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

left = tk.Frame(body, bg=CARD, relief=tk.FLAT, bd=0)
left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

tk.Label(left, text="END-EFFECTOR", font=("Arial", 13, "bold"), fg=BLUE, bg=CARD).pack(anchor="w", padx=14, pady=(12, 8))

ee_card = tk.Frame(left, bg=BG, relief=tk.FLAT, bd=0)
ee_card.pack(fill=tk.X, padx=10, pady=0)

lbl_ee = tk.Label(ee_card, text="Pos  ---", font=("Consolas", 18), fg=TEXT, bg=BG, anchor="w")
lbl_ee.pack(fill=tk.X, padx=12, pady=(6, 2))
lbl_tgt = tk.Label(ee_card, text="Tgt  ---", font=("Consolas", 18), fg=GREEN, bg=BG, anchor="w")
lbl_tgt.pack(fill=tk.X, padx=12, pady=(2, 6))

def _update_ee_display():
    ep = data.xpos[ee_body_id]
    lbl_ee.config(text=f"Pos  [{ep[0]:+7.4f}, {ep[1]:+7.4f}, {ep[2]:+7.4f}]")
    lbl_tgt.config(text=f"Tgt  [{target_pos[0]:+7.4f}, {target_pos[1]:+7.4f}, {target_pos[2]:+7.4f}]")

right = tk.Frame(body, bg=CARD, relief=tk.FLAT, bd=0)
right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(6, 0))

tk.Label(right, text="JOINTS", font=("Arial", 13, "bold"), fg=PURPLE, bg=CARD).pack(anchor="w", padx=14, pady=(12, 8))

joint_card = tk.Frame(right, bg=BG, relief=tk.FLAT, bd=0)
joint_card.pack(fill=tk.BOTH, padx=10, pady=0)

def _joint_btn(parent, text, direction):
    btn = tk.Button(parent, text=text, width=2, height=1, font=("Arial", 12, "bold"),
                     bg=PURPLE, fg="white", activebackground="#b388ff",
                     relief=tk.FLAT, bd=0, cursor="hand2")
    btn.bind("<ButtonPress>",   lambda e, d=direction: joint_press(d))
    btn.bind("<ButtonRelease>", lambda e, d=direction: joint_release(d))
    return btn

def ee_press(d):
    global ik_progress, target_pos, target_orient
    if current_mode[0] == "joint":
        ik_progress = 1.0
        q_now = np.array([data.qpos[j_qpos_ids[i]] for i in range(5)])
        T = ik.fk.forward_kinematics_5dof_matrix(q_now)
        target_pos = T[:3, 3].copy(); target_orient = T[:3, :3].copy()
    ee_active[d] = True; current_mode[0] = "ee"

def ee_release(d): ee_active[d] = False

def joint_press(d):
    global ik_progress
    if current_mode[0] == "ee": ik_progress = 1.0
    joint_active[d] = True; current_mode[0] = "joint"

def joint_release(d): joint_active[d] = False

joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_P", "Wrist_R"]
lbl_joint_vals = []
for j, jname in enumerate(joint_names):
    row = tk.Frame(joint_card, bg=BG); row.pack(pady=3, fill=tk.X)
    _joint_btn(row, "−", f"-J{j}").pack(side=tk.LEFT, padx=1)
    _joint_btn(row, "+", f"+J{j}").pack(side=tk.LEFT, padx=(0, 8))
    tk.Label(row, text=jname, width=9, anchor="w", font=("Arial", 14), fg=TEXT, bg=BG).pack(side=tk.LEFT)
    lbl_val = tk.Label(row, text="+0.000", width=8, anchor="e", font=("Consolas", 18), fg=PURPLE, bg=BG)
    lbl_val.pack(side=tk.RIGHT); lbl_joint_vals.append(lbl_val)

import ctypes
root.update()
_root_hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
ctypes.windll.imm32.ImmAssociateContext(_root_hwnd, 0)
_VK_LEFT, _VK_UP, _VK_RIGHT, _VK_DOWN = 0x25, 0x26, 0x27, 0x28
_VK_SHIFT, _VK_CONTROL = 0x10, 0x11
_VK_COMMA, _VK_PERIOD = 0xBC, 0xBE

def _poll_keys():
    gks = ctypes.windll.user32.GetAsyncKeyState
    for vk, dk in [(_VK_UP,"+Y"),(_VK_DOWN,"-Y"),(_VK_LEFT,"+X"),(_VK_RIGHT,"-X")]:
        if gks(vk) & 0x8000:
            if not ee_active[dk]: ee_press(dk)
        elif ee_active[dk]: ee_release(dk)
    if gks(_VK_SHIFT) & 0x8000:
        if not ee_active["+Z"]: ee_press("+Z")
    elif ee_active["+Z"]: ee_release("+Z")
    if gks(_VK_CONTROL) & 0x8000:
        if not ee_active["-Z"]: ee_press("-Z")
    elif ee_active["-Z"]: ee_release("-Z")
    jaw_active["-"] = bool(gks(_VK_COMMA) & 0x8000)
    jaw_active["+"] = bool(gks(_VK_PERIOD) & 0x8000)

bottom = tk.Frame(root, bg=ACCENT); bottom.pack(fill=tk.X)
btn_row = tk.Frame(bottom, bg=ACCENT); btn_row.pack(pady=10)

def toggle_record():
    global recording, record_start_time, frames, record_name
    if not recording:
        default_name = time.strftime("traj_%m%d_%H%M%S")
        name = simpledialog.askstring("Save Trajectory", "Trajectory name:", initialvalue=default_name, parent=root)
        if not name: return
        record_name = name
        recording = True; frames = []; record_start_time = time.time()
        btn_record.config(text="⏹  STOP", bg=RED)
        lbl_status.config(text=f"● RECORDING: {record_name}", fg=RED)
    else:
        recording = False; save_traj()
        btn_record.config(text="⏺  REC", bg=GREEN)
        lbl_status.config(text="✓ Saved", fg=GREEN)

def _settle_cube():
    for _ in range(300):
        data.ctrl[:6] = [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]
        mujoco.mj_step(model, data)

def reset_robot():
    global ik_progress
    for d in ee_active: ee_active[d] = False
    for d in joint_active: joint_active[d] = False
    for d in jaw_active: jaw_active[d] = False
    ik_progress = 1.0
    mujoco.mj_resetData(model, data)
    cube_qpos_local = data.qpos[6:13].copy()
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[6:13] = cube_qpos_local
    mujoco.mj_forward(model, data)
    _settle_cube()
    reset_target()

btn_record = tk.Button(btn_row, text="⏺  REC", width=12, height=2, font=("Arial", 18, "bold"),
                        bg=GREEN, fg="white", activebackground="#66bb6a",
                        relief=tk.FLAT, bd=0, cursor="hand2", command=toggle_record)
btn_record.pack(side=tk.LEFT, padx=8)

btn_reset = tk.Button(btn_row, text="↺  RESET", width=12, height=2, font=("Arial", 18, "bold"),
                       bg=GREY, fg="white", activebackground="#78909c",
                       relief=tk.FLAT, bd=0, cursor="hand2", command=reset_robot)
btn_reset.pack(side=tk.LEFT, padx=8)

info_row = tk.Frame(bottom, bg=ACCENT); info_row.pack(pady=(0, 10))
lbl_status = tk.Label(info_row, text="Ready", font=("Arial", 14, "bold"),
                       fg=TEXT_DIM, bg=ACCENT, width=30, anchor="center")
lbl_status.pack()

def _update_mode_indicator():
    if current_mode[0] == "joint":
        lbl_mode.config(text="● JOINT", fg=PURPLE)
    else:
        lbl_mode.config(text="● EE", fg=BLUE)

def _update_joint_display():
    for j in range(5):
        lbl_joint_vals[j].config(text=f"{data.ctrl[j]:+7.3f}")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
print("=" * 55)
print("SO100 Pick & Place")
print(f"  IK: position-only | IK interval: {IK_INTERVAL*1000:.0f} ms")
print(f"  Frame: Hybrid | Control: Keyboard")
print("=" * 55)

reset_target()

with mujoco.viewer.launch_passive(model, data,
                                   show_left_ui=False,
                                   show_right_ui=False) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [0, -0.35, 0.10]
    viewer.cam.distance = 0.7
    viewer.cam.azimuth = 270
    viewer.cam.elevation = -89

    wrist_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wrist")
    wrist_renderer = mujoco.Renderer(model, 720, 960)

    last_time = time.perf_counter()
    last_display_update = time.perf_counter()
    last_wrist_update = time.perf_counter()

    while viewer.is_running():
        now = time.perf_counter()
        dt = min(now - last_time, 0.05); last_time = now

        root.update(); _poll_keys()

        if current_mode[0] == "ee":
            any_pos = any(ee_active[k] for k in ee_active)
            if any_pos:
                step = POS_SPEED * dt
                btn_x = step * (ee_active["+X"] - ee_active["-X"])
                btn_y = step * (ee_active["+Y"] - ee_active["-Y"])
                btn_z = step * (ee_active["+Z"] - ee_active["-Z"])
                fwd, left, up = _hybrid_directions(target_orient)
                target_pos = target_pos + btn_y * fwd + btn_x * left + btn_z * up
                target_pos = np.clip(target_pos, [-0.20, -0.60, 0.02], [0.25, -0.10, 0.40])

            if any_pos and (now - last_ik_time) >= IK_INTERVAL:
                q_target, ok, _ = ik.solve(target_pos, None, data.ctrl[:5])
                if ok:
                    q_ik_start = data.ctrl[:5].copy()
                    q_ik_target = np.array(q_target); ik_progress = 0.0; last_ik_time = now

            if any_pos and ik_progress < 1.0:
                ik_progress += dt / IK_INTERVAL
                if ik_progress >= 1.0: ik_progress = 1.0
                data.ctrl[:5] = q_ik_start + ik_progress * (q_ik_target - q_ik_start)

        elif current_mode[0] == "joint":
            for j in range(5):
                d = joint_active[f"+J{j}"] - joint_active[f"-J{j}"]
                if d:
                    lo, hi = model.actuator_ctrlrange[j]
                    data.ctrl[j] = max(lo, min(hi, data.ctrl[j] + d * 1.0 * dt))

        jd = jaw_active["+"] - jaw_active["-"]
        if jd:
            lo, hi = model.actuator_ctrlrange[5]
            data.ctrl[5] = max(lo, min(hi, data.ctrl[5] + jd * JAW_SPEED * dt))

        if recording:
            t = time.time() - record_start_time
            frames.append(np.concatenate([[t], data.qpos[:6].copy(), data.ctrl[:6].copy(),
                          data.xpos[ee_body_id].copy(), data.xquat[ee_body_id].copy()]))

        if now - last_display_update > 0.10:
            try: _update_ee_display(); _update_mode_indicator(); _update_joint_display(); last_display_update = now
            except Exception: pass

        if now - last_wrist_update > 0.066:
            try:
                wrist_renderer.update_scene(data, camera=wrist_cam_id)
                wrist_img = wrist_renderer.render()
                cv2.imshow("Wrist Camera", cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1); last_wrist_update = now
            except Exception: pass

        mujoco.mj_step(model, data); viewer.sync()

cv2.destroyAllWindows()
print("[EXIT] Done.")
