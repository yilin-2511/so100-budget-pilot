"""SO100 Pick & Place — LeRobot Dataset Recording.

Reuses MuJoCo scene + IK + Hybrid Intuitive Frame from demo_cam.py.
Replaces .npz recording with direct LeRobotDataset writes + streaming video encoding.

Key differences from demo_cam.py:
  - Dual renderer: 640×480 for recording (LeRobot standard), 960×720 for display
  - LeRobotDataset with streaming_encoding=True (real-time MP4, no PNG round-trip)
  - 30 fps enforced via precise_sleep
  - Z key discards current episode (calls dataset.clear_episode_buffer)
  - Episode auto-reset to home pose after STOP
  - Episode counter in UI
"""
import os
import sys
import time
import tkinter as tk
from tkinter import simpledialog

import numpy as np
import mujoco
import cv2

# In budget_pilot/ — direct imports
from so100_ik import So100IK

# LeRobot imports
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ---------------------------------------------------------------------------
# MuJoCo scene (identical to demo_cam.py)
# ---------------------------------------------------------------------------
XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    "model", "so100_pick_place.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
model.dof_damping[:] = 3.0

j_qpos_ids = [model.jnt_qposadr[j] for j in range(6)]
ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")

# Scene init: save cube qpos, keyframe, restore cube, settle
mujoco.mj_resetData(model, data)
cube_qpos = data.qpos[6:13].copy()
mujoco.mj_resetDataKeyframe(model, data, 0)
data.qpos[6:13] = cube_qpos
mujoco.mj_forward(model, data)
for _ in range(300):
    data.ctrl[:6] = [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]
    mujoco.mj_step(model, data)

# ---------------------------------------------------------------------------
# IK (identical to demo_cam.py)
# ---------------------------------------------------------------------------
ik = So100IK()

POS_SPEED = 0.10
IK_INTERVAL = 0.05
JAW_SPEED = 1.0

target_pos = np.zeros(3)
target_orient = np.eye(3)

# Lock orientation from actual HOME gripper pose
_home_q5 = np.array([data.qpos[j_qpos_ids[i]] for i in range(5)])
_home_T = ik.fk.forward_kinematics_5dof_matrix(_home_q5)
target_orient = _home_T[:3, :3].copy()
print(f"[LOCK] Gripper Z = [{target_orient[0,2]:+.4f}, {target_orient[1,2]:+.4f}, {target_orient[2,2]:+.4f}]")

# IK interpolation state
q_ik_start = np.zeros(5)
q_ik_target = np.zeros(5)
ik_progress = 1.0
last_ik_time = 0.0

# ---------------------------------------------------------------------------
# Hybrid Intuitive Frame (identical to demo_cam.py)
# ---------------------------------------------------------------------------
def _hybrid_directions(R):
    z_tool = R[:3, 2]
    fwd = z_tool.copy()
    fwd[2] = 0.0
    norm = np.linalg.norm(fwd)
    if norm < 1e-6:
        fwd = np.array([0.0, 1.0, 0.0])
    else:
        fwd /= norm
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
# Button flags (identical to demo_cam.py)
# ---------------------------------------------------------------------------
ee_active = {
    "+X": False, "-X": False, "+Y": False, "-Y": False,
    "+Z": False, "-Z": False,
}

jaw_active = {"+": False, "-": False}

def jaw_press(d):
    jaw_active[d] = True

def jaw_release(d):
    jaw_active[d] = False

joint_active = {}
for j in range(5):
    joint_active[f"+J{j}"] = False
    joint_active[f"-J{j}"] = False

current_mode = [None]  # "ee" or "joint"

# ---------------------------------------------------------------------------
# LeRobot Dataset setup
# ---------------------------------------------------------------------------
import shutil as _shutil

# Datasets stored locally in budget_pilot/datasets/ — gitignored
DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    "datasets", "so100_sim")
# Auto-increment: if dataset already exists, use so100_sim_2, so100_sim_3, ...
_ds_base = DATASET_ROOT
_ds_idx = 1
while True:
    if not os.path.exists(DATASET_ROOT):
        break
    if os.path.exists(os.path.join(DATASET_ROOT, "meta", "info.json")):
        # Completed dataset — skip to next index
        _ds_idx += 1
        DATASET_ROOT = f"{_ds_base}_{_ds_idx}"
    else:
        # Incomplete / stale directory — remove it and reuse
        print(f"[DS] Removing incomplete directory: {DATASET_ROOT}")
        _shutil.rmtree(DATASET_ROOT)
        break

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ],
    },
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper",
        ],
    },
    "observation.images.wrist": {
        "dtype": "video",
        "shape": (480, 640, 3),       # H, W, C — auto-converted to CHW at training time
        "names": ["height", "width", "channels"],
    },
}

# ---------------------------------------------------------------------------
# Recording state
# ---------------------------------------------------------------------------
RECORD_FPS = 10   # lerobot IL-in-sim standard; 100ms/frame budget on Iris Xe
EPISODE_MAX_DURATION = 999.0  # no auto-stop — user manually presses STOP when done

dataset = None
episode_idx = 0
target_episodes = 0  # 0 = indefinite
recording = False
episode_start_time = 0.0


def _create_or_get_dataset():
    global dataset
    if dataset is not None:
        return dataset
    print("[DS] Creating LeRobotDataset with streaming encoding...")
    dataset = LeRobotDataset.create(
        repo_id="so100_sim_pick_place",
        fps=RECORD_FPS,
        features=FEATURES,
        root=DATASET_ROOT,
        use_videos=True,
        streaming_encoding=True,
        vcodec="libsvtav1",
    )
    print(f"[DS] Dataset created at {DATASET_ROOT}")
    return dataset


def _settle_cube():
    """Let the cube drop onto the table."""
    for _ in range(300):
        data.ctrl[:6] = [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]
        mujoco.mj_step(model, data)


def reset_to_home():
    """Reset robot arm to home keyframe + restore cube to initial position."""
    global ik_progress, target_pos
    # Stop all active keys
    for d in ee_active:
        ee_active[d] = False
    for d in joint_active:
        joint_active[d] = False
    for d in jaw_active:
        jaw_active[d] = False
    ik_progress = 1.0

    # Full scene reset + randomize cube XY (±3cm)
    mujoco.mj_resetData(model, data)
    cube_qpos = data.qpos[6:13].copy()
    mujoco.mj_resetDataKeyframe(model, data, 0)
    # Random XY offset: cube starts anywhere in a 6cm×6cm square
    cube_qpos[0] += np.random.uniform(-0.03, 0.03)
    cube_qpos[1] += np.random.uniform(-0.03, 0.03)
    data.qpos[6:13] = cube_qpos
    mujoco.mj_forward(model, data)
    _settle_cube()

    # Re-lock EE target to new arm position
    q_now = np.array([data.qpos[j_qpos_ids[i]] for i in range(5)])
    T = ik.fk.forward_kinematics_5dof_matrix(q_now)
    target_pos = T[:3, 3].copy()
    print(f"[RESET] Scene restored. Cube @ ({data.qpos[6]:.3f}, {data.qpos[7]:.3f}, {data.qpos[8]:.3f})")


# ---------------------------------------------------------------------------
# Keyboard polling (Windows: GetAsyncKeyState; cross-platform: tkinter binds)
# ---------------------------------------------------------------------------
import platform as _platform

_IS_WINDOWS = _platform.system() == "Windows"
if _IS_WINDOWS:
    import ctypes

_z_pressed = False
_q_pressed = False
_r_pressed = False


def _setup_tk_keyboard():
    """Bind keys via tkinter for non-Windows platforms."""
    root.bind_all("<KeyPress-Up>",    lambda e: ee_press("+Y"))
    root.bind_all("<KeyRelease-Up>",  lambda e: ee_release("+Y"))
    root.bind_all("<KeyPress-Down>",  lambda e: ee_press("-Y"))
    root.bind_all("<KeyRelease-Down>",lambda e: ee_release("-Y"))
    root.bind_all("<KeyPress-Left>",  lambda e: ee_press("+X"))
    root.bind_all("<KeyRelease-Left>",lambda e: ee_release("+X"))
    root.bind_all("<KeyPress-Right>", lambda e: ee_press("-X"))
    root.bind_all("<KeyRelease-Right>",lambda e: ee_release("-X"))
    root.bind_all("<KeyPress-Shift_L>", lambda e: ee_press("+Z"))
    root.bind_all("<KeyRelease-Shift_L>", lambda e: ee_release("+Z"))
    root.bind_all("<KeyPress-Control_L>", lambda e: ee_press("-Z"))
    root.bind_all("<KeyRelease-Control_L>", lambda e: ee_release("-Z"))
    root.bind_all("<KeyPress-comma>",   lambda e: jaw_active.update({"-": True}))
    root.bind_all("<KeyRelease-comma>", lambda e: jaw_active.update({"-": False}))
    root.bind_all("<KeyPress-period>",  lambda e: jaw_active.update({"+": True}))
    root.bind_all("<KeyRelease-period>",lambda e: jaw_active.update({"+": False}))
    root.bind_all("<KeyPress-z>", lambda e: _tk_discard())
    root.bind_all("<KeyPress-r>", lambda e: _tk_toggle_mode())
    root.bind_all("<KeyPress-q>", lambda e: _tk_quit())
    root.bind_all("<KeyPress-Escape>", lambda e: _tk_quit())


def _tk_discard():
    global recording
    if recording and dataset is not None:
        dataset.clear_episode_buffer()
        recording = False
        btn_record.config(text="⏺ REC", bg="#4caf50")
        lbl_status.config(text="✗ Discarded", fg="#e53935")
        print("[DS] Episode discarded (Z key).")


def _tk_toggle_mode():
    if current_mode[0] == "joint":
        current_mode[0] = "ee"
        lbl_mode.config(text="● EE", fg="#448aff")
    else:
        current_mode[0] = "joint"
        lbl_mode.config(text="● JOINT", fg="#7c4dff")


_quit_flag = False


def _tk_quit():
    global _quit_flag
    _quit_flag = True


def _poll_keys():
    if _IS_WINDOWS:
        _poll_keys_windows()
    # Non-Windows: tkinter bindings handle everything


def _poll_keys_windows():
    global _z_pressed, _q_pressed, _r_pressed, current_mode
    gks = ctypes.windll.user32.GetAsyncKeyState

    # Arrow keys → EE XY
    for vk, dir_key in [(0x26, "+Y"), (0x28, "-Y"), (0x25, "+X"), (0x27, "-X")]:
        if gks(vk) & 0x8000:
            if not ee_active[dir_key]:
                ee_press(dir_key)
        else:
            if ee_active[dir_key]:
                ee_release(dir_key)

    # Shift → Z up, Ctrl → Z down
    if gks(0x10) & 0x8000:
        if not ee_active["+Z"]: ee_press("+Z")
    elif ee_active["+Z"]: ee_release("+Z")
    if gks(0x11) & 0x8000:
        if not ee_active["-Z"]: ee_press("-Z")
    elif ee_active["-Z"]: ee_release("-Z")

    # < > jaw
    jaw_active["-"] = bool(gks(0xBC) & 0x8000)
    jaw_active["+"] = bool(gks(0xBE) & 0x8000)

    # Z — discard
    z_now = bool(gks(0x5A) & 0x8000)
    if z_now and not _z_pressed:
        global recording
        if recording and dataset is not None:
            dataset.clear_episode_buffer()
            btn_record.config(text="⏺ REC", bg="#4caf50")
            lbl_status.config(text="✗ Discarded", fg="#e53935")
            recording = False
            print("[DS] Episode discarded (Z key).")
    _z_pressed = z_now

    # Q / ESC — quit
    _q_pressed = bool(gks(0x51) & 0x8000) or bool(gks(0x1B) & 0x8000)

    # R — toggle mode
    r_now = bool(gks(0x52) & 0x8000)
    if r_now and not _r_pressed:
        if current_mode[0] == "joint":
            current_mode[0] = "ee"
            lbl_mode.config(text="● EE", fg="#448aff")
        else:
            current_mode[0] = "joint"
            lbl_mode.config(text="● JOINT", fg="#7c4dff")
    _r_pressed = r_now


# ---------------------------------------------------------------------------
# Reusable from demo_cam.py
# ---------------------------------------------------------------------------

def ee_press(d):
    global ik_progress, target_pos, target_orient
    if current_mode[0] == "joint":
        ik_progress = 1.0
        q_now = np.array([data.qpos[j_qpos_ids[i]] for i in range(5)])
        T = ik.fk.forward_kinematics_5dof_matrix(q_now)
        target_pos = T[:3, 3].copy()
        target_orient = T[:3, :3].copy()
    ee_active[d] = True
    current_mode[0] = "ee"


def ee_release(d):
    ee_active[d] = False


def joint_press(d):
    global ik_progress
    if current_mode[0] == "ee":
        ik_progress = 1.0
    joint_active[d] = True
    current_mode[0] = "joint"


def joint_release(d):
    joint_active[d] = False


# ---------------------------------------------------------------------------
# Recording controls
# ---------------------------------------------------------------------------

def toggle_record():
    global recording, episode_start_time, episode_idx

    if not recording:
        _create_or_get_dataset()
        recording = True
        episode_start_time = time.perf_counter()
        btn_record.config(text="⏹ STOP", bg="#e53935")
        lbl_status.config(text=f"● RECORDING: Episode {episode_idx}", fg="#e53935")
        print(f"[REC] Episode {episode_idx} started.")
    else:
        recording = False
        if dataset is not None:
            dataset.save_episode()
            print(f"[REC] Episode {episode_idx} saved ({dataset.num_episodes} episodes total).")
        episode_idx += 1
        btn_record.config(text="⏺ REC", bg="#4caf50")
        lbl_status.config(text=f"✓ Episode {episode_idx - 1} saved", fg="#4caf50")

        reset_to_home()

        if target_episodes > 0:
            lbl_ep_counter.config(text=f"Ep {episode_idx}/{target_episodes}")
        else:
            lbl_ep_counter.config(text=f"Ep {episode_idx}")


# ---------------------------------------------------------------------------
# tkinter panel
# ---------------------------------------------------------------------------
BG = "#1a1a2e"
CARD = "#16213e"
ACCENT = "#0f3460"
GREEN = "#4caf50"
RED = "#e53935"
PURPLE = "#7c4dff"
BLUE = "#448aff"
GREY = "#546e7a"
TEXT = "#eceff1"
TEXT_DIM = "#90a4ae"

root = tk.Tk()
root.title("SO-ARM100 Teleop")
root.attributes("-topmost", True)
root.configure(bg=BG)
root.geometry("1080x880")

# ===== Top bar =====
top = tk.Frame(root, bg=ACCENT, height=60)
top.pack(fill=tk.X)
top.pack_propagate(False)

lbl_mode = tk.Label(top, text="● EE", font=("Arial", 22, "bold"),
                     fg=BLUE, bg=ACCENT)
lbl_mode.pack(side=tk.LEFT, padx=(20, 0))

tk.Label(top, text="↑↓←→ = XY   Shift/Ctrl = Z   ,/. = Jaw   Z = Discard   R = Mode",
         font=("Arial", 11), fg=TEXT_DIM, bg=ACCENT).pack(side=tk.RIGHT, padx=20)

# ===== Main body =====
body = tk.Frame(root, bg=BG)
body.pack(fill=tk.BOTH, expand=True, padx=15, pady=8)

# --- Left panel: EE info ---
left = tk.Frame(body, bg=CARD, relief=tk.FLAT, bd=0)
left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

tk.Label(left, text="END-EFFECTOR", font=("Arial", 13, "bold"),
         fg=BLUE, bg=CARD).pack(anchor="w", padx=15, pady=(12, 10))

# EE position card
ee_card = tk.Frame(left, bg=BG, relief=tk.FLAT, bd=0)
ee_card.pack(fill=tk.X, padx=10, pady=0)

lbl_ee = tk.Label(ee_card, text="Pos  ---", font=("Consolas", 20),
                   fg=TEXT, bg=BG, anchor="w")
lbl_ee.pack(fill=tk.X, padx=12, pady=(6, 2))
lbl_tgt = tk.Label(ee_card, text="Tgt  ---", font=("Consolas", 20),
                    fg=GREEN, bg=BG, anchor="w")
lbl_tgt.pack(fill=tk.X, padx=12, pady=(2, 6))

# Dataset info card
tk.Label(left, text="DATASET", font=("Arial", 13, "bold"),
         fg=BLUE, bg=CARD).pack(anchor="w", padx=15, pady=(16, 10))

info_card = tk.Frame(left, bg=BG, relief=tk.FLAT, bd=0)
info_card.pack(fill=tk.X, padx=10, pady=0)

lbl_ds_name = tk.Label(info_card, text="Name  ---", font=("Consolas", 17),
                        fg=TEXT_DIM, bg=BG, anchor="w")
lbl_ds_name.pack(fill=tk.X, padx=12, pady=(6, 2))
lbl_ds_eps = tk.Label(info_card, text="Eps   0", font=("Consolas", 17),
                       fg=TEXT_DIM, bg=BG, anchor="w")
lbl_ds_eps.pack(fill=tk.X, padx=12, pady=2)
lbl_ds_frames = tk.Label(info_card, text="Frames  0", font=("Consolas", 17),
                          fg=TEXT_DIM, bg=BG, anchor="w")
lbl_ds_frames.pack(fill=tk.X, padx=12, pady=(2, 6))

def _update_dataset_info(name, n_eps):
    lbl_ds_name.config(text=f"Name  {name}")
    lbl_ds_eps.config(text=f"Eps   {n_eps}")
    n_frames = dataset.num_frames if dataset is not None else 0
    lbl_ds_frames.config(text=f"Frames  {n_frames}")

def _update_ee_display():
    ep = data.xpos[ee_body_id]
    lbl_ee.config(text=f"Pos  [{ep[0]:+7.4f}, {ep[1]:+7.4f}, {ep[2]:+7.4f}]")
    lbl_tgt.config(text=f"Tgt  [{target_pos[0]:+7.4f}, {target_pos[1]:+7.4f}, {target_pos[2]:+7.4f}]")
    # Update dataset info
    ds_name = os.path.basename(DATASET_ROOT)
    n_eps = dataset.num_episodes if dataset is not None else 0
    _update_dataset_info(ds_name, n_eps)

# --- Right panel: Joints ---
right = tk.Frame(body, bg=CARD, relief=tk.FLAT, bd=0)
right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(6, 0))

tk.Label(right, text="JOINTS", font=("Arial", 13, "bold"),
         fg=PURPLE, bg=CARD).pack(anchor="w", padx=15, pady=(12, 10))

joint_card = tk.Frame(right, bg=BG, relief=tk.FLAT, bd=0)
joint_card.pack(fill=tk.BOTH, padx=10, pady=0)

def joint_btn(parent, text, direction):
    btn = tk.Button(parent, text=text, width=2, height=1,
                     font=("Arial", 12, "bold"),
                     bg=PURPLE, fg="white", activebackground="#b388ff",
                     relief=tk.FLAT, bd=0, cursor="hand2")
    btn.bind("<ButtonPress>",   lambda e, d=direction: joint_press(d))
    btn.bind("<ButtonRelease>", lambda e, d=direction: joint_release(d))
    return btn

joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_P", "Wrist_R"]
lbl_joint_vals = []
for j, jname in enumerate(joint_names):
    row = tk.Frame(joint_card, bg=BG)
    row.pack(pady=3, fill=tk.X)
    joint_btn(row, "−", f"-J{j}").pack(side=tk.LEFT, padx=1)
    joint_btn(row, "+", f"+J{j}").pack(side=tk.LEFT, padx=(0, 8))
    tk.Label(row, text=jname, width=9, anchor="w",
             font=("Arial", 15), fg=TEXT, bg=BG).pack(side=tk.LEFT)
    lbl_val = tk.Label(row, text="+0.000", width=7, anchor="e",
                        font=("Consolas", 20), fg=PURPLE, bg=BG)
    lbl_val.pack(side=tk.RIGHT)
    lbl_joint_vals.append(lbl_val)

def _update_joint_display():
    for j in range(5):
        lbl_joint_vals[j].config(text=f"{data.ctrl[j]:+7.3f}")

# ===== Bottom bar =====
bottom = tk.Frame(root, bg=ACCENT)
bottom.pack(fill=tk.X, pady=(0, 0))

# Buttons row
btn_row = tk.Frame(bottom, bg=ACCENT)
btn_row.pack(pady=(15, 5))

btn_record = tk.Button(btn_row, text="⏺  REC", width=12, height=2,
                        font=("Arial", 22, "bold"), bg=GREEN, fg="white",
                        activebackground="#388e3c",
                        relief=tk.FLAT, bd=0, cursor="hand2", command=toggle_record)
btn_record.pack(side=tk.LEFT, padx=8)

def discard_episode():
    global recording
    if recording and dataset is not None:
        dataset.clear_episode_buffer()
        recording = False
        btn_record.config(text="⏺  REC", bg=GREEN)
        lbl_status.config(text="✗ Discarded", fg=RED)
        print("[REC] Episode discarded.")
        reset_to_home()
        lbl_status.config(text="Ready", fg=TEXT_DIM)

btn_discard = tk.Button(btn_row, text="✗  DISCARD", width=12, height=2,
                         font=("Arial", 22, "bold"), bg=RED, fg="white",
                         activebackground="#c62828",
                         relief=tk.FLAT, bd=0, cursor="hand2", command=discard_episode)
btn_discard.pack(side=tk.LEFT, padx=8)

def reset_robot():
    global recording
    if recording and dataset is not None:
        dataset.clear_episode_buffer()
        recording = False
        btn_record.config(text="⏺  REC", bg=GREEN)
        lbl_status.config(text="✗ Discarded", fg=RED)
        print("[REC] Episode discarded during reset.")
    reset_to_home()
    lbl_status.config(text="Ready", fg=TEXT_DIM)

btn_reset = tk.Button(btn_row, text="↺  RESET", width=12, height=2,
                       font=("Arial", 22, "bold"), bg=GREY, fg="white",
                       activebackground="#455a64",
                       relief=tk.FLAT, bd=0, cursor="hand2", command=reset_robot)
btn_reset.pack(side=tk.LEFT, padx=8)

# Status + counter in one row
info_row = tk.Frame(bottom, bg=ACCENT)
info_row.pack(pady=(0, 15))

lbl_status = tk.Label(info_row, text="Ready", font=("Arial", 18, "bold"),
                      fg=TEXT_DIM, bg=ACCENT, width=20, anchor="center")
lbl_status.pack(side=tk.LEFT)

lbl_ep_counter = tk.Label(info_row, text="Ep 0", font=("Arial", 18, "bold"),
                           fg=TEXT, bg=ACCENT, width=10, anchor="center")
lbl_ep_counter.pack(side=tk.LEFT)

# ===== IME setup (Windows) / tk keyboard (cross-platform) =====
root.update()
if _IS_WINDOWS:
    _root_hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
    ctypes.windll.imm32.ImmAssociateContext(_root_hwnd, 0)
else:
    _setup_tk_keyboard()

# ===== Renderers: record (640×480) + display (960×720) =====
wrist_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wrist")

RECORD_H, RECORD_W = 480, 640
display_renderer = record_renderer = mujoco.Renderer(model, RECORD_H, RECORD_W)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

print("=" * 55)
print("SO100 Pick & Place — LeRobot Record")
print(f"  IK: position-only | IK interval: {IK_INTERVAL*1000:.0f} ms")
print(f"  Frame: Hybrid | FPS: {RECORD_FPS} | Streaming: True")
print(f"  Dataset: {DATASET_ROOT}")
print("=" * 55)
print("  Keys: Arrow=EE XY | Shift/Ctrl=Z | </>=Jaw")
print("        Z=Discard | R=Toggle EE/JOINT | Q/ESC=Quit")
print("=" * 55)

reset_target()

# Disable shadows/force-vis to reduce GPU load on integrated graphics
model.vis.map.force = 0.0
model.vis.map.shadowclip = 0.0

last_time = time.perf_counter()
last_display_update = time.perf_counter()
last_wrist_display = time.perf_counter()
last_record_time = time.perf_counter()

_wrist_window_shown = False

_quit = False

print("[INFO] No-viewer mode — wrist camera window only.")

while not _quit:
    loop_start = time.perf_counter()
    dt = min(loop_start - last_time, 0.05)
    last_time = loop_start

    # 1. Process tkinter events + poll arrow keys
    root.update()
    _poll_keys()

    # 1b. Check cv2 window close (only after window was shown at least once)
    if _wrist_window_shown:
        try:
            visible = cv2.getWindowProperty("Wrist Camera (display)", cv2.WND_PROP_VISIBLE)
            if visible < 1.0:
                print("[EXIT] Wrist camera window closed.")
                _quit = True
                break
        except Exception:
            pass

    # 2. Check quit
    if _q_pressed or _quit_flag:
        print("[EXIT] Quit requested.")
        _quit = True
        break

    # 3. Auto-STOP: episode too long
    if recording:
        elapsed = time.perf_counter() - episode_start_time
        if elapsed >= EPISODE_MAX_DURATION:
            print(f"[REC] Episode {episode_idx} reached max duration "
                  f"({EPISODE_MAX_DURATION}s), auto-stopping.")
            toggle_record()

        if target_episodes > 0 and episode_idx >= target_episodes:
            print(f"[REC] Target episodes ({target_episodes}) reached.")
            dataset.finalize()
            _quit = True
            break

    # 4. Read flags, update target pose
    if current_mode[0] == "ee":
        any_pos = any([ee_active[k] for k in ["+X","-X","+Y","-Y","+Z","-Z"]])

        if any_pos:
            step = POS_SPEED * dt
            btn_x = step * (ee_active["+X"] - ee_active["-X"])
            btn_y = step * (ee_active["+Y"] - ee_active["-Y"])
            btn_z = step * (ee_active["+Z"] - ee_active["-Z"])

            fwd, left, up = _hybrid_directions(target_orient)
            target_pos = target_pos + btn_y * fwd + btn_x * left + btn_z * up
            target_pos = np.clip(target_pos,
                [-0.20, -0.60, 0.02], [0.25, -0.10, 0.40])

        if any_pos and (loop_start - last_ik_time) >= IK_INTERVAL:
            q_target, ok, _ = ik.solve(target_pos, None, data.ctrl[:5])
            if ok:
                q_ik_start = data.ctrl[:5].copy()
                q_ik_target = np.array(q_target)
                ik_progress = 0.0
                last_ik_time = loop_start

        if any_pos and ik_progress < 1.0:
            ik_progress += dt / IK_INTERVAL
            if ik_progress >= 1.0:
                ik_progress = 1.0
            alpha = ik_progress
            data.ctrl[:5] = q_ik_start + alpha * (q_ik_target - q_ik_start)

    elif current_mode[0] == "joint":
        JOINT_SPEED = 1.0
        for j in range(5):
            d = joint_active[f"+J{j}"] - joint_active[f"-J{j}"]
            if d:
                lo, hi = model.actuator_ctrlrange[j]
                data.ctrl[j] = max(lo, min(hi,
                    data.ctrl[j] + d * JOINT_SPEED * dt))

    # 5. Jaw position
    jaw_idx = 5
    jd = jaw_active["+"] - jaw_active["-"]
    if jd:
        lo, hi = model.actuator_ctrlrange[jaw_idx]
        data.ctrl[jaw_idx] = max(lo, min(hi,
            data.ctrl[jaw_idx] + jd * JAW_SPEED * dt))

    # 6. Physics step
    mujoco.mj_step(model, data)

    # 7. Recording — add frame at RECORD_FPS intervals (skip render otherwise)
    if recording and dataset is not None:
        if loop_start - last_record_time >= 1.0 / RECORD_FPS:
            record_renderer.update_scene(data, camera=wrist_cam_id)
            wrist_img = record_renderer.render()  # uint8 (480, 640, 3) HWC

            dataset.add_frame({
                "observation.state": np.rad2deg(data.qpos[:6]).astype(np.float32),
                "action":             np.rad2deg(data.ctrl[:6]).astype(np.float32),
                "observation.images.wrist": wrist_img,
                "task": "Grab the red cube and move to target",
            })

    # 8. Display updates (every 100ms)
    if loop_start - last_display_update > 0.10:
        try:
            _update_ee_display()
            _update_joint_display()
            last_display_update = loop_start
        except Exception as e:
            print(f"[WARN] Display update failed: {e}")

    # 9. Wrist camera display (~15fps)
    if loop_start - last_wrist_display > 0.066:
        try:
            display_renderer.update_scene(data, camera=wrist_cam_id)
            display_img = display_renderer.render()
            # Upscale display image for larger viewing window
            big = cv2.resize(display_img, (960, 720), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Wrist Camera (display)", cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            last_wrist_display = loop_start
            _wrist_window_shown = True
        except Exception as e:
            print(f"[WARN] Wrist cam display failed: {e}")

    # 10. No FPS throttle — control runs full speed, recording gates at RECORD_FPS

# Cleanup
try:
    cv2.destroyAllWindows()
except Exception:
    pass
print("[EXIT] Done.")
if dataset is not None:
    print(f"[DS] Total episodes: {dataset.num_episodes}")
    print(f"[DS] Data root: {DATASET_ROOT}")
