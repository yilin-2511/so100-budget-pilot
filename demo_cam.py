"""SO100 Pick & Place — UI Redesign v2.

Same control engine as demo_sota.py, redesigned UI:
  - Mode indicator (EE blue / Joint green)
  - Speed slider (50–300 mm/s)
  - Joint angle display with real-time values
  - Color-coded EE buttons
  - Keyboard shortcut reference
  - HOME confirmation dialog
"""
import os
import sys
import time
import tkinter as tk
from tkinter import simpledialog, messagebox

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
# IK
# ---------------------------------------------------------------------------
ik = So100IK()

# Speed constants (m/s and rad/s — dt-scaled, loop-rate independent)
POS_SPEED = 0.10     # 100 mm/s EE (same as 2mm × 50Hz)
IK_INTERVAL = 0.05   # re-solve IK every 50ms (20 Hz)

JAW_SPEED = 1.0    # rad/s for jaw movement

# Persistent state
target_pos = np.zeros(3)
target_orient = np.eye(3)   # overwritten below from actual HOME pose

# Lock orientation from actual HOME gripper pose (NOT identity)
_home_q5 = np.array([data.qpos[j_qpos_ids[i]] for i in range(5)])
_home_T = ik.fk.forward_kinematics_5dof_matrix(_home_q5)
target_orient = _home_T[:3, :3].copy()
print(f"[LOCK] Gripper Z = [{target_orient[0,2]:+.4f}, {target_orient[1,2]:+.4f}, {target_orient[2,2]:+.4f}]")

# IK interpolation state (time-based, not waypoint-based)
q_ik_start = np.zeros(5)       # joint pos at last IK solve
q_ik_target = np.zeros(5)      # target joint pos from IK
ik_progress = 1.0              # 0→1 within one IK_INTERVAL; >=1.0 = done
last_ik_time = 0.0

# ---------------------------------------------------------------------------
# Recording
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
# Hybrid Intuitive Frame (ICRA 2024)
# ---------------------------------------------------------------------------
def _hybrid_directions(R):
    """forward=ground-projected gripper Z, left=perpendicular, up=world Z."""
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
    # target_orient is LOCKED — never overwritten after startup
    ik_progress = 1.0

# ---------------------------------------------------------------------------
# Button flags (set by tkinter bindings, read by main loop)
# ---------------------------------------------------------------------------
ee_active = {
    "+X": False, "-X": False, "+Y": False, "-Y": False,
    "+Z": False, "-Z": False,
}

jaw_active = {"+": False, "-": False}  # jaw +/- independent of mode

def jaw_press(d):
    jaw_active[d] = True

def jaw_release(d):
    jaw_active[d] = False

joint_active = {}
for j in range(5):  # arm joints only, jaw is separate
    joint_active[f"+J{j}"] = False
    joint_active[f"-J{j}"] = False

current_mode = [None]  # "ee" or "joint"

# ---------------------------------------------------------------------------
# tkinter panel — UI v2
# ---------------------------------------------------------------------------
root = tk.Tk()
root.title("SO-ARM100 Teleop")
root.attributes("-topmost", True)
root.geometry("560x680")

# ===== Top bar: mode + speed =====
top = tk.Frame(root, bg="#263238", height=40)
top.pack(fill=tk.X)
top.pack_propagate(False)

lbl_mode = tk.Label(top, text="● EE", font=("Arial", 13, "bold"),
                     fg="#64B5F6", bg="#263238", width=14, anchor="w")
lbl_mode.pack(side=tk.LEFT, padx=(12, 0))

tk.Label(top, text="↑↓←→=XY  Shift/Ctrl=Z  </>=Jaw", font=("Arial", 9),
         fg="#78909C", bg="#263238").pack(side=tk.RIGHT, padx=12)

# ===== Main content =====
main = tk.Frame(root)
main.pack(pady=6, fill=tk.BOTH, expand=True)

# --- Left: EE control ---
ee_frame = tk.Frame(main)
ee_frame.pack(side=tk.LEFT, padx=(12, 6))

tk.Label(ee_frame, text="End-Effector", font=("Arial", 11, "bold"),
         fg="#1565C0").pack(pady=(0, 6))

def ee_btn(parent, text, direction, w=4, h=2, color="#424242"):
    btn = tk.Button(parent, text=text, width=w, height=h,
                     font=("Arial", 16, "bold"),
                     bg=color, fg="white", activebackground="#616161",
                     relief=tk.RAISED, bd=2)
    btn.bind("<ButtonPress>",   lambda e, d=direction: ee_press(d))
    btn.bind("<ButtonRelease>", lambda e, d=direction: ee_release(d))
    return btn

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

# Arrow key polling
import ctypes
root.update()
_root_hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
ctypes.windll.imm32.ImmAssociateContext(_root_hwnd, 0)
_VK_LEFT, _VK_UP, _VK_RIGHT, _VK_DOWN = 0x25, 0x26, 0x27, 0x28
_VK_SHIFT, _VK_CONTROL = 0x10, 0x11
_VK_COMMA, _VK_PERIOD = 0xBC, 0xBE  # < and > keys

def _poll_keys():
    gks = ctypes.windll.user32.GetAsyncKeyState
    for vk, dir_key in [(_VK_UP, "+Y"), (_VK_DOWN, "-Y"),
                         (_VK_LEFT, "+X"), (_VK_RIGHT, "-X")]:
        if gks(vk) & 0x8000:
            if not ee_active[dir_key]:
                ee_press(dir_key)
        else:
            if ee_active[dir_key]:
                ee_release(dir_key)
    # Shift → Z up, Ctrl → Z down
    if gks(_VK_SHIFT) & 0x8000:
        if not ee_active["+Z"]:
            ee_press("+Z")
    else:
        if ee_active["+Z"]:
            ee_release("+Z")
    if gks(_VK_CONTROL) & 0x8000:
        if not ee_active["-Z"]:
            ee_press("-Z")
    else:
        if ee_active["-Z"]:
            ee_release("-Z")
    # < key → jaw close, > key → jaw open
    jaw_active["-"] = bool(gks(_VK_COMMA) & 0x8000)
    jaw_active["+"] = bool(gks(_VK_PERIOD) & 0x8000)

# EE position display
hf = tk.Frame(ee_frame, bg="#ECEFF1", relief=tk.GROOVE, bd=1)
hf.pack(pady=6, fill=tk.X, padx=2)

tk.Label(hf, text="EE Position", font=("Arial", 9, "bold"),
         bg="#ECEFF1", fg="#37474F").pack(anchor="w", padx=6, pady=(4, 0))
lbl_ee = tk.Label(hf, text="Actual: ---", font=("Consolas", 9),
                   fg="#1565C0", bg="#ECEFF1")
lbl_ee.pack(anchor="w", padx=10)
lbl_tgt = tk.Label(hf, text="Target: ---", font=("Consolas", 9),
                    fg="#2E7D32", bg="#ECEFF1")
lbl_tgt.pack(anchor="w", padx=10, pady=(0, 5))

def _update_ee_display():
    ep = data.xpos[ee_body_id]
    lbl_ee.config(text=f"Actual: [{ep[0]:+7.4f}, {ep[1]:+7.4f}, {ep[2]:+7.4f}]")
    lbl_tgt.config(text=f"Target: [{target_pos[0]:+7.4f}, {target_pos[1]:+7.4f}, {target_pos[2]:+7.4f}]")

# --- Separator ---
tk.Frame(main, width=2, bg="#B0BEC5").pack(side=tk.LEFT, padx=10, fill=tk.Y)

# --- Right: Joint control ---
jf = tk.Frame(main)
jf.pack(side=tk.LEFT, padx=(6, 12))

tk.Label(jf, text="Joints", font=("Arial", 11, "bold"),
         fg="#6A1B9A").pack(pady=(0, 6))

def joint_btn(parent, text, direction, w=3, h=2):
    btn = tk.Button(parent, text=text, width=w, height=h,
                     font=("Arial", 13, "bold"),
                     bg="#6A1B9A", fg="white", activebackground="#7B1FA2",
                     relief=tk.RAISED, bd=2)
    btn.bind("<ButtonPress>",   lambda e, d=direction: joint_press(d))
    btn.bind("<ButtonRelease>", lambda e, d=direction: joint_release(d))
    return btn

def joint_press(d):
    global ik_progress
    if current_mode[0] == "ee":
        ik_progress = 1.0
    joint_active[d] = True
    current_mode[0] = "joint"

def joint_release(d):
    joint_active[d] = False

joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_P", "Wrist_R"]
lbl_joint_vals = []  # keep references to update later
for j, jname in enumerate(joint_names):
    row = tk.Frame(jf)
    row.pack(pady=2)
    joint_btn(row, "−", f"-J{j}").pack(side=tk.LEFT, padx=2)
    joint_btn(row, "+", f"+J{j}").pack(side=tk.LEFT, padx=2)
    tk.Label(row, text=jname, width=9, anchor="w",
             font=("Arial", 11)).pack(side=tk.LEFT, padx=4)
    lbl_val = tk.Label(row, text="+0.000", width=7, anchor="e",
                        font=("Consolas", 10), fg="#6A1B9A")
    lbl_val.pack(side=tk.LEFT, padx=2)
    lbl_joint_vals.append(lbl_val)

# ===== Bottom bar: actions + status =====
bottom = tk.Frame(root, bg="#ECEFF1", height=44)
bottom.pack(fill=tk.X, side=tk.BOTTOM)
bottom.pack_propagate(False)

def toggle_record():
    global recording, record_start_time, frames, record_name
    if not recording:
        default_name = time.strftime("traj_%m%d_%H%M%S")
        name = simpledialog.askstring("Save Trajectory",
            "Trajectory name:", initialvalue=default_name, parent=root)
        if not name:
            return
        record_name = name
        recording = True; frames = []; record_start_time = time.time()
        btn_record.config(text="⏹ STOP", bg="#D32F2F")
        lbl_status.config(text=f"● RECORDING: {record_name}", fg="#D32F2F")
    else:
        recording = False; save_traj()
        btn_record.config(text="⏺ REC", bg="#2E7D32")
        lbl_status.config(text="✓ Saved", fg="#2E7D32")

def _settle_cube():
    for _ in range(300):
        data.ctrl[:6] = [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]
        mujoco.mj_step(model, data)

def reset_robot():
    global ik_progress
    for d in ee_active:    ee_active[d]    = False
    for d in joint_active: joint_active[d] = False
    for d in jaw_active:   jaw_active[d]   = False
    ik_progress = 1.0
    mujoco.mj_resetData(model, data)
    cube_qpos = data.qpos[6:13].copy()
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[6:13] = cube_qpos
    mujoco.mj_forward(model, data)
    _settle_cube()
    reset_target()

# Action bar
af = tk.Frame(root)
af.pack(pady=6)

btn_record = tk.Button(af, text="⏺ REC", width=7, height=1,
                        font=("Arial", 11, "bold"), bg="#2E7D32", fg="white",
                        relief=tk.RAISED, bd=2, command=toggle_record)
btn_record.pack(side=tk.LEFT, padx=3)

btn_reset = tk.Button(af, text="RESET", width=7, height=1,
                       font=("Arial", 11, "bold"), bg="#546E7A", fg="white",
                       relief=tk.RAISED, bd=2, command=reset_robot)
btn_reset.pack(side=tk.LEFT, padx=3)

lbl_status = tk.Label(af, text="Ready", font=("Arial", 10), fg="gray", width=30)
lbl_status.pack(side=tk.LEFT, padx=8)

def _update_mode_indicator():
    """Update mode indicator color and text."""
    if current_mode[0] == "joint":
        lbl_mode.config(text="● JOINT", fg="#CE93D8")
        # Joint panel gets purple highlight
    else:
        lbl_mode.config(text="● EE", fg="#64B5F6")

def _update_joint_display():
    """Update joint angle labels from current data.ctrl."""
    for j in range(5):
        lbl_joint_vals[j].config(text=f"{data.ctrl[j]:+7.3f}")

# ---------------------------------------------------------------------------
# Main loop (single loop — ctrl & physics are synced)
# ---------------------------------------------------------------------------
print("=" * 55)
print("SO100 Pick & Place — UI v2")
print(f"  IK: position-only | IK interval: {IK_INTERVAL*1000:.0f} ms")
print(f"  Frame: Hybrid | Control: Keyboard + Mouse")
print("=" * 55)

reset_target()

with mujoco.viewer.launch_passive(model, data,
                                   show_left_ui=False,
                                   show_right_ui=False) as viewer:
    # Camera: overhead (top-down) view
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [0, -0.35, 0.10]
    viewer.cam.distance = 0.7
    viewer.cam.azimuth = 270
    viewer.cam.elevation = -89  # nearly straight down

    # Wrist camera: offscreen render from cam_wrist (defined in so_arm100.xml)
    wrist_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wrist")
    wrist_w, wrist_h = 960, 720
    wrist_renderer = mujoco.Renderer(model, wrist_h, wrist_w)

    last_time = time.perf_counter()
    last_display_update = time.perf_counter()
    last_wrist_update = time.perf_counter()

    while viewer.is_running():
        now = time.perf_counter()
        dt = min(now - last_time, 0.05)  # cap to avoid jump after pause / lag
        last_time = now

        # 1. Process tkinter events + poll arrow keys (GetAsyncKeyState)
        root.update()
        _poll_keys()

        # 2. Read flags, update target pose
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

            # 3. Rate-limited IK solve → position-only (orientation free)
            #    Orientation stays close to current via initial guess proximity.
            if any_pos and (now - last_ik_time) >= IK_INTERVAL:
                q_target, ok, _ = ik.solve(target_pos, None, data.ctrl[:5])
                if ok:
                    q_ik_start = data.ctrl[:5].copy()
                    q_ik_target = np.array(q_target)
                    ik_progress = 0.0
                    last_ik_time = now

            # 4. Advance time-based interpolation → ctrl
            if any_pos and ik_progress < 1.0:
                ik_progress += dt / IK_INTERVAL
                if ik_progress >= 1.0:
                    ik_progress = 1.0
                alpha = ik_progress  # linear interpolation 0→1 over IK_INTERVAL
                data.ctrl[:5] = q_ik_start + alpha * (q_ik_target - q_ik_start)

        elif current_mode[0] == "joint":
            # Direct joint control (dt-scaled)
            JOINT_SPEED = 1.0  # rad/s
            for j in range(5):  # arm joints only
                d = joint_active[f"+J{j}"] - joint_active[f"-J{j}"]
                if d:
                    lo, hi = model.actuator_ctrlrange[j]
                    data.ctrl[j] = max(lo, min(hi,
                        data.ctrl[j] + d * JOINT_SPEED * dt))

        # 5. Jaw position — +/- keys, always active (lo=open, hi=close)
        jaw_idx = 5
        jd = jaw_active["+"] - jaw_active["-"]
        if jd:
            lo, hi = model.actuator_ctrlrange[jaw_idx]
            data.ctrl[jaw_idx] = max(lo, min(hi,
                data.ctrl[jaw_idx] + jd * JAW_SPEED * dt))

        # 6. Recording
        if recording:
            t = time.time() - record_start_time
            q = data.qpos[:6].copy()
            c = data.ctrl[:6].copy()
            ep = data.xpos[ee_body_id].copy()
            eq = data.xquat[ee_body_id].copy()
            frames.append(np.concatenate([[t], q, c, ep, eq]))

        # 7. Display updates (every 100ms)
        if now - last_display_update > 0.10:
            try:
                _update_ee_display()
                _update_mode_indicator()
                _update_joint_display()
                last_display_update = now
            except Exception:
                pass

        # 8. Wrist camera rendering (every ~66ms, ~15fps)
        if now - last_wrist_update > 0.066:
            try:
                wrist_renderer.update_scene(data, camera=wrist_cam_id)
                wrist_img = wrist_renderer.render()
                cv2.imshow("Wrist Camera", cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
                last_wrist_update = now
            except Exception:
                pass

        # 9. Physics
        mujoco.mj_step(model, data)
        viewer.sync()

cv2.destroyAllWindows()
print("[EXIT] Done.")
