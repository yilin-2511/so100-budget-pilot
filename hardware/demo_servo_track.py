"""SO100 Pick & Place — LeRobot Dataset Recording.

Reuses MuJoCo scene + IK + Hybrid Intuitive Frame from demo_cam.py.
All parameters configurable via CLI with draccus dataclass.

Usage:
  python demo_lerobot_record.py                           # defaults
  python demo_lerobot_record.py --record-fps 30           # override FPS
  python demo_lerobot_record.py --pos-speed 0.15          # faster EE
  python demo_lerobot_record.py --target-episodes 50      # auto-stop after 50 eps
"""
import os
import sys
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import simpledialog

import draccus
import numpy as np
import mujoco
import cv2
# --- 舵机 SDK ---
import sys as _sys
_servo_sdk_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "servo", "sdk")
_sys.path.insert(0, _servo_sdk_dir)
from scservo_sdk import *  # noqa
import serial.tools.list_ports as _serial_ports

_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from so100_ik import So100IK
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class RecordConfig:
    """SO-ARM100 teleop + recording configuration."""

    # --- Control ---
    pos_speed: float = 0.10          # EE 移动速度 (m/s)
    ik_interval: float = 0.05        # IK 重解间隔 (s)
    jaw_speed: float = 1.0           # 夹爪速度 (rad/s)
    joint_speed: float = 1.0         # 关节模式速度 (rad/s)

    # --- Recording ---
    record_fps: int = 10             # 录制帧率
    wrist_width: int = 640           # 腕部相机宽度
    wrist_height: int = 480          # 腕部相机高度
    episode_max_duration: float = 120.0  # 单集最长秒数
    target_episodes: int = 0         # 目标集数，0=无限

    # --- Dataset ---
    dataset_repo_id: str = "so100_sim_pick_place"
    dataset_root: str = "datasets/so100_sim"
    task: str = "Grab the red cube and move to target"

    # --- Cube randomization ---
    cube_random_xy: float = 0.03     # XY 随机偏移 (m)
# --- 舵机配置 ---
SERVO_BAUD = 1000000
SERVO_SPEED = 400           # 运动速度 (0-1023 步/s)
SERVO_DIR = -1              # 方向：1 或 -1（已翻转）
SERVO_CENTER_DEG = 150.0    # 0 rad 对应的舵机角度

# 关节索引 → 舵机 ID 映射 (Joint 0=Rotation, 1=Pitch, 2=Elbow, 3=Wrist_Pitch, 4=Wrist_Roll, 5=Jaw)
# Daisy-chain order from base to tip: ID 1-6, ID 7 reserved
SERVO_MAP = {
    0: 1,   # Rotation (base)
    1: 2,   # Pitch
    2: 3,   # Elbow
    3: 4,   # Wrist_Pitch
    4: 5,   # Wrist_Roll
    5: 6,   # Jaw
    # 6: 7, # Extra (reserved)
}

def init_servo():
    """初始化舵机连接。扫描 SERVO_MAP 里所有舵机。无硬件时安全跳过。"""
    ports = list(_serial_ports.comports())
    if not ports:
        print("[SERVO] No COM port — running sim-only")
        return None, None
    port = PortHandler(ports[0].device)
    port.setBaudRate(SERVO_BAUD)
    if not port.openPort():
        print("[SERVO] Cannot open port — running sim-only")
        return None, None
    bus = scscl(port)

    for joint_idx, sid in SERVO_MAP.items():
        pos, result, _ = bus.ReadPos(sid)
        if result == COMM_SUCCESS:
            print(f"[SERVO] Joint{joint_idx}=ID{sid} online, pos={pos}")
        else:
            print(f"[SERVO] Joint{joint_idx}=ID{sid} OFFLINE!")

    return port, bus

def rad_to_servo(rad):
    """MuJoCo 关节角 (rad) → SCS225 位置 (0-1023), 1:1 角度映射"""
    deg = np.degrees(rad) * SERVO_DIR + SERVO_CENTER_DEG
    raw = int(round(deg * 1023.0 / 300.0))
    return max(0, min(1023, raw))



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(cfg: RecordConfig):
    # -- MuJoCo scene --
    XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        "..", "model", "so100_pick_place.xml")

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

    # -- IK --
    ik = So100IK()

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

    # -- Hybrid Intuitive Frame --
    def _hybrid_directions(R):
        z_tool = R[:3, 2]; fwd = z_tool.copy(); fwd[2] = 0.0
        norm = np.linalg.norm(fwd)
        if norm < 1e-6: fwd = np.array([0.0, 1.0, 0.0])
        else: fwd /= norm
        left = np.array([-fwd[1], fwd[0], 0.0])
        up = np.array([0.0, 0.0, 1.0])
        return fwd, left, up

    def reset_target():
        nonlocal target_pos, ik_progress
        q_now = np.array([data.qpos[j_qpos_ids[i]] for i in range(5)])
        T = ik.fk.forward_kinematics_5dof_matrix(q_now)
        target_pos = T[:3, 3].copy()
        ik_progress = 1.0

    # -- Button flags --
    ee_active = {k: False for k in ["+X","-X","+Y","-Y","+Z","-Z"]}
    jaw_active = {"+": False, "-": False}
    joint_active = {f"{d}J{j}": False for d in "+-" for j in range(5)}
    current_mode = ["ee"]

    # -- LeRobot Dataset setup --
    import shutil as _shutil

    DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.dataset_root)
    _ds_base = DATASET_ROOT; _ds_idx = 1
    while True:
        if not os.path.exists(DATASET_ROOT): break
        if os.path.exists(os.path.join(DATASET_ROOT, "meta", "info.json")):
            _ds_idx += 1; DATASET_ROOT = f"{_ds_base}_{_ds_idx}"
        else:
            print(f"[DS] Removing incomplete directory: {DATASET_ROOT}")
            _shutil.rmtree(DATASET_ROOT); break

    FEATURES = {
        "observation.state": {
            "dtype": "float32", "shape": (6,),
            "names": ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"],
        },
        "action": {
            "dtype": "float32", "shape": (6,),
            "names": ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (cfg.wrist_height, cfg.wrist_width, 3),
            "names": ["height", "width", "channels"],
        },
    }

    dataset = None
    episode_idx = 0
    recording = False
    episode_start_time = 0.0

    def _create_or_get_dataset():
        nonlocal dataset
        if dataset is not None: return dataset
        print("[DS] Creating new dataset...")
        dataset = LeRobotDataset.create(
            repo_id=cfg.dataset_repo_id, fps=cfg.record_fps, features=FEATURES,
            root=DATASET_ROOT, use_videos=True, streaming_encoding=True, vcodec="libsvtav1",
        )
        print(f"[DS] Created at {DATASET_ROOT}")
        return dataset

    def _settle_cube():
        for _ in range(300):
            data.ctrl[:6] = [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]
            mujoco.mj_step(model, data)

    def reset_to_home():
        nonlocal ik_progress, target_pos
        for d in ee_active: ee_active[d] = False
        for d in joint_active: joint_active[d] = False
        for d in jaw_active: jaw_active[d] = False
        ik_progress = 1.0

        mujoco.mj_resetData(model, data)
        cube_qpos_local = data.qpos[6:13].copy()
        mujoco.mj_resetDataKeyframe(model, data, 0)
        r = cfg.cube_random_xy
        cube_qpos_local[0] += np.random.uniform(-r, r)
        cube_qpos_local[1] += np.random.uniform(-r, r)
        data.qpos[6:13] = cube_qpos_local
        mujoco.mj_forward(model, data)
        _settle_cube()

        q_now = np.array([data.qpos[j_qpos_ids[i]] for i in range(5)])
        T = ik.fk.forward_kinematics_5dof_matrix(q_now)
        target_pos = T[:3, 3].copy()
        print(f"[RESET] Cube @ ({data.qpos[6]:.3f}, {data.qpos[7]:.3f}, {data.qpos[8]:.3f})")

    # -- EE / Joint press handlers --
    def ee_press(d):
        nonlocal ik_progress, target_pos, target_orient
        if current_mode[0] == "joint":
            ik_progress = 1.0
            q_now = np.array([data.qpos[j_qpos_ids[i]] for i in range(5)])
            T = ik.fk.forward_kinematics_5dof_matrix(q_now)
            target_pos = T[:3, 3].copy(); target_orient = T[:3, :3].copy()
        ee_active[d] = True; current_mode[0] = "ee"

    def ee_release(d): ee_active[d] = False

    def joint_press(d):
        nonlocal ik_progress
        if current_mode[0] == "ee": ik_progress = 1.0
        joint_active[d] = True; current_mode[0] = "joint"

    def joint_release(d): joint_active[d] = False

    # -- Keyboard polling (Windows) --
    import platform as _platform
    _IS_WINDOWS = _platform.system() == "Windows"
    if _IS_WINDOWS: import ctypes

    _z_pressed, _r_pressed = False, False

    def _poll_keys_windows():
        nonlocal _z_pressed, _r_pressed, recording, episode_idx
        gks = ctypes.windll.user32.GetAsyncKeyState

        for vk, dk in [(0x26,"+Y"), (0x28,"-Y"), (0x25,"+X"), (0x27,"-X")]:
            if gks(vk) & 0x8000:
                if not ee_active[dk]: ee_press(dk)
            elif ee_active[dk]: ee_release(dk)

        if gks(0x10) & 0x8000:
            if not ee_active["+Z"]: ee_press("+Z")
        elif ee_active["+Z"]: ee_release("+Z")
        if gks(0x11) & 0x8000:
            if not ee_active["-Z"]: ee_press("-Z")
        elif ee_active["-Z"]: ee_release("-Z")

        jaw_active["-"] = bool(gks(0xBC) & 0x8000)
        jaw_active["+"] = bool(gks(0xBE) & 0x8000)

        z_now = bool(gks(0x5A) & 0x8000)
        if z_now and not _z_pressed:
            if recording and dataset is not None:
                dataset.clear_episode_buffer()
                btn_record.config(text="⏺ REC", bg=GREEN)
                lbl_status.config(text="✗ Discarded", fg=RED)
                recording = False
                print("[DS] Episode discarded (Z key).")
        _z_pressed = z_now

        r_now = bool(gks(0x52) & 0x8000)
        if r_now and not _r_pressed:
            if current_mode[0] == "joint":
                current_mode[0] = "ee"; lbl_mode.config(text="● EE", fg=BLUE)
            else:
                current_mode[0] = "joint"; lbl_mode.config(text="● JOINT", fg=PURPLE)
        _r_pressed = r_now

        return bool(gks(0x51) & 0x8000) or bool(gks(0x1B) & 0x8000)  # Q or ESC

    # -- Recording controls --
    def toggle_record():
        nonlocal recording, episode_start_time, episode_idx
        if not recording:
            _create_or_get_dataset()
            recording = True; episode_start_time = time.perf_counter()
            btn_record.config(text="⏹ STOP", bg=RED)
            lbl_status.config(text=f"● RECORDING: Episode {episode_idx}", fg=RED)
            print(f"[REC] Episode {episode_idx} started.")
        else:
            recording = False
            if dataset is not None:
                dataset.save_episode()
                print(f"[REC] Episode {episode_idx} saved ({dataset.num_episodes} total).")
            episode_idx += 1
            btn_record.config(text="⏺ REC", bg=GREEN)
            lbl_status.config(text=f"✓ Episode {episode_idx - 1} saved", fg=GREEN)
            reset_to_home()
            ctr = f"Ep {episode_idx}/{cfg.target_episodes}" if cfg.target_episodes > 0 else f"Ep {episode_idx}"
            lbl_ep_counter.config(text=ctr)

    # -- UI colors --
    BG, CARD, ACCENT = "#1a1a2e", "#16213e", "#0f3460"
    GREEN, RED, PURPLE, BLUE, GREY = "#4caf50", "#e53935", "#7c4dff", "#448aff", "#546e7a"
    TEXT, TEXT_DIM = "#eceff1", "#90a4ae"

    root = tk.Tk()
    root.title("SO-ARM100 Teleop")
    root.attributes("-topmost", True)
    root.configure(bg=BG)
    root.geometry("1080x880")

    top = tk.Frame(root, bg=ACCENT, height=60)
    top.pack(fill=tk.X); top.pack_propagate(False)

    lbl_mode = tk.Label(top, text="● EE", font=("Arial", 22, "bold"), fg=BLUE, bg=ACCENT)
    lbl_mode.pack(side=tk.LEFT, padx=(20, 0))
    tk.Label(top, text="↑↓←→ = XY   Shift/Ctrl = Z   ,/. = Jaw   Z = Discard   R = Mode",
             font=("Arial", 11), fg=TEXT_DIM, bg=ACCENT).pack(side=tk.RIGHT, padx=20)

    body = tk.Frame(root, bg=BG)
    body.pack(fill=tk.BOTH, expand=True, padx=15, pady=8)

    left = tk.Frame(body, bg=CARD, relief=tk.FLAT, bd=0)
    left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
    tk.Label(left, text="END-EFFECTOR", font=("Arial", 13, "bold"), fg=BLUE, bg=CARD).pack(anchor="w", padx=15, pady=(12, 10))
    ee_card = tk.Frame(left, bg=BG, relief=tk.FLAT, bd=0)
    ee_card.pack(fill=tk.X, padx=10, pady=0)
    lbl_ee = tk.Label(ee_card, text="Pos  ---", font=("Consolas", 20), fg=TEXT, bg=BG, anchor="w")
    lbl_ee.pack(fill=tk.X, padx=12, pady=(6, 2))
    lbl_tgt = tk.Label(ee_card, text="Tgt  ---", font=("Consolas", 20), fg=GREEN, bg=BG, anchor="w")
    lbl_tgt.pack(fill=tk.X, padx=12, pady=(2, 6))

    tk.Label(left, text="DATASET", font=("Arial", 13, "bold"), fg=BLUE, bg=CARD).pack(anchor="w", padx=15, pady=(16, 10))
    info_card = tk.Frame(left, bg=BG, relief=tk.FLAT, bd=0)
    info_card.pack(fill=tk.X, padx=10, pady=0)
    lbl_ds_name = tk.Label(info_card, text="Name  ---", font=("Consolas", 17), fg=TEXT_DIM, bg=BG, anchor="w")
    lbl_ds_name.pack(fill=tk.X, padx=12, pady=(6, 2))
    lbl_ds_eps = tk.Label(info_card, text="Eps   0", font=("Consolas", 17), fg=TEXT_DIM, bg=BG, anchor="w")
    lbl_ds_eps.pack(fill=tk.X, padx=12, pady=2)
    lbl_ds_frames = tk.Label(info_card, text="Frames  0", font=("Consolas", 17), fg=TEXT_DIM, bg=BG, anchor="w")
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
        ds_name = os.path.basename(DATASET_ROOT)
        n_eps = dataset.num_episodes if dataset is not None else 0
        _update_dataset_info(ds_name, n_eps)

    right = tk.Frame(body, bg=CARD, relief=tk.FLAT, bd=0)
    right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(6, 0))
    tk.Label(right, text="JOINTS", font=("Arial", 13, "bold"), fg=PURPLE, bg=CARD).pack(anchor="w", padx=15, pady=(12, 10))
    joint_card = tk.Frame(right, bg=BG, relief=tk.FLAT, bd=0)
    joint_card.pack(fill=tk.BOTH, padx=10, pady=0)

    def _joint_btn(parent, text, direction):
        btn = tk.Button(parent, text=text, width=2, height=1, font=("Arial", 12, "bold"),
                         bg=PURPLE, fg="white", activebackground="#b388ff",
                         relief=tk.FLAT, bd=0, cursor="hand2")
        btn.bind("<ButtonPress>", lambda e, d=direction: joint_press(d))
        btn.bind("<ButtonRelease>", lambda e, d=direction: joint_release(d))
        return btn

    joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_P", "Wrist_R"]
    lbl_joint_vals = []
    for j, jname in enumerate(joint_names):
        row = tk.Frame(joint_card, bg=BG); row.pack(pady=3, fill=tk.X)
        _joint_btn(row, "−", f"-J{j}").pack(side=tk.LEFT, padx=1)
        _joint_btn(row, "+", f"+J{j}").pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(row, text=jname, width=9, anchor="w", font=("Arial", 15), fg=TEXT, bg=BG).pack(side=tk.LEFT)
        lbl_val = tk.Label(row, text="+0.000", width=7, anchor="e", font=("Consolas", 20), fg=PURPLE, bg=BG)
        lbl_val.pack(side=tk.RIGHT); lbl_joint_vals.append(lbl_val)

    def _update_joint_display():
        for j in range(5): lbl_joint_vals[j].config(text=f"{data.ctrl[j]:+7.3f}")

    bottom = tk.Frame(root, bg=ACCENT); bottom.pack(fill=tk.X)
    btn_row = tk.Frame(bottom, bg=ACCENT); btn_row.pack(pady=(15, 5))

    btn_record = tk.Button(btn_row, text="⏺  REC", width=12, height=2, font=("Arial", 22, "bold"),
                            bg=GREEN, fg="white", activebackground="#388e3c",
                            relief=tk.FLAT, bd=0, cursor="hand2", command=toggle_record)
    btn_record.pack(side=tk.LEFT, padx=8)

    def discard_episode():
        nonlocal recording
        if recording and dataset is not None:
            dataset.clear_episode_buffer(); recording = False
            btn_record.config(text="⏺  REC", bg=GREEN); lbl_status.config(text="✗ Discarded", fg=RED)
            print("[REC] Episode discarded.")
            reset_to_home(); lbl_status.config(text="Ready", fg=TEXT_DIM)

    btn_discard = tk.Button(btn_row, text="✗  DISCARD", width=12, height=2, font=("Arial", 22, "bold"),
                             bg=RED, fg="white", activebackground="#c62828",
                             relief=tk.FLAT, bd=0, cursor="hand2", command=discard_episode)
    btn_discard.pack(side=tk.LEFT, padx=8)

    def reset_robot():
        nonlocal recording
        if recording and dataset is not None:
            dataset.clear_episode_buffer(); recording = False
            btn_record.config(text="⏺  REC", bg=GREEN); lbl_status.config(text="✗ Discarded", fg=RED)
        reset_to_home(); lbl_status.config(text="Ready", fg=TEXT_DIM)

    btn_reset = tk.Button(btn_row, text="↺  RESET", width=12, height=2, font=("Arial", 22, "bold"),
                           bg=GREY, fg="white", activebackground="#455a64",
                           relief=tk.FLAT, bd=0, cursor="hand2", command=reset_robot)
    btn_reset.pack(side=tk.LEFT, padx=8)

    info_row = tk.Frame(bottom, bg=ACCENT); info_row.pack(pady=(0, 15))
    lbl_status = tk.Label(info_row, text="Ready", font=("Arial", 18, "bold"),
                           fg=TEXT_DIM, bg=ACCENT, width=20, anchor="center")
    lbl_status.pack(side=tk.LEFT)
    lbl_ep_counter = tk.Label(info_row, text="Ep 0", font=("Arial", 18, "bold"),
                               fg=TEXT, bg=ACCENT, width=10, anchor="center")
    lbl_ep_counter.pack(side=tk.LEFT)

    root.update()

    # --- 舵机初始化 ---
    servo_port, servo_bus = init_servo()
    if _IS_WINDOWS:
        _root_hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        ctypes.windll.imm32.ImmAssociateContext(_root_hwnd, 0)

    # -- Renderers --
    wrist_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wrist")
    record_renderer = mujoco.Renderer(model, cfg.wrist_height, cfg.wrist_width)
    display_renderer = record_renderer

    # -- Main loop --
    print("=" * 55)
    print("SO100 Pick & Place — LeRobot Record")
    print(f"  IK interval: {cfg.ik_interval*1000:.0f}ms  FPS: {cfg.record_fps}")
    print(f"  Speed: EE={cfg.pos_speed}m/s  Jaw={cfg.jaw_speed}rad/s")
    print(f"  Dataset: {DATASET_ROOT}")
    print(f"  Cube random: ±{cfg.cube_random_xy*100:.0f}cm")
    print("=" * 55)

    reset_target()

    model.vis.map.force = 0.0
    model.vis.map.shadowclip = 0.0

    last_time = time.perf_counter()
    last_display_update = time.perf_counter()
    last_wrist_display = time.perf_counter()
    last_record_time = time.perf_counter()
    _wrist_window_shown = False
    _quit = False
    _last_servo_sync = 0.0   # 50Hz 频率限制
    _last_servo_land = {}    # 软着陆缓存

    while not _quit:
        loop_start = time.perf_counter()
        dt = min(loop_start - last_time, 0.05); last_time = loop_start

        root.update()
        if _IS_WINDOWS:
            q_pressed = _poll_keys_windows()
        else:
            q_pressed = False  # tkinter bindings would go here

        if _wrist_window_shown:
            try:
                if cv2.getWindowProperty("Wrist Camera (display)", cv2.WND_PROP_VISIBLE) < 1.0:
                    print("[EXIT] Wrist camera window closed."); break
            except Exception: pass

        if q_pressed:
            print("[EXIT] Quit requested."); break

        if recording:
            elapsed = time.perf_counter() - episode_start_time
            if elapsed >= cfg.episode_max_duration:
                print(f"[REC] Episode {episode_idx} reached max duration ({cfg.episode_max_duration}s).")
                toggle_record()
            if cfg.target_episodes > 0 and episode_idx >= cfg.target_episodes:
                print(f"[REC] Target episodes ({cfg.target_episodes}) reached.")
                if dataset is not None: dataset.finalize()
                break

        # 4. EE / Joint control
        if current_mode[0] == "ee":
            any_pos = any(ee_active[k] for k in ee_active)
            if any_pos:
                step = cfg.pos_speed * dt
                btn_x = step * (ee_active["+X"] - ee_active["-X"])
                btn_y = step * (ee_active["+Y"] - ee_active["-Y"])
                btn_z = step * (ee_active["+Z"] - ee_active["-Z"])
                fwd, left, up = _hybrid_directions(target_orient)
                target_pos = target_pos + btn_y * fwd + btn_x * left + btn_z * up
                target_pos = np.clip(target_pos, [-0.20, -0.60, 0.02], [0.25, -0.10, 0.40])

            if any_pos and (loop_start - last_ik_time) >= cfg.ik_interval:
                q_target, ok, _ = ik.solve(target_pos, None, data.ctrl[:5])
                if ok:
                    q_ik_start = data.ctrl[:5].copy()
                    q_ik_target = np.array(q_target); ik_progress = 0.0; last_ik_time = loop_start

            if any_pos and ik_progress < 1.0:
                ik_progress += dt / cfg.ik_interval
                if ik_progress >= 1.0: ik_progress = 1.0
                data.ctrl[:5] = q_ik_start + ik_progress * (q_ik_target - q_ik_start)

        elif current_mode[0] == "joint":
            for j in range(5):
                d = joint_active[f"+J{j}"] - joint_active[f"-J{j}"]
                if d:
                    lo, hi = model.actuator_ctrlrange[j]
                    data.ctrl[j] = max(lo, min(hi, data.ctrl[j] + d * cfg.joint_speed * dt))

        # 5. Jaw
        jd = jaw_active["+"] - jaw_active["-"]
        if jd:
            lo, hi = model.actuator_ctrlrange[5]
            data.ctrl[5] = max(lo, min(hi, data.ctrl[5] + jd * cfg.jaw_speed * dt))

        # 6. Sync physical servos (50Hz, 柔顺着陆)
        if servo_bus is not None and (loop_start - _last_servo_sync) > 0.02:
            for joint_idx, servo_id in SERVO_MAP.items():
                target = rad_to_servo(data.ctrl[joint_idx])
                # 软着陆：如果跳变太大，逐步逼近而不是一次到位
                key = ("last", joint_idx)
                prev = _last_servo_land.get(key)
                if prev is not None and abs(target - prev) > 20:
                    # 大步跳变（reset）→ 每帧最多走 8 步，慢慢过去
                    step_limit = 8
                    target = prev + max(-step_limit, min(step_limit, target - prev))
                _last_servo_land[key] = target
                servo_bus.SyncWritePos(servo_id, target, 0, SERVO_SPEED)
            servo_bus.groupSyncWrite.txPacket()
            servo_bus.groupSyncWrite.clearParam()
            _last_servo_sync = loop_start

        # 7. Physics
        mujoco.mj_step(model, data)

        # 8. Recording
        if recording and dataset is not None:
            if loop_start - last_record_time >= 1.0 / cfg.record_fps:
                record_renderer.update_scene(data, camera=wrist_cam_id)
                wrist_img = record_renderer.render()
                dataset.add_frame({
                    "observation.state": np.rad2deg(data.qpos[:6]).astype(np.float32),
                    "action": np.rad2deg(data.ctrl[:6]).astype(np.float32),
                    "observation.images.wrist": wrist_img,
                    "task": cfg.task,
                })
                last_record_time = loop_start

        # 9. Display (100ms)
        if loop_start - last_display_update > 0.10:
            try: _update_ee_display(); _update_joint_display(); last_display_update = loop_start
            except Exception as e: print(f"[WARN] Display: {e}")

        #10. Wrist cam display (~15fps)
        if loop_start - last_wrist_display > 0.066:
            try:
                display_renderer.update_scene(data, camera=wrist_cam_id)
                display_img = display_renderer.render()
                big = cv2.resize(display_img, (960, 720), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Wrist Camera (display)", cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1); last_wrist_display = loop_start; _wrist_window_shown = True
            except Exception as e: print(f"[WARN] Cam: {e}")

    # Cleanup
    if servo_port is not None:
        servo_port.closePort()
        print("[SERVO] Port closed.")
    try: cv2.destroyAllWindows()
    except Exception: pass
    print("[EXIT] Done.")
    if dataset is not None:
        print(f"[DS] Total episodes: {dataset.num_episodes}")
        print(f"[DS] Data root: {DATASET_ROOT}")


if __name__ == "__main__":
    cfg = draccus.parse(RecordConfig)
    main(cfg)
