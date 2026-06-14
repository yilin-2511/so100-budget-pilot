# SO-ARM100 Teleoperation — MuJoCo Simulation

SO-ARM100 robotic arm teleop in MuJoCo. Position-only IK + Hybrid Intuitive Frame + time interpolation.
Dark theme tkinter UI. `.npz` + LeRobot dataset recording.

## Quick Start

```bash
conda create -n so100 python=3.11 -y
conda activate so100
pip install -r requirements.txt

# Run (pick one)
python demo_lerobot_record.py # ★ Recommended — teleop + LeRobot dataset recording
python demo_cam.py            # Wrist camera + MuJoCo viewer + .npz recording
python demo_basic.py          # Minimal teleop — tkinter panel + keyboard
python replay.py              # Replay recorded trajectories (.npz)
```

## Programs

| Program | Description |
|---------|-------------|
| `demo_lerobot_record.py` | **★ Recommended** — teleop + LeRobotDataset v3 recording (MP4 + Parquet), draccus CLI config, dark theme UI |
| `demo_cam.py` | Wrist camera + MuJoCo viewer, dark theme UI, .npz recording |
| `demo_basic.py` | Basic teleop — tkinter panel + keyboard, .npz recording |
| `replay.py` | Trajectory player — scan `recordings/`, pick from list, replay with physics |
| [`traj_viewer/`](traj_viewer/) | Offline analysis — joint curves, EE 3D plot, arm animation, multi-trajectory compare |

---

## demo_lerobot_record.py — LeRobot Dataset Recording ★

Keyboard teleop + direct LeRobotDataset v3 recording. Writes MP4 video + Parquet joint data ready for ACT/Diffusion training. All parameters configurable via CLI with draccus.

### Quick Run

```bash
python demo_lerobot_record.py                          # Default: 10 FPS, ±3cm cube random
python demo_lerobot_record.py --record_fps 20          # 20 FPS
python demo_lerobot_record.py --cube_random_xy 0.05    # Wider cube randomization (±5cm)
python demo_lerobot_record.py --help                   # Show all configurable parameters
```

### Configurable Parameters

```bash
--pos_speed 0.15              # EE move speed (m/s, default 0.10)
--record_fps 20               # Recording FPS (default 10)
--cube_random_xy 0.05         # Cube XY randomization range in meters (default 0.03)
--episode_max_duration 60.0   # Auto-stop after N seconds per episode (default 120)
--target_episodes 50          # Auto-quit after N episodes (default 0 = indefinite)
--dataset_root datasets/my_data  # Custom dataset path
--wrist_width 640             # Wrist camera resolution width
--wrist_height 480            # Wrist camera resolution height
```

### Workflow

1. Press **⏺ REC** → recording starts
2. Keyboard-teleop the arm (pick → move → place)
3. Press **⏹ STOP** → episode saved, arm auto-resets to home with randomized cube
4. Press **✗ DISCARD** (or Z key) to dump the current episode
5. Repeat until done → Q/ESC to finalize

Each run auto-increments dataset directory (`datasets/so100_sim_1`, `_2`, …). Previous data is never overwritten.

### Keyboard Controls

| Key | Action |
|-----|--------|
| ↑↓←→ | EE XY move (Hybrid Intuitive Frame) |
| Shift / Ctrl | EE Z up / down |
| `,` / `.` | Jaw close / open |
| Z | Discard current episode |
| R | Toggle EE / JOINT mode |
| Q / ESC | Quit & finalize |

### Recording Specs

| Parameter | Value |
|-----------|-------|
| FPS | Configurable (default 10, recommended 20–30) |
| Resolution | 640 × 480 |
| Video | MP4 (SVT-AV1, streaming encoding) |
| Features | `observation.state` (6 joints, deg) + `action` (6 joints, deg) + `observation.images.wrist` |
| Cube randomization | Configurable (±3 cm XY default) |
| Format | LeRobot v3.0 — ready for `lerobot-train` |

### Output Structure

```
datasets/so100_sim_N/
├── data/         # Parquet — joint states & actions
├── videos/       # MP4 — wrist camera
└── meta/         # info.json, stats.json, tasks.parquet
```

### Training

```bash
lerobot-train \
    --dataset.repo_id "budget_pilot/datasets/so100_sim_1 budget_pilot/datasets/so100_sim_2 ..." \
    --policy.type act \
    --output_dir outputs/so100_act \
    --steps 50000
```

---

## demo_cam.py — Wrist Camera + MuJoCo Viewer

Keyboard teleop with MuJoCo viewer (3D scene) + wrist camera window. `.npz` recording.

### Keyboard Controls

| Key | Action |
|-----|--------|
| ↑↓←→ | EE XY move (Hybrid Intuitive Frame) |
| Shift / Ctrl | EE Z up / down |
| `,` / `.` | Jaw close / open |

### UI Layout (Dark Theme)

| Section | Content |
|---------|---------|
| **Top bar** | Mode indicator (● EE blue / ● JOINT purple) + keyboard cheat sheet |
| **End-Effector** | Live actual / target position readout |
| **Joints** | 5 joints — ± buttons + real-time angle display |
| **Bottom** | ⏺ REC (record .npz) / ↺ RESET / status |

### Control Modes

- **EE mode** (default, blue): Arrow keys move EE. Position-only IK at 20 Hz, linearly interpolated over 50 ms. EE speed = 100 mm/s.
- **Joint mode** (purple): Click joint ± buttons. Direct joint-space control at 1.0 rad/s. Switching back to EE locks current pose as new IK target.
- **Jaw**: `,` / `.` keys, always active, 1.0 rad/s.

### Wrist Camera

- Renders `cam_wrist` to separate OpenCV window
- Offscreen render at 960×720, ~15 FPS display
- First-person view of gripper and workspace

---

## demo_basic.py — Minimal Teleop

Same control engine, simpler UI. Arrow keys for EE XY. tkinter panel with EE XYZ buttons, joint ± buttons, jaw ± buttons, REC / HOME.

---

## replay.py — Trajectory Replay

- Scans `recordings/` for `.npz` files, sorted by most recent
- tkinter picker UI: select a trajectory → play
- Replays with full physics — cube can be pushed and collided
- Returns to picker after playback ends

---

## Core Architecture (all programs)

- **Position-only IK**: 3 constraints on 5 DOF — fast and stable. 20 Hz re-solving keeps orientation drift negligible.
- **Time interpolation**: IK results linearly smoothed over 50 ms — no joint snapping.
- **Hybrid Intuitive Frame** (ICRA 2024): Forward = ground-projected gripper Z. Directions always feel intuitive.
- **Single-loop + dt-scaled**: Synced physics/control in one `while` loop. Constant EE speed immune to frame-rate jitter.

---

## File Structure

```
so100-budget-pilot/
├── demo_lerobot_record.py     # ★ Recommended — teleop + LeRobot dataset recording
├── demo_cam.py                # Wrist camera + MuJoCo viewer + .npz
├── demo_basic.py              # Basic teleop
├── replay.py                  # Trajectory replay with picker (.npz)
├── traj_viewer/               # Offline analysis tool
├── so100_fk.py                # Forward kinematics (pure NumPy)
├── so100_ik.py                # Inverse kinematics (ikpy-based)
├── requirements.txt           # Python dependencies
├── model/
│   ├── so100_pick_place.xml   # MuJoCo scene (table + cube)
│   ├── so_arm100.xml          # SO-ARM100 robot model
│   └── assets/                # Mesh files (.stl)
├── recordings/                # Saved trajectories (.npz)
└── datasets/                  # LeRobot datasets (gitignored)
```

## Requirements

- Python ≥ 3.10
- MuJoCo ≥ 3.0
- ikpy ≥ 3.4
- NumPy ≥ 1.26
- opencv-python ≥ 4.0
- lerobot (`demo_lerobot_record.py`)
- draccus (`demo_lerobot_record.py`)
- pandas, av, scipy
- tkinter (bundled with Python)

## Trajectory Format (.npz)

```
[time, qpos(6), ctrl(6), ee_x, ee_y, ee_z, ee_qw, ee_qx, ee_qy, ee_qz]
```

Replay: `python replay.py` → select from list → play.
