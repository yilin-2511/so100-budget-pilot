# SO-ARM100 SOTA Teleoperation

Single-loop EE teleoperation for the SO-ARM100 robotic arm in MuJoCo simulation. Position-only IK + Hybrid Intuitive Frame + time interpolation.

## Quick Start

```bash
# 1. Create and activate conda environment
conda create -n so100 python=3.11 -y
conda activate so100

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run (choose one)
python demo_sota.py      # Basic version — tkinter panel + keyboard
python demo_ui_v2.py     # Enhanced version — redesigned UI + wrist camera
python replay.py          # Trajectory replay with picker UI
```

## Programs

| Program | Description |
|---------|-------------|
| `demo_sota.py` | Basic teleoperation: tkinter panel + keyboard control, trajectory recording |
| `demo_ui_v2.py` | **Enhanced**: redesigned UI with mode indicator, speed slider, real-time joint display, wrist camera |
| `replay.py` | Trajectory player: scan `recordings/`, pick from list, replay with physics |

## Controls

### Keyboard (works even when MuJoCo window is focused)

| Key | Action |
|-----|--------|
| ↑ ↓ ← → | Move EE in XY (Hybrid Frame) |

### tkinter Panel

| Section | Controls | Action |
|---------|----------|--------|
| End-Effector | XY pad, Z buttons | Move EE position |
| Joints | +/- buttons for 5 joints | Direct joint control |
| Jaw | +/- buttons | Gripper open/close |
| Bottom | REC / HOME | Record trajectory / Reset scene |

### Control Modes

- **EE mode** (default): Press any EE button → position-only IK moves the arm. Hybrid Frame directions adapt to gripper orientation.
- **Joint mode**: Press any Joint button → direct joint-space control. Switch back to EE to lock the new pose.
- **Jaw**: Always active, independent of mode.

## demo_ui_v2.py — Enhanced UI & Wrist Camera

Redesigned control panel with real-time visual feedback:

- **Mode indicator**: EE mode (blue) vs Joint mode (green) — always visible
- **Speed slider**: Adjust EE movement speed from 50–300 mm/s
- **Joint angle display**: Real-time numeric readout of all 5 joint angles
- **Color-coded EE buttons**: Direction buttons visually distinguished
- **Keyboard shortcut reference**: On-screen cheat sheet
- **HOME confirmation dialog**: Prevent accidental resets

**Wrist camera**: Renders the onboard `cam_wrist` camera to an OpenCV window at ~15 FPS, giving a first-person view of the gripper and workspace.

## replay.py — Trajectory Replay

- Scans `recordings/` for saved `.npz` trajectories, sorted by most recent
- tkinter picker UI: select a trajectory from the list, click to start replay
- Replays with full physics interaction — cube can be pushed, collided, etc.
- Loop back to picker after replay completes

## Key Features

- **Position-only IK**: 3 constraints on 5 DOF — fast and stable. Orientation maintained naturally by solver proximity.
- **Hybrid Intuitive Frame** (ICRA 2024): Forward = ground-projected gripper Z. Directions feel intuitive regardless of gripper orientation.
- **Single-loop architecture**: Control and physics synced in one `while` loop — no jitter.
- **dt-scaled movement**: Constant EE speed regardless of simulation frame rate.
- **Time interpolation**: Joint commands smoothly interpolated over 50ms between IK solves.
- **Trajectory recording**: Named `.npz` files saved to `recordings/`.

## File Structure

```
so100-budget-pilot/
├── demo_sota.py              # Basic teleoperation
├── demo_ui_v2.py             # Enhanced teleoperation (UI v2 + wrist camera)
├── replay.py                 # Trajectory replay with picker
├── so100_fk.py               # Forward kinematics (pure NumPy)
├── so100_ik.py               # Inverse kinematics (ikpy-based)
├── __init__.py               # Module init
├── requirements.txt          # Python dependencies
├── README.md
├── README_CN.md
├── model/
│   ├── so100_pick_place.xml   # MuJoCo scene (table + cube)
│   ├── so_arm100.xml          # SO-ARM100 robot model
│   └── assets/                # Mesh files (.stl)
└── recordings/               # Saved trajectories (.npz)
```

## Requirements

- Python ≥ 3.10
- MuJoCo ≥ 3.0
- ikpy ≥ 3.4
- NumPy ≥ 1.26
- opencv-python ≥ 4.0 (for wrist camera in `demo_ui_v2.py`)
- tkinter (bundled with Python on most platforms)

## Recording & Replay

Click **REC** → enter a name → operate the arm → click **STOP**.
Trajectories save to `recordings/` as `.npz` files.

Replay with: `python replay.py` — select from the list UI.

Format per frame: `[time, qpos(6), ctrl(6), ee_xyz(3), ee_quat(4)]` — 20 columns total.
