# SO-ARM100 BUDGET Teleoperation

SO-ARM100 robotic arm teleoperation in MuJoCo simulation. Position-only IK + Hybrid Intuitive Frame + time interpolation.

## Quick Start

```bash
conda create -n so100 python=3.11 -y
conda activate so100
pip install -r requirements.txt

# Run (pick one)
python demo_cam.py       # Recommended — wrist camera + enhanced UI
python demo_basic.py     # Minimal version — tkinter panel + keyboard
python replay.py         # Replay recorded trajectories
```

## Programs

| Program | Description |
|---------|-------------|
| `demo_cam.py` | **Recommended** — wrist camera, EE position display, real-time joint angles, redesigned UI |
| `demo_basic.py` | Basic teleoperation — tkinter panel + keyboard, trajectory recording |
| `replay.py` | Trajectory player — scan `recordings/`, pick from list, replay with physics |
| [`traj_viewer/`](traj_viewer/) | **Offline analysis** — joint curves, EE 3D plot, arm animation, multi-trajectory compare (no MuJoCo) |

---

## demo_cam.py — Wrist Camera & Enhanced UI

### Keyboard Shortcuts (global — works even when MuJoCo window is focused)

| Key | Action |
|-----|--------|
| **Arrow keys** ↑ ↓ ← → | Move EE in XY (Hybrid Frame directions) |
| **Shift** | Move EE up (+Z) |
| **Ctrl** | Move EE down (−Z) |
| **<** (comma) | Close jaw |
| **>** (period) | Open jaw |

Shortcuts displayed on-screen in the tkinter top bar for reference.

### tkinter Panel

| Section | Content |
|---------|---------|
| **Top bar** | Mode indicator (● EE blue / ● JOINT purple) + keyboard cheat sheet |
| **End-Effector** | Direction buttons (+X/−X/+Y/−Y/+Z/−Z) + live actual/target position readout |
| **Joints** | 5 joints (Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll) — ± buttons + real-time angle display |
| **Bottom** | ⏺ REC (record trajectory) / RESET (restore home pose) / status bar |

### Control Modes

- **EE mode** (default, blue indicator): Press any EE direction button or arrow key. Position-only IK drives the arm via Hybrid Intuitive Frame. IK solved at 20 Hz, linearly interpolated over 50 ms for smooth motion. EE speed = 100 mm/s.
- **Joint mode** (purple indicator): Press any Joint ± button. Direct joint-space control at 1.0 rad/s. Switching back to EE mode locks the current pose as new IK target.
- **Jaw**: Controlled via keyboard **<** / **>** keys, always active regardless of mode (1.0 rad/s).

### Wrist Camera

- Renders the onboard `cam_wrist` to a separate OpenCV window ("Wrist Camera")
- Offscreen render at 960×720, displayed at ~15 FPS
- Provides first-person view of the gripper and workspace

### Other Features

- **EE position display**: Actual (from physics) vs target (from IK) shown side-by-side in mm
- **Joint angle display**: Real-time numeric readout of all 5 joint angles in radians
- **Overhead camera**: MuJoCo viewer set to top-down view of the workspace
- Trajectory recording: **REC** → name prompt → operate → **STOP** → saves `.npz` to `recordings/`

---

## demo_basic.py — Minimal Teleoperation

Same control engine, simpler UI:

| Key (keyboard) | Action |
|----------------|--------|
| Arrow keys ↑ ↓ ← → | Move EE in XY (Hybrid Frame) |

tkinter panel with EE XY/Z buttons, joint ± buttons, jaw ± buttons, REC / HOME.

---

## replay.py — Trajectory Replay

- Scans `recordings/` for `.npz` files, sorted by most recent
- tkinter picker UI: select a trajectory → play
- Replays with full physics — cube can be pushed and collided
- Returns to picker after playback ends

---

## Key Features (all programs)

- **Position-only IK**: 3 constraints on 5 DOF — fast and stable. Orientation drifts very little due to high-frequency re-solving (20 Hz).
- **Time interpolation**: IK results linearly smoothed over 50 ms — no joint snapping even when IK output jumps.
- **Hybrid Intuitive Frame** (ICRA 2024): Forward = ground-projected gripper Z. Directions always feel intuitive regardless of gripper orientation.
- **Single-loop + dt-scaled**: Synced physics/control in one `while` loop. Constant EE speed (100 mm/s) immune to frame-rate jitter.

> **Why does it feel so smooth?** Position-only IK (fast, minimal constraints) + time interpolation (absorbs IK jitter) + Hybrid Frame (intuitive directions) + single-loop (no threading artifacts).

---

## File Structure

```
so100-budget-pilot/
├── demo_cam.py               # Recommended — wrist camera + enhanced UI
├── demo_basic.py              # Basic teleoperation
├── replay.py                  # Trajectory replay with picker
├── traj_viewer/               # Offline analysis tool (joint curves, EE 3D, animation)
├── so100_fk.py                # Forward kinematics (pure NumPy)
├── so100_ik.py                # Inverse kinematics (ikpy-based)
├── __init__.py                # Module init
├── requirements.txt           # Python dependencies
├── model/
│   ├── so100_pick_place.xml   # MuJoCo scene (table + cube)
│   ├── so_arm100.xml          # SO-ARM100 robot model
│   └── assets/                # Mesh files (.stl)
└── recordings/                # Saved trajectories (.npz)
```

## Requirements

- Python ≥ 3.10
- MuJoCo ≥ 3.0
- ikpy ≥ 3.4
- NumPy ≥ 1.26
- opencv-python ≥ 4.0 (`demo_cam.py` wrist camera)
- tkinter (bundled with Python)

## Trajectory Format

Recorded `.npz` files contain a 20-column array per frame:

```
[time, qpos(6), ctrl(6), ee_x, ee_y, ee_z, ee_qw, ee_qx, ee_qy, ee_qz]
```

Replay: `python replay.py` → select from list → play.
