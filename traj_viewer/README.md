# SO-ARM100 Trajectory Viewer

Offline analysis tool for trajectories recorded by [so100-budget-pilot](https://github.com/yilin-2511/so100-budget-pilot) (demo_cam). Load `.npz` files and visualize joint angles, end-effector paths, and arm kinematics — no MuJoCo required.

## Quick Start

```bash
pip install numpy matplotlib
python main.py
```

## Features

| Function | Description |
|----------|-------------|
| Joint angle curves | 5 arm joints over time — qpos (actual) vs ctrl (target) |
| EE 3D trajectory | MuJoCo ground-truth vs pure-NumPy FK prediction overlay |
| 3D arm animation | Stick-figure playback with 1×/3×/5× speed, pause, scrub |
| Multi-trajectory compare | Overlay joint angles from 2+ recordings |
| Stats summary | Duration, joint ranges, tracking error (°), FK accuracy (mm) |

## Requirements

- Python ≥ 3.10
- NumPy ≥ 1.26
- matplotlib ≥ 3.5
- tkinter (bundled with Python)

## Trajectory Format

`.npz` files with array shape `(N, 20)`:

```
[time, qpos(6), ctrl(6), ee_x, ee_y, ee_z, ee_qw, ee_qx, ee_qy, ee_qz]
```

Recorded by [so100-budget-pilot](https://github.com/yilin-2511/so100-budget-pilot).
