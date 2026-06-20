# SO-ARM100 Budget Pilot — Sim-to-Real 模仿学习数据管道

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SO-ARM100 机械臂 MuJoCo 遥操作与数据采集系统。面向 sim-to-real 行为克隆管线：人类遥操作 → 数据集录制 → ACT/Diffusion 训练 → 实体部署。

仅位置 IK + 混合直觉坐标系 + 时间插值。暗色主题 tkinter UI。支持 `.npz` + LeRobot v3 数据集录制。

![SO-ARM100 Teleop](docs/image.png)

![控制面板](docs/board.png)

## 项目背景

训练视动策略（ACT、Diffusion Policy、π₀、iFlyBot-VLA）需要**演示数据**：由人类完成任务的 (图像, 关节状态) → 动作序列对。本仓库提供仿真侧工具链：

```
┌─────────────────────────────────────────────────────────────┐
│                    SO-ARM100 Budget Pilot                    │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌────────────┐             │
│  │ 键盘遥操作 │───>│    IK    │───>│  MuJoCo     │             │
│  │          │    │ (ikpy)   │    │  仿真       │             │
│  │          │    │ 20Hz DLS │    │            │             │
│  │ ↑↓←→     │    │ 仅位置    │    │  SO-ARM100 │             │
│  │ Shift/Ctrl   │ 5-DOF    │    │  + 桌面     │             │
│  │ , . 夹爪     │          │    │  + 方块     │             │
│  └──────────┘    └──────────┘    └─────┬──────┘             │
│                                        │                    │
│                          ┌─────────────┴─────────────┐      │
│                          │                           │      │
│                     腕部相机                     关节状态    │
│                     (640×480)                   (6 DoF, °)  │
│                          │                           │      │
│                          └──────────┬────────────────┘      │
│                                     │                       │
│                                     v                       │
│                          ┌──────────────────┐               │
│                          │  LeRobotDataset   │               │
│                          │  v3.0             │               │
│                          │                   │               │
│                          │  /data/*.parquet  │               │
│                          │  /videos/*.mp4    │               │
│                          │  /meta/           │               │
│                          └────────┬─────────┘               │
│                                   │                         │
└───────────────────────────────────┼─────────────────────────┘
                                    │
                                    v
┌─────────────────────────────────────────────────────────────┐
│                      训练 (GPU)                              │
│                                                             │
│  lerobot-train --policy.type act --dataset.repo_id ...      │
│                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   ACT    │    │  Diffusion   │    │  iFlyBot-VLA     │   │
│  │  (CVAE)  │    │  Policy      │    │  (Flow Matching) │   │
│  │  ~80M    │    │              │    │  ~2B             │   │
│  └──────────┘    └──────────────┘    └──────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**研究背景**：本管道瞄准与 [ACT](https://arxiv.org/abs/2304.13705)（CVAE + Action Chunking）、[OpenVLA](https://arxiv.org/abs/2406.09246)（VLM→动作）、[π₀](https://arxiv.org/abs/2410.24164)（Flow Matching VLA）、[iFlyBot-VLA](https://arxiv.org/abs/2511.01914)（双动作表征）相同的问题设定——它们都从人类演示数据出发，这里提供生成这些数据的工具。

## 快速开始

```bash
conda create -n so100 python=3.11 -y
conda activate so100
pip install -r requirements.txt

# 录制 LeRobot 数据集（★ 推荐）
python demo_lerobot_record.py

# 或其他程序：
python demo_cam.py            # 腕部相机 + MuJoCo 窗口 + .npz
python demo_basic.py          # 基础版 — tkinter 面板 + 键盘
python replay.py              # 回放已录制 .npz 轨迹
```

## 程序说明

| 程序 | 说明 | 数据输出 |
|------|------|----------|
| `demo_lerobot_record.py` | **★ 推荐** — 遥操作 + LeRobotDataset v3 录制（MP4 + Parquet），draccus CLI | LeRobot v3 数据集 |
| `demo_cam.py` | 腕部相机 + MuJoCo 窗口，暗色主题 UI | `.npz` |
| `demo_basic.py` | 基础遥操作 — tkinter 面板 + 键盘 | `.npz` |
| `replay.py` | 轨迹回放 — 扫描 `recordings/`，列表选择后物理回放 | — |
| `traj_viewer/` | 离线分析 — 关节曲线、EE 3D 图、机械臂动画 | — |
| `hardware/` | SCS225 舵机桥接 — MuJoCo ↔ 实体臂 | — |

---

## demo_lerobot_record.py — LeRobot 数据集录制 ★

键盘遥操作 + 直接写入 LeRobotDataset v3。输出 MP4 视频 + Parquet 关节数据，可直接用于 ACT / Diffusion Policy / VLA 训练。所有参数通过 draccus 命令行配置。

### 快速运行

```bash
python demo_lerobot_record.py                          # 默认：10 FPS，方块 ±3cm 随机
python demo_lerobot_record.py --record_fps 20          # 20 FPS
python demo_lerobot_record.py --cube_random_xy 0.05    # 方块随机范围 ±5cm
python demo_lerobot_record.py --help                   # 查看所有参数
```

### 命令行参数

```bash
--record_fps 20               # 录制帧率（默认 10）
--cube_random_xy 0.05         # 方块 XY 随机偏移范围 m（默认 0.03）
--pos_speed 0.15              # EE 移动速度 m/s（默认 0.10）
--episode_max_duration 60.0   # 单集最长秒数（默认 120）
--target_episodes 50          # 录满 N 集自动退出（默认 0 = 无限）
--dataset_root datasets/my_data  # 自定义数据集路径
--wrist_width 640             # 腕部相机宽度
--wrist_height 480            # 腕部相机高度
```

### 操作流程

1. 按 **⏺ REC** → 开始录制
2. 键盘操控机械臂完成 pick-place
3. 按 **⏹ STOP** → 保存 episode，机械臂自动归位，方块随机化
4. 按 **✗ DISCARD**（或 `Z` 键）→ 丢弃当前 episode
5. 重复 → `Q` / `ESC` 退出

每次启动自动递增数据集目录（`so100_sim_1`、`_2`…），旧数据不被覆盖。

### 键盘控制

| 按键 | 功能 |
|------|------|
| ↑↓←→ | 末端 XY 移动（混合直觉坐标系） |
| Shift / Ctrl | 末端 Z 升降 |
| `,` / `.` | 夹爪关 / 开 |
| `Z` | 丢弃当前 episode |
| `Q` / `ESC` | 退出 |

### 输出格式（LeRobot v3）

```
datasets/so100_sim_N/
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet   # 关节状态 + 动作，逐帧
├── videos/
│   └── observation.images.wrist/
│       └── episode_000000.mp4       # 腕部相机，H.264
└── meta/
    ├── info.json                    # fps, features, total_episodes
    ├── stats.json                   # 归一化统计（compute_stats）
    └── tasks.parquet                # 任务索引
```

**录制的特征**：

| 特征 | 形状 | 说明 |
|------|------|------|
| `observation.state` | (6,) float32 | 关节角度（度） |
| `action` | (6,) float32 | 目标关节角度（度） |
| `observation.images.wrist` | (480, 640, 3) uint8 | 腕部相机 RGB |

每次 episode 重置时方块 XY 随机偏移 ±3cm，保证数据多样性。格式直接兼容 `lerobot-train`。

### 训练

```bash
# 生成 stats.json
python -c "
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.compute_stats import compute_stats
dataset = make_dataset(repo_id='budget_pilot/datasets/so100_sim_1')
stats = compute_stats(dataset, num_workers=0, batch_size=8)
stats.save('budget_pilot/datasets/so100_sim_1/stats.json')
"

# 训练 ACT
lerobot-train \
    --dataset.repo_id "budget_pilot/datasets/so100_sim_1" \
    --policy.type act \
    --output_dir outputs/so100_act \
    --steps 50000 \
    --batch_size 8
```

---

## 核心架构（所有程序共用）

- **仅位置 IK** — 3 约束 / 5 自由度，[ikpy](https://github.com/Phylliade/ikpy) 阻尼最小二乘。20 Hz 求解，姿态不约束（实测无漂移）。
- **时间插值** — IK 结果在 50 ms 内线性过渡，消除关节跳动。
- **混合直觉坐标系**（ICRA 2024）— 前向 = 夹爪 Z 轴地面投影。操控方向始终直觉。
- **单循环 + dt 缩放** — 控制与物理同在一个 `while` 循环，末端速度恒定。

---

## `hardware/` — SCS225 舵机桥接

MuJoCo 仿真与物理 SCS225 舵机之间的控制桥接。

| 程序 | 方向 | 说明 |
|------|------|------|
| `hardware/demo_cam_servo.py` | MuJoCo → 舵机 | 键盘遥操作 + 舵机同步跟随 |
| `hardware/demo_servo_track.py` | MuJoCo → 舵机 | LeRobot 录制 + 舵机同步跟随 |
| `hardware/demo_servo_mirror.py` | 舵机 → MuJoCo | 被动回驱 — 手拧舵机，MuJoCo 跟随 |

```bash
python hardware/demo_cam_servo.py      # MuJoCo 控制舵机
python hardware/demo_servo_mirror.py    # 舵机控制 MuJoCo（扭矩关闭）
```

### 控制特性

- **max_relative_target**: 30 steps/frame 限幅 — 防止复位时危险跳跃
- **速度前馈**: 根据位移量自动调节舵机速度
- **定期读回**: 每 500ms 校验舵机跟随精度
- **50Hz 同步率**: 流畅性与总线负载之间的平衡点

### 接线

```
PC → USB → 驱动板 → TTL 总线 → SCS225 舵机
                              ├─ 白: 信号
                              ├─ 红: VCC（6-8.4V 独立供电）
                              └─ 黑: GND
```

### 舵机实测参数

| 指标 | 值 |
|------|-----|
| 稳态精度 | ±2 步（±0.6°） |
| 单次 ReadPos延迟 | ~443μs |
| 6 舵机串行读取 | ~2.7ms（占 60Hz 帧 16%） |
| SyncWrite | ✅ 支持 |
| SyncRead | ❌ 不支持（SCS 协议 v1） |

---

## `traj_viewer/` — 轨迹分析工具

`.npz` 录制数据离线可视化（来自 `demo_cam.py` / `demo_basic.py`）。不兼容 LeRobot 数据集，后者请用 `lerobot-dataset-viz`。

```bash
python traj_viewer/main.py                    # GUI — 浏览选择轨迹
python traj_viewer/main.py <file.npz>         # CLI — 快速输出统计
```

**功能**：关节角度曲线、EE 3D 轨迹图、机械臂动画、多轨迹对比、统计弹窗。

---

## 环境要求

- Python ≥ 3.10
- MuJoCo ≥ 3.0
- ikpy ≥ 3.4
- NumPy ≥ 1.26
- opencv-python ≥ 4.0
- tkinter（Python 自带）
- [lerobot](https://github.com/huggingface/lerobot) — `demo_lerobot_record.py` 需要
- [draccus](https://github.com/dlwh/draccus) — `demo_lerobot_record.py` 需要
- pandas, av, scipy

---

## 文件结构

```
so100-budget-pilot/
├── demo_lerobot_record.py     # ★ 推荐 — 遥操作 + LeRobot 数据集
├── demo_cam.py                # 腕部相机 + MuJoCo 窗口 + .npz
├── demo_basic.py              # 基础遥操作
├── replay.py                  # 轨迹回放器（.npz）
├── traj_viewer/               # 离线分析工具
├── hardware/                  # SCS225 舵机桥接（MuJoCo ↔ 硬件）
│   ├── demo_cam_servo.py      #   键盘遥操作 + 舵机跟随
│   ├── demo_servo_track.py    #   LeRobot 录制 + 舵机跟随
│   └── demo_servo_mirror.py   #   舵机 → MuJoCo 镜像（主端模式）
├── so100_fk.py                # 正运动学（纯 NumPy）
├── so100_ik.py                # 逆运动学（ikpy DLS）
├── requirements.txt
├── model/
│   ├── so100_pick_place.xml   # MuJoCo 场景（桌面 + 方块）
│   ├── so_arm100.xml          # SO-ARM100 机械臂模型
│   └── assets/                # 网格文件 (.stl)
├── recordings/                # .npz 轨迹文件
└── datasets/                  # LeRobot 数据集（gitignored）
```

## 参考文献

- [ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705) — Zhao et al., 2023
- [What Matters in Learning from Offline Human Demonstrations](https://arxiv.org/abs/2108.03298) — Mandlekar et al., 2021
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246) — Kim et al., 2024
- [π₀: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164) — Black et al., 2024
- [iFlyBot-VLA Technical Report](https://arxiv.org/abs/2511.01914) — Zhang et al., 2025
