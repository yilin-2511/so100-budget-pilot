# SO-ARM100 遥操作 — MuJoCo 仿真

在 MuJoCo 中操控 SO-ARM100 机械臂。仅位置 IK + 混合直觉坐标系 + 时间插值。
暗色主题 tkinter UI。支持 `.npz` + LeRobot 数据集录制。

## 快速开始

```bash
conda create -n so100 python=3.11 -y
conda activate so100
pip install -r requirements.txt

# 运行（任选其一）
python demo_lerobot_record.py # ★ 推荐 — 遥操作 + LeRobot 数据集录制
python demo_cam.py            # 腕部相机 + MuJoCo 窗口 + .npz 录制
python demo_basic.py          # 基础版 — tkinter 面板 + 键盘
python replay.py              # 回放已录制轨迹 (.npz)
```

## 程序说明

| 程序 | 说明 |
|------|------|
| `demo_lerobot_record.py` | **★ 推荐** — 遥操作 + LeRobotDataset v3 录制（MP4 + Parquet），draccus 命令行配置，暗色主题 UI |
| `demo_cam.py` | 腕部相机 + MuJoCo 3D 窗口，暗色主题 UI，.npz 录制 |
| `demo_basic.py` | 基础遥操作 — tkinter 面板 + 键盘，.npz 录制 |
| `replay.py` | 轨迹回放器 — 扫描 `recordings/`，列表选择后物理回放 |
| [`traj_viewer/`](traj_viewer/) | 离线分析 — 关节曲线、EE 3D 图、机械臂动画、多轨迹对比 |

---

## demo_lerobot_record.py — LeRobot 数据集录制 ★

键盘遥操作 + 直接写入 LeRobotDataset v3。输出 MP4 视频 + Parquet 关节数据，可直接用于 ACT/Diffusion 训练。所有参数通过 draccus 命令行配置。

### 快速运行

```bash
python demo_lerobot_record.py                          # 默认：10 FPS，方块 ±3cm 随机
python demo_lerobot_record.py --record_fps 20          # 20 FPS
python demo_lerobot_record.py --cube_random_xy 0.05    # 方块随机范围 ±5cm
python demo_lerobot_record.py --help                   # 查看所有可配参数
```

### 可配置参数

```bash
--pos_speed 0.15              # EE 移动速度（m/s，默认 0.10）
--record_fps 20               # 录制帧率（默认 10）
--cube_random_xy 0.05         # 方块 XY 随机偏移范围（m，默认 0.03）
--episode_max_duration 60.0   # 单集最长秒数（默认 120）
--target_episodes 50          # 录满 N 集自动退出（默认 0 = 无限）
--dataset_root datasets/my_data  # 自定义数据集路径
--wrist_width 640             # 腕部相机宽度
--wrist_height 480            # 腕部相机高度
```

### 操作流程

1. 按 **⏺ REC** → 开始录制
2. 键盘操控机械臂完成 pick-place
3. 按 **⏹ STOP** → 保存 episode，机械臂自动回 HOME，方块随机化
4. 按 **✗ DISCARD**（或 Z 键）→ 丢弃当前 episode
5. 重复至完成 → Q/ESC 退出

每次启动自动递增数据集目录（`datasets/so100_sim_1`、`_2`…），旧数据不被覆盖。

### 键盘控制

| 按键 | 功能 |
|------|------|
| ↑↓←→ | 末端 XY 移动（混合直觉坐标系） |
| Shift / Ctrl | 末端 Z 升降 |
| `,` / `.` | 夹爪关/开 |
| Z | 丢弃当前 episode |
| R | 切换 EE / JOINT 模式 |
| Q / ESC | 退出 |

### 录制规格

| 参数 | 值 |
|------|-----|
| FPS | 可配（默认 10，推荐 20–30） |
| 分辨率 | 640 × 480 |
| 视频 | MP4（SVT-AV1，流式编码） |
| 特征 | `observation.state`（6 关节,度）+ `action`（6 关节,度）+ `observation.images.wrist` |
| 方块随机化 | 可配（默认 ±3 cm XY） |
| 格式 | LeRobot v3.0 — 直接用于 `lerobot-train` |

### 输出结构

```
datasets/so100_sim_N/
├── data/         # Parquet — 关节状态 & 动作
├── videos/       # MP4 — 腕部相机
└── meta/         # info.json, stats.json, tasks.parquet
```

### 训练

```bash
lerobot-train \
    --dataset.repo_id "budget_pilot/datasets/so100_sim_1 budget_pilot/datasets/so100_sim_2 ..." \
    --policy.type act \
    --output_dir outputs/so100_act \
    --steps 50000
```

---

## demo_cam.py — 腕部相机 + MuJoCo 窗口

键盘遥操作 + MuJoCo 3D 场景窗口 + 腕部相机窗口。.npz 录制。

### 键盘控制

| 按键 | 功能 |
|------|------|
| ↑↓←→ | 末端 XY 移动（混合直觉坐标系） |
| Shift / Ctrl | 末端 Z 升降 |
| `,` / `.` | 夹爪关/开 |

### UI 布局（暗色主题）

| 区域 | 内容 |
|------|------|
| **顶部栏** | 模式指示灯（● EE 蓝色 / ● JOINT 紫色）+ 快捷键提示 |
| **末端控制** | 实际/目标位置实时显示 |
| **关节控制** | 5 关节 — ± 按钮 + 实时角度值 |
| **底部** | ⏺ REC（录制 .npz）/ ↺ RESET / 状态 |

### 控制模式

- **EE 模式**（默认，蓝色）：方向键移动末端。仅位置 IK，20 Hz 求解，50 ms 线性插值。末端速度 = 100 mm/s。
- **Joint 模式**（紫色）：点击关节 ± 按钮。直接关节空间控制，速度 1.0 rad/s。切回 EE 时锁定当前位姿。
- **夹爪**：`,` / `.` 键，不受模式限制，速度 1.0 rad/s。

### 腕部相机

- 渲染 `cam_wrist` 至独立 OpenCV 窗口
- 离屏渲染 960×720，~15 FPS
- 夹爪与操作区域第一人称视角

---

## demo_basic.py — 基础遥操作

相同控制引擎，更简洁 UI。方向键控制 EE XY。tkinter 面板含 EE XYZ 按钮、关节 ± 按钮、夹爪 ± 按钮、REC / HOME。

---

## replay.py — 轨迹回放

- 扫描 `recordings/` 中的 `.npz` 文件，按时间倒序
- tkinter 选择界面：点选轨迹 → 播放
- 完整物理回放，方块可被碰撞推动
- 播放结束后返回选择界面

---

## 核心架构（所有程序共用）

- **仅位置 IK**：3 约束 / 5 自由度 — 快速稳定。20 Hz 高频求解，姿态漂移可忽略。
- **时间插值**：IK 结果在 50 ms 内线性过渡，消除关节跳动。
- **混合直觉坐标系**（ICRA 2024）：前 = 夹爪 Z 轴地面投影。操控方向始终直觉。
- **单循环 + dt 缩放**：控制与物理同在一个 `while` 循环，末端速度恒定，不受帧率波动影响。

---

## 文件结构

```
so100-budget-pilot/
├── demo_lerobot_record.py     # ★ 推荐 — 遥操作 + LeRobot 数据集录制
├── demo_cam.py                # 腕部相机 + MuJoCo 窗口 + .npz
├── demo_basic.py              # 基础遥操作
├── replay.py                  # 轨迹回放器（.npz）
├── traj_viewer/               # 离线分析工具
├── so100_fk.py                # 正运动学（纯 NumPy）
├── so100_ik.py                # 逆运动学（ikpy）
├── requirements.txt           # Python 依赖
├── model/
│   ├── so100_pick_place.xml   # MuJoCo 场景（桌面 + 方块）
│   ├── so_arm100.xml          # SO-ARM100 机械臂模型
│   └── assets/                # 网格文件 (.stl)
├── recordings/                # 录制的轨迹 (.npz)
└── datasets/                  # LeRobot 数据集（gitignored）
```

## 环境要求

- Python ≥ 3.10
- MuJoCo ≥ 3.0
- ikpy ≥ 3.4
- NumPy ≥ 1.26
- opencv-python ≥ 4.0
- lerobot（`demo_lerobot_record.py` 需要）
- draccus（`demo_lerobot_record.py` 需要）
- pandas, av, scipy
- tkinter（Python 自带）

## 轨迹格式 (.npz)

```
[时间, qpos(6), ctrl(6), ee_x, ee_y, ee_z, ee_qw, ee_qx, ee_qy, ee_qz]
```

回放：`python replay.py` → 列表选择 → 播放。
