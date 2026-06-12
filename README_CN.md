# SO-ARM100 桌面遥操作（廉价版）

在 MuJoCo 仿真中操控 SO-ARM100 机械臂。仅位置 IK + 混合直觉坐标系 + 时间插值。

## 快速开始

```bash
conda create -n so100 python=3.11 -y
conda activate so100
pip install -r requirements.txt

# 运行（任选其一）
python demo_cam.py       # 推荐 — 腕部相机 + 增强 UI
python demo_basic.py     # 基础版 — tkinter 面板 + 键盘控制
python replay.py         # 回放已录制的轨迹
```

## 程序说明

| 程序 | 说明 |
|------|------|
| `demo_cam.py` | **推荐** — 腕部相机、末端位置显示、关节角度实时显示、重新设计 UI |
| `demo_basic.py` | 基础遥操作 — tkinter 控制面板 + 键盘，支持轨迹录制 |
| `replay.py` | 轨迹回放器 — 扫描 `recordings/`，列表选择后物理回放 |
| [`traj_viewer/`](traj_viewer/) | **离线分析** — 关节曲线、末端 3D 图、机械臂动画、多轨迹对比（无需 MuJoCo） |

---

## demo_cam.py — 腕部相机与增强 UI

### 键盘快捷键（全局 — MuJoCo 窗口在前台也能用）

| 按键 | 功能 |
|------|------|
| **方向键** ↑ ↓ ← → | 末端 XY 方向移动（混合直觉坐标系） |
| **Shift** | 末端上升 (+Z) |
| **Ctrl** | 末端下降 (−Z) |
| **<**（逗号） | 夹爪闭合 |
| **>**（句号） | 夹爪张开 |

快捷键提示显示在 tkinter 顶部栏中。

### tkinter 控制面板

| 区域 | 内容 |
|------|------|
| **顶部栏** | 模式指示灯（● EE 蓝色 / ● JOINT 紫色）+ 键盘快捷键提示 |
| **末端控制** | 方向按钮（+X/−X/+Y/−Y/+Z/−Z）+ 实际/目标位置实时显示 |
| **关节控制** | 5 关节（Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll）— ± 按钮 + 实时角度值 |
| **底部** | ⏺ REC（录制轨迹）/ RESET（复位）/ 状态栏 |

### 控制模式

- **EE 模式**（默认，蓝色指示灯）：按任意 EE 方向按钮或方向键触发。仅位置 IK 通过混合直觉坐标系驱动机械臂。IK 以 20 Hz 求解、50 ms 线性插值，运动平滑。末端速度 = 100 mm/s。
- **Joint 模式**（紫色指示灯）：按任意关节 ± 按钮进入。直接关节空间控制，速度 1.0 rad/s。切回 EE 模式时自动锁定当前位姿为新的 IK 目标。
- **夹爪**：仅通过键盘 **<** / **>** 控制，不受 EE/Joint 模式限制，速度 1.0 rad/s。

### 腕部相机

- 将内置 `cam_wrist` 相机渲染至独立 OpenCV 窗口（"Wrist Camera"）
- 离屏渲染分辨率 960×720，显示帧率 ~15 FPS
- 提供夹爪与操作区域的第一人称视角

### 其他特性

- **末端位置显示**：实际位置（物理引擎）vs 目标位置（IK 解算）并列显示，单位 mm
- **关节角度显示**：5 个关节角的实时数值，单位弧度
- **俯视相机**：MuJoCo 窗口设为桌面俯瞰视角
- 轨迹录制：**REC** → 弹窗命名 → 操作 → **STOP** → 保存 `.npz` 至 `recordings/`

---

## demo_basic.py — 基础遥操作

相同控制引擎，更简洁的 UI：

| 键盘按键 | 功能 |
|----------|------|
| 方向键 ↑ ↓ ← → | 末端 XY 方向移动（混合坐标系） |

tkinter 面板包含 EE XYZ 按钮、关节 ± 按钮、夹爪 ± 按钮、REC / HOME。

---

## replay.py — 轨迹回放

- 扫描 `recordings/` 中的 `.npz` 文件，按时间倒序排列
- tkinter 选择界面：点选轨迹 → 播放
- 完整物理交互回放：物块可被碰撞、推动
- 播放结束后返回选择界面

---

## 核心特性（所有程序共用）

- **仅位置 IK**：3 个约束 / 5 自由度 — 快速稳定。20 Hz 高频求解，姿态几乎不漂。
- **时间插值**：IK 结果在 50 ms 内线性过渡，即使解算结果有跳动，关节运动依然平滑。
- **混合直觉坐标系**（ICRA 2024）：前 = 夹爪 Z 轴地面投影。操控方向始终直觉，不随夹爪朝向改变。
- **单循环 + dt 缩放**：控制与物理同在一个 `while` 循环，末端速度恒定 100 mm/s，不受帧率波动影响。

> **为什么手感这么好？** 仅位置 IK（约束少、解算快）+ 时间插值（消纳 IK 跳动）+ 混合坐标系（方向直觉）+ 单循环（无多线程抖动）。

---

## 文件结构

```
so100-budget-pilot/
├── demo_cam.py               # 推荐 — 腕部相机 + 增强 UI
├── demo_basic.py              # 基础遥操作
├── replay.py                  # 轨迹回放器（带选择界面）
├── traj_viewer/               # 离线分析工具（关节曲线、EE 3D、动画）
├── so100_fk.py                # 正运动学（纯 NumPy）
├── so100_ik.py                # 逆运动学（ikpy）
├── __init__.py                # 模块初始化
├── requirements.txt           # Python 依赖
├── model/
│   ├── so100_pick_place.xml   # MuJoCo 场景（桌面 + 方块）
│   ├── so_arm100.xml          # SO-ARM100 机械臂模型
│   └── assets/                # 网格文件 (.stl)
└── recordings/                # 录制的轨迹 (.npz)
```

## 环境要求

- Python ≥ 3.10
- MuJoCo ≥ 3.0
- ikpy ≥ 3.4
- NumPy ≥ 1.26
- opencv-python ≥ 4.0（`demo_cam.py` 腕部相机需要）
- tkinter（Python 自带）

## 轨迹格式

录制的 `.npz` 文件每帧包含 20 列：

```
[时间, qpos(6), ctrl(6), ee_x, ee_y, ee_z, ee_qw, ee_qx, ee_qy, ee_qz]
```

回放：`python replay.py` → 列表选择 → 播放。

## 致谢

- MuJoCo: DeepMind 物理仿真引擎
- ikpy: 逆运动学求解库
- Hybrid Intuitive Frame: ICRA 2024 论文方案
