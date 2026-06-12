# SO-ARM100 桌面遥操作（预算版）

用键盘 + 鼠标在 MuJoCo 仿真中操控 SO-ARM100 机械臂。仅位置 IK + 混合直觉坐标系 + 时间插值。

## 快速开始

```bash
# 1. 创建 conda 环境
conda create -n so100 python=3.11 -y
conda activate so100

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行（任选其一）
python demo_sota.py      # 基础版 — tkinter 面板 + 键盘控制
python demo_ui_v2.py     # 增强版 — 重新设计 UI + 腕部相机
python replay.py          # 轨迹回放，带选择界面
```

## 程序说明

| 程序 | 说明 |
|------|------|
| `demo_sota.py` | 基础遥操作：tkinter 控制面板 + 键盘，支持轨迹录制 |
| `demo_ui_v2.py` | **增强版**：重新设计 UI，新增模式指示灯、速度滑块、关节角度实时显示、腕部相机 |
| `replay.py` | 轨迹回放器：扫描 `recordings/` 中的轨迹，列表选择后物理回放 |

## 操控方式

### 键盘（MuJoCo 窗口在前台也能用）

| 按键 | 功能 |
|------|------|
| ↑ ↓ ← → | 前后左右移动末端（混合直觉坐标系） |

### tkinter 控制面板

| 区域 | 控件 | 功能 |
|------|------|------|
| 末端控制 | XY 方向键、Z 升降按钮 | 移动末端位置 |
| 关节控制 | 5 个关节的 +/− 按钮 | 直接调关节角 |
| 夹爪 | +/− 按钮 | 夹爪开合 |
| 底部 | REC / HOME | 录制轨迹 / 复位 |

### 两种模式

- **EE 模式**（默认）：点任意 EE 按钮进入。仅位置 IK 驱动机器臂，混合直觉坐标系方向随夹爪朝向自适态。
- **Joint 模式**：点任意关节按钮进入。直接关节空间控制。切回 EE 模式时自动锁当前位姿。
- **夹爪**：始终可用，不受 EE/Joint 模式限制。

## demo_ui_v2.py — 增强 UI 与腕部相机

重新设计控制面板，提供实时视觉反馈：

- **模式指示灯**：EE 模式（蓝色）/ Joint 模式（绿色），一目了然
- **速度滑块**：末端移动速度 50–300 mm/s 可调
- **关节角度显示**：5 个关节角的实时数值显示
- **EE 按钮颜色区分**：方向按钮用颜色区分，防误触
- **键盘快捷键提示**：面板内嵌快捷键参考
- **HOME 二次确认**：弹窗确认，防止误触复位

**腕部相机**：将内置 `cam_wrist` 相机画面渲染至 OpenCV 窗口（~15 FPS），提供夹爪与操作区域的第一人称视图。

## replay.py — 轨迹回放

- 扫描 `recordings/` 中的 `.npz` 轨迹文件，按时间倒序排列
- tkinter 选择界面：点选轨迹即可开始回放
- 完整物理交互回放：物块可被碰撞、推动
- 回放结束后返回选择界面，可继续选下一个

## 核心特性

- **仅位置 IK**：3 个约束 / 5 自由度 —— 快速稳定。夹爪姿态由求解器贴近初始猜测自然保持。
- **混合直觉坐标系**（ICRA 2024）：前 = 夹爪 Z 轴地面投影，左 = 垂直前方向，上 = 世界 Z。方向感直觉，不随夹爪朝向混乱。
- **单循环架构**：控制与物理同步在一个 `while` 循环中 —— 无抖动。
- **dt 速度缩放**：末端速度恒定 100mm/s，不受仿真帧率影响。
- **时间插值**：IK 解算结果 50ms 线性过渡到控制器，平滑不跳。
- **轨迹录制**：命名保存为 `.npz` 文件，存在 `recordings/` 目录。

## 文件结构

```
so100-budget-pilot/
├── demo_sota.py              # 基础遥操作
├── demo_ui_v2.py             # 增强遥操作（UI v2 + 腕部相机）
├── replay.py                 # 轨迹回放器（带选择界面）
├── so100_fk.py               # 正运动学（纯 NumPy）
├── so100_ik.py               # 逆运动学（ikpy）
├── __init__.py               # 模块初始化
├── requirements.txt          # Python 依赖
├── README.md                 # 英文说明
├── README_CN.md              # 中文说明（本文）
├── model/
│   ├── so100_pick_place.xml   # MuJoCo 场景（桌面 + 方块）
│   ├── so_arm100.xml          # SO-ARM100 机械臂模型
│   └── assets/                # 网格文件 (.stl)
└── recordings/               # 录制的轨迹 (.npz)
```

## 环境要求

- Python ≥ 3.10
- MuJoCo ≥ 3.0
- ikpy ≥ 3.4
- NumPy ≥ 1.26
- opencv-python ≥ 4.0（`demo_ui_v2.py` 腕部相机需要）
- tkinter（Python 自带）

## 录制与回放

点击 **REC** → 输入轨迹名称 → 操作机械臂 → 点击 **STOP**。

轨迹保存在 `recordings/` 目录，格式为 `.npz`。每帧 20 列：

```
[时间, qpos(6), ctrl(6), ee_x, ee_y, ee_z, ee_qw, ee_qx, ee_qy, ee_qz]
```

回放命令：`python replay.py`，从列表中点选轨迹即可播放。

## 致谢

- MuJoCo: DeepMind 物理仿真引擎
- ikpy: 逆运动学求解库
- Hybrid Intuitive Frame: ICRA 2024 论文方案
- tutorial_for_mujoco: CSDN 博客《MuJoCo 全流程实战教程》
