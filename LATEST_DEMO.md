# Demo — SO-ARM100 LeRobot Dataset Recording

> 最后更新：2026-06-14

## 程序

**文件**: `d:\robot\experiments\bc_training\demo_lerobot_record.py`

键盘遥操作 + 直接写入 LeRobotDataset（跳过 .npz 中转），适配 Intel Iris Xe 集成显卡。

### 运行

```powershell
D:\conda\envs\robot_sim\python.exe D:\robot\experiments\bc_training\demo_lerobot_record.py
```

### 界面

深色主题，1080×880，三区域布局：

```
┌──────────────────────────────────────────┐
│  ● EE    ↑↓←→=XY  Shift/Ctrl=Z  ...     │  顶栏：模式 + 快捷键提示
├──────────────┬───────────────────────────┤
│ END-EFFECTOR │  JOINTS                   │
│ Pos  [x,y,z] │  − +  Rotation   +0.00   │
│ Tgt  [x,y,z] │  − +  Pitch      +0.00   │
│              │  − +  Elbow      +0.00   │
│ DATASET      │  − +  Wrist_P    +0.00   │
│ Name  sim_14 │  − +  Wrist_R    +0.00   │
│ Eps   3      │                           │
│ Frames 4521  │                           │
├──────────────┴───────────────────────────┤
│  [⏺ REC]    [✗ DISCARD]    [↺ RESET]    │  底栏
│             Ready     Ep 0               │
└──────────────────────────────────────────┘
```

- **左栏**: EE 实时坐标（Pos=实际, Tgt=目标） + 数据集信息（名称/条数/帧数）
- **右栏**: 5 个关节的 ± 按钮 + 实时角度值
- **底栏**: REC/STOP（录制控制）、DISCARD（丢弃当前集）、RESET（场景重置）、状态文字、episode 计数

### 键盘映射

| 按键 | 功能 |
|------|------|
| ↑↓←→ | EE 前后左右移动（Hybrid Intuitive Frame） |
| Shift / Ctrl | EE 升降（世界 Z 轴） |
| `,` / `.` | 夹爪关/开 |
| Z | 丢弃当前 episode |
| R | 切换 EE / JOINT 模式 |
| Q / ESC | 退出并 finalize 数据集 |

### 操作流程

1. 启动 → 看到 wrist camera 窗口 + tkinter 面板
2. 按 `⏺ REC` → 开始录制（按钮变 `⏹ STOP`，状态变 `● RECORDING: Episode 0`）
3. 键盘操控机械臂完成 pick-place
4. 按 `⏹ STOP` → 保存 episode，自动回 HOME 姿态，方块随机偏移
5. 重复 2-4
6. 按 Q 退出 → `dataset.finalize()` 写入元数据
7. 下次启动自动创建新目录（不覆盖旧数据）

### 录制参数

| 参数 | 值 | 说明 |
|------|-----|------|
| FPS | 10 | lerobot IL-in-sim 标准 |
| 分辨率 | 640×480 | LeRobot 标准 |
| 视频编码 | SVT-AV1 (MP4) | `streaming_encoding=True` |
| 方块随机化 | ±3cm XY | 每次 RESET 改变起始位置 |
| Episode 限制 | 无（手动停止） | `EPISODE_MAX_DURATION=999` |

## 最新录制数据（2026-06-14 下午）

| 数据集 | Episodes | 帧数 | OK | 时长 | 备注 |
|--------|----------|------|-----|------|------|
| so100_sim_17 | 3 | 4809 | 3/3 | 481s | ★ 全优 |
| so100_sim_18 | 1 | 61 | 0/1 | 6s | 废弃（太短） |
| so100_sim_19 | 1 | 1879 | 1/1 | 188s | ★ |
| so100_sim_20 | 4 | 2886 | 1/4 | 289s | 3条夹爪未动 |
| so100_sim_21 | 1 | 1284 | 1/1 | 128s | ★ |
| so100_sim_23 | 1 | 1410 | 1/1 | 141s | ★ 最新 |

**可用数据**: sim_17 (3eps) + sim_19 (1ep) + sim_21 (1ep) + sim_23 (1ep) = **6 条高质量 episode**

## 与之前数据合并

| 批 | 数据集 | 可用 Eps |
|---|--------|----------|
| 早期 (10fps) | so100_sim_5 | 7 ✓ |
| 早期 | so100_sim_6 | 2 ✓ |
| 早期 | so100_sim_9 | 1 ✓ |
| 早期 | so100_sim_10 | 1 ✓ |
| 早期 | so100_sim_13 | 2 ✓ |
| 最新 | so100_sim_{17,19,21,23} | 6 ✓ |
| **总计** | | **19 episodes** |

## 已知问题

- **操控延迟**: Intel Iris Xe 集显渲染瓶颈（`mjr_readPixels` 像素回读），软件层面无法优化。不影响数据质量
- **sim_20 失败率高**: 可能是录制时夹爪未关，需确认操作手法
- **空白目录**: sim_22 等为空，系脚本启动后未录制就关闭

## 下一步

暑假 GTX 1080/AutoDL GPU 上训练：
```bash
lerobot-train --policy.type=act --dataset.root=... --policy.chunk_size=16 ...
```
