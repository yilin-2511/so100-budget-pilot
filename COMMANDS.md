# Pipeline 命令行速查手册

> 最后更新：2026-07-11
> Pipeline: 录制 → 打包 → 上传 → GPU 训练 → 下载 → 评估

---

## 一、录制数据

```powershell
# 本地 — conda robot_sim 环境
D:/conda/envs/robot_sim/python.exe D:/robot/budget_pilot/demo_lerobot_record.py

# 可选参数
--record_fps 20                # 录制帧率（默认 10）
--cube_random_xy 0.05          # 方块随机范围 m（默认 0.03）
--target_episodes 50           # 录满自动退出（默认 0=无限）
```

数据存于 `d:\robot\budget_pilot\datasets\so100_sim\`，重复运行自动递增 `so100_sim_2`, `_3`...

---

## 二、打包 + 上传 AutoDL

```powershell
# 打包（本地）
cd D:\robot\budget_pilot
tar -czf so100_data.tar.gz datasets/so100_sim/

# 上传（本地 PowerShell）
scp -P <端口> D:/robot/budget_pilot/so100_data.tar.gz root@connect.nmb1.seetacloud.com:/root/autodl-tmp/
```

---

## 三、AutoDL 环境配置（SSH 连接后）

```bash
# 解压
cd /root/autodl-tmp
tar -xzf so100_data.tar.gz

# 安装依赖（一次性）
pip install 'lerobot[training]'
apt install -y libosmesa6          # 无头渲染（eval 录屏需要）
```

---

## 四、GPU 训练

```bash
# ACT（推荐）
nohup lerobot-train \
  --dataset.repo_id=so100_sim \
  --dataset.root=/root/autodl-tmp/datasets/so100_sim \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.repo_id=local/act_so100 \
  --batch_size=64 \
  --steps=20000 \
  --num_workers=8 \
  --output_dir=/root/autodl-tmp/outputs/act_so100 \
  --log_freq=100 \
  --save_freq=5000 \
  --seed=42 \
  > /root/autodl-tmp/train.log 2>&1 &

# 查看进度
tail -f /root/autodl-tmp/train.log
```

---

## 五、上传评估脚本 + 模型文件

```powershell
# 本地 PowerShell — 上传 eval + MuJoCo 模型
scp -P <端口> D:/robot/budget_pilot/eval_policy_headless.py root@connect.nmb1.seetacloud.com:/root/autodl-tmp/
scp -r -P <端口> D:/robot/budget_pilot/model root@connect.nmb1.seetacloud.com:/root/autodl-tmp/
```

---

## 六、AutoDL 评估

```bash
# 基础评估（纯日志）
MUJOCO_GL=egl python eval_policy_headless.py \
  --checkpoint ./outputs/act_so100/checkpoints/005000/pretrained_model \
  --episodes 20

# 带录屏
MUJOCO_GL=egl python eval_policy_headless.py \
  --checkpoint ./outputs/act_so100/checkpoints/005000/pretrained_model \
  --episodes 20 --record
```

---

## 七、泛化测试（多颜色）

```bash
# AutoDL — 6 色可选: red / blue / green / yellow / white / black
MUJOCO_GL=egl python eval_policy_headless.py \
  --checkpoint ./outputs/act_so100/checkpoints/005000/pretrained_model \
  --episodes 20 --cube_color blue --record
```

---

## 八、下载 checkpoint + 视频

```powershell
# 本地 PowerShell
scp -r -P <端口> root@connect.nmb1.seetacloud.com:/root/autodl-tmp/outputs/act_so100/checkpoints/005000 D:\robot\budget_pilot\act_checkpoint_5000

# 下载录屏
scp -r -P <端口> root@connect.nmb1.seetacloud.com:/root/autodl-tmp/recordings/<时间戳> D:/robot/budget_pilot/recordings/
```

---

## 九、本地评估

```powershell
# 渲染版（MuJoCo 窗口 + 相机画面）
D:/conda/envs/robot_sim/python.exe D:/robot/budget_pilot/eval_policy.py \
  --checkpoint D:/robot/budget_pilot/act_checkpoint_5000/pretrained_model \
  --episodes 10 --render

# 无头版（纯日志）
D:/conda/envs/robot_sim/python.exe D:/robot/budget_pilot/eval_policy_headless.py \
  --checkpoint D:/robot/budget_pilot/act_checkpoint_5000/pretrained_model \
  --episodes 10
```

---

## 十、GitHub 提交

```powershell
cd D:\robot\budget_pilot
D:/Git/bin/git.exe add <files>
D:/Git/bin/git.exe commit -m "<message>"
D:/Git/bin/git.exe push origin master
```

---

## 结果速查

| 指标 | 值 |
|------|-----|
| 训练数据 | 47 episodes, 9209 frames, 10 FPS |
| 模型 | ACT, 52M params |
| 训练 | 4090, 5000 steps, batch_size=64 |
| 红色方块 | **40%** |
| 蓝色方块（泛化） | **70%** |
| Loss | 47 → 0.153 |
