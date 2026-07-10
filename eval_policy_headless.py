"""SO-ARM100 Policy Evaluation — Headless (AutoDL / no GUI).

Usage:
  python eval_policy_headless.py --checkpoint <path> --episodes 20
"""
import os, sys, time, argparse, json
import numpy as np
import mujoco
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, "model", "so100_pick_place.xml")
HOME_CTRL = [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]
SETTLE_STEPS = 500     # allow cube to fully settle on table
MAX_STEPS_DEFAULT = 200     # policy queries per episode
ACTION_REPEAT = 50          # physics steps per policy action (10 FPS @ 0.002s timestep)

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
def init_scene():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    model.dof_damping[:] = 3.0
    mujoco.mj_resetData(model, data)
    cube_qpos = data.qpos[6:13].copy()
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[6:13] = cube_qpos
    mujoco.mj_forward(model, data)
    for _ in range(SETTLE_STEPS):
        data.ctrl[:6] = HOME_CTRL
        mujoco.mj_step(model, data)
    ids = {
        "cube": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube"),
        "wrist_cam": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wrist"),
        "overhead_cam": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_overhead"),
    }
    return model, data, ids


def reset_episode(model, data, randomize_xy=0.03):
    mujoco.mj_resetData(model, data)
    cube_qpos = data.qpos[6:13].copy()
    mujoco.mj_resetDataKeyframe(model, data, 0)
    if randomize_xy > 0:
        cube_qpos[0] += np.random.uniform(-randomize_xy, randomize_xy)
        cube_qpos[1] += np.random.uniform(-randomize_xy, randomize_xy)
    data.qpos[6:13] = cube_qpos
    mujoco.mj_forward(model, data)
    for _ in range(SETTLE_STEPS):
        data.ctrl[:6] = HOME_CTRL
        mujoco.mj_step(model, data)


def check_success(data, cube_body_id):
    target_xy = np.array([-0.08, -0.48])
    return np.linalg.norm(data.xpos[cube_body_id][:2] - target_xy) < 0.025


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
def load_policy(checkpoint_path, device="cpu"):
    """Load ACT policy + normalizer stats from checkpoint."""
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    from lerobot.policies.act.modeling_act import ACTPolicy
    import json, safetensors.torch
    policy = ACTPolicy.from_pretrained(checkpoint_path, device=device)
    policy.eval()
    # Enable temporal ensemble (GPU is fast enough)
    if policy.config.temporal_ensemble_coeff is None:
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
        policy.config.temporal_ensemble_coeff = 0.01
        policy.temporal_ensembler = ACTTemporalEnsembler(
            policy.config.temporal_ensemble_coeff, policy.config.chunk_size)
    ns = safetensors.torch.load_file(
        f"{checkpoint_path}/policy_preprocessor_step_3_normalizer_processor.safetensors",
        device="cpu")
    us = safetensors.torch.load_file(
        f"{checkpoint_path}/policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        device="cpu")
    state_norm = {"mean": ns.get("observation.state.mean", 0.0), "std": ns.get("observation.state.std", 1.0)}
    action_norm = {"mean": us.get("action.mean", 0.0), "std": us.get("action.std", 1.0)}
    print(f"[POLICY] Loaded ACT from {checkpoint_path}")
    return policy, state_norm, action_norm


# ---------------------------------------------------------------------------
# Observation (headless — no cv2 display)
# ---------------------------------------------------------------------------
def capture_obs(data, wrist_renderer, wrist_cam_id, overhead_renderer, overhead_cam_id):
    qpos_deg = np.rad2deg(data.qpos[:6]).astype(np.float32)
    wrist_renderer.update_scene(data, camera=wrist_cam_id)
    wrist_img = wrist_renderer.render()
    overhead_renderer.update_scene(data, camera=overhead_cam_id)
    overhead_img = overhead_renderer.render()
    return {
        "observation.state": qpos_deg,
        "observation.images.wrist": wrist_img,
        "observation.images.overhead": overhead_img,
    }


def obs_to_tensors(obs_dict, state_norm, device="cpu"):
    batch = {}
    raw = obs_dict["observation.state"].astype(np.float32)
    mean = np.array(state_norm["mean"])
    std = np.array(state_norm["std"])
    state = torch.from_numpy((raw - mean) / (std + 1e-8)).float().unsqueeze(0).to(device)
    for key in ["observation.images.wrist", "observation.images.overhead"]:
        img = obs_dict[key]
        img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        img_t = img_t.unsqueeze(0).to(device)
        batch[key] = img_t
    batch["observation.state"] = state
    return batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cube_random_xy", type=float, default=0.03)
    parser.add_argument("--record", action="store_true", help="Save rollout videos to recordings/")
    parser.add_argument("--cube_color", type=str, default="red",
                        choices=["red", "blue", "green", "yellow", "white", "black"],
                        help="Cube color for generalization test (default: red)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")

    model, data, ids = init_scene()
    cube_body_id = ids["cube"]
    wrist_cam_id = ids["wrist_cam"]

    COLORS = {
        "red": [0.95, 0.15, 0.15, 1.0],
        "blue": [0.15, 0.35, 0.85, 1.0],
        "green": [0.15, 0.85, 0.35, 1.0],
        "yellow": [0.95, 0.85, 0.15, 1.0],
        "white": [0.95, 0.95, 0.95, 1.0],
        "black": [0.15, 0.15, 0.15, 1.0],
    }
    if args.cube_color != "red":
        cube_mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "cube_red")
        model.mat_rgba[cube_mat_id] = COLORS[args.cube_color]
        print(f"[SCENE] Cube color: {args.cube_color.upper()} (generalization test)")
    overhead_cam_id = ids["overhead_cam"]

    wrist_renderer = mujoco.Renderer(model, 480, 640)
    overhead_renderer = mujoco.Renderer(model, 240, 320)
    scene_renderer = mujoco.Renderer(model, 480, 640) if args.record else None
    if args.record:
        import cv2
        run_id = time.strftime("%m%d_%H%M%S")
        os.makedirs(f"recordings/{run_id}", exist_ok=True)

    policy, state_norm, action_norm = load_policy(args.checkpoint, device=device)

    successes = []
    step_counts = []
    t_start = time.perf_counter()

    for ep in range(args.episodes):
        reset_episode(model, data, randomize_xy=args.cube_random_xy)
        policy.reset()

        # Recording per episode
        out = None
        if args.record:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(f"recordings/{run_id}/ep_{ep:03d}.mp4", fourcc, 10, (640, 480))

        # 2-second delay: arm stays at home, cube fully settles
        for _ in range(int(2.0 / 0.002)):
            data.ctrl[:6] = HOME_CTRL
            mujoco.mj_step(model, data)

        done = False
        step = 0
        while not done and step < args.max_steps:
            obs = capture_obs(data, wrist_renderer, wrist_cam_id, overhead_renderer, overhead_cam_id)
            batch = obs_to_tensors(obs, state_norm, device=device)
            with torch.inference_mode():
                action = policy.select_action(batch)
            a_normed = action.squeeze(0).cpu()
            mean_a = np.array(action_norm["mean"]); std_a = np.array(action_norm["std"])
            action_deg = (a_normed.numpy() * std_a) + mean_a
            data.ctrl[:6] = np.deg2rad(action_deg.astype(np.float64))
            for _ in range(ACTION_REPEAT):
                mujoco.mj_step(model, data)
            # Record 1 frame per policy step (10 FPS → video matches training data)
            if out is not None:
                scene_renderer.update_scene(data, camera=overhead_cam_id)
                out.write(cv2.cvtColor(scene_renderer.render(), cv2.COLOR_RGB2BGR))
            step += 1
            if check_success(data, cube_body_id):
                done = True
                successes.append(True)
                step_counts.append(step)
        if out is not None:
            out.release()

        if not done:
            successes.append(False)
            step_counts.append(step)

        elapsed = time.perf_counter() - t_start
        rate = sum(successes) / (ep + 1) * 100
        print(f"[EP {ep+1:2d}/{args.episodes}] "
              f"{'SUCCESS' if done else 'FAIL'} in {step:3d} steps | "
              f"success rate: {rate:.0f}% | "
              f"elapsed: {elapsed:.0f}s")

    # Report
    total = len(successes)
    n_success = sum(successes)
    print(f"\n{'='*50}")
    print(f"Results: {n_success}/{total} ({n_success/total*100:.1f}%)")
    if n_success > 0:
        avg_steps = np.mean([s for i, s in enumerate(step_counts) if successes[i]])
        print(f"Avg steps (success): {avg_steps:.0f}")
    print(f"Total time: {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
