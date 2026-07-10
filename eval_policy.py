"""SO-ARM100 Policy Evaluation — Load trained ACT policy, rollout in MuJoCo.

Usage:
  python eval_policy.py --checkpoint outputs/act_so100_sim4/checkpoint-5000
  python eval_policy.py --checkpoint <path> --episodes 20 --max_steps 200 --render
"""
import os, sys, time, argparse
import numpy as np
import mujoco
import mujoco.viewer
import torch
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))

XML_PATH = os.path.join(HERE, "model", "so100_pick_place.xml")
HOME_CTRL = [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]
SETTLE_STEPS = 300
MAX_STEPS_DEFAULT = 200

# ---------------------------------------------------------------------------
# Scene helpers (reused from demo_lerobot_record.py)
# ---------------------------------------------------------------------------
def init_scene():
    """Load MuJoCo scene, settle cube at home. Returns (model, data, body_ids)."""
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
        "ee": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw"),
        "wrist_cam": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wrist"),
        "overhead_cam": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_overhead"),
    }
    return model, data, ids


def reset_episode(model, data, randomize_xy=0.03):
    """Reset arm to home, cube to source zone with XY random offset, settle."""
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
    """Check if cube centre is within ~2.5cm of target zone centre."""
    target_xy = np.array([-0.08, -0.48])
    cube_xy = data.xpos[cube_body_id][:2]
    return np.linalg.norm(cube_xy - target_xy) < 0.025


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------
def load_policy(checkpoint_path, device="cpu"):
    """Load ACT policy from a lerobot checkpoint directory."""
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    from lerobot.policies.act.modeling_act import ACTPolicy
    policy = ACTPolicy.from_pretrained(checkpoint_path, device=device)
    policy.eval()
    print(f"[POLICY] Loaded ACT from {checkpoint_path}")
    print(f"  device={device}, chunks={policy.config.chunk_size}, "
          f"obs_steps={policy.config.n_obs_steps}")
    return policy


# ---------------------------------------------------------------------------
# Observation capture
# ---------------------------------------------------------------------------
def capture_obs(data, j_qpos_ids, wrist_renderer, wrist_cam_id, overhead_renderer, overhead_cam_id):
    """Capture observation dict from current MuJoCo state."""
    # Joint angles (deg)
    qpos_deg = np.rad2deg(data.qpos[:6]).astype(np.float32)

    # Wrist camera
    wrist_renderer.update_scene(data, camera=wrist_cam_id)
    wrist_img = wrist_renderer.render()  # (480, 640, 3) uint8 RGB

    # Overhead camera
    overhead_renderer.update_scene(data, camera=overhead_cam_id)
    overhead_img = overhead_renderer.render()  # (240, 320, 3) uint8 RGB

    return {
        "observation.state": qpos_deg,
        "observation.images.wrist": wrist_img,
        "observation.images.overhead": overhead_img,
    }


def obs_to_tensors(obs_dict, device="cpu"):
    """Convert numpy observation dict to tensor batch (B=1)."""
    batch = {}

    # State: (6,) -> (1, 6) float32
    state = torch.from_numpy(obs_dict["observation.state"]).float().unsqueeze(0).to(device)

    # Images: HWC uint8 -> BCHW float32 [0,1]
    for key in ["observation.images.wrist", "observation.images.overhead"]:
        img = obs_dict[key]
        img_t = torch.from_numpy(img).float().permute(2, 0, 1)  # CHW
        img_t = img_t / 255.0
        img_t = img_t.unsqueeze(0).to(device)  # (1, 3, H, W)
        batch[key] = img_t

    batch["observation.state"] = state
    return batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained ACT policy in MuJoCo")
    parser.add_argument("--checkpoint", required=True, help="Path to policy checkpoint dir")
    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes")
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS_DEFAULT, help="Max steps per episode")
    parser.add_argument("--render", action="store_true", help="Show MuJoCo viewer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cube_random_xy", type=float, default=0.03, help="Cube XY random offset (m)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")

    # ---- Init scene ----
    model, data, ids = init_scene()
    cube_body_id = ids["cube"]
    wrist_cam_id = ids["wrist_cam"]
    overhead_cam_id = ids["overhead_cam"]
    j_qpos_ids = [model.jnt_qposadr[j] for j in range(6)]

    # Renderers
    wrist_renderer = mujoco.Renderer(model, 480, 640)
    overhead_renderer = mujoco.Renderer(model, 240, 320)

    # ---- Load policy ----
    policy = load_policy(args.checkpoint, device=device)

    # ---- Results ----
    successes = []
    step_counts = []

    # ---- Viewer (optional) ----
    viewer = None
    if args.render:
        viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)

    try:
        for ep in range(args.episodes):
            reset_episode(model, data, randomize_xy=args.cube_random_xy)
            policy.reset()

            done = False
            step = 0

            while not done and step < args.max_steps:
                # 1. Capture observation
                obs = capture_obs(data, j_qpos_ids, wrist_renderer, wrist_cam_id,
                                  overhead_renderer, overhead_cam_id)

                # 2. Convert to tensors
                batch = obs_to_tensors(obs, device=device)

                # 3. Policy inference
                with torch.inference_mode():
                    action = policy.select_action(batch)  # (1, 6) in deg

                # 4. Execute
                action_deg = action.squeeze(0).cpu().numpy()  # (6,) deg
                action_rad = np.deg2rad(action_deg)
                data.ctrl[:6] = action_rad.astype(np.float64)

                mujoco.mj_step(model, data)
                if viewer is not None:
                    viewer.sync()
                step += 1

                # 5. Check success
                if check_success(data, cube_body_id):
                    done = True
                    successes.append(True)
                    step_counts.append(step)
                    print(f"[EP {ep+1:2d}] SUCCESS in {step} steps")

            if not done:
                successes.append(False)
                step_counts.append(step)
                print(f"[EP {ep+1:2d}] FAIL (max steps reached)")

            time.sleep(0.1)

    finally:
        if viewer is not None:
            viewer.close()
        cv2.destroyAllWindows()

    # ---- Report ----
    print(f"\n{'='*50}")
    print(f"Results: {sum(successes)}/{len(successes)} ({sum(successes)/len(successes)*100:.1f}%)")
    if sum(successes) > 0:
        successful_steps = [s for i, s in enumerate(step_counts) if successes[i]]
        print(f"Avg steps (success): {np.mean(successful_steps):.0f}")
    print(f"Total episodes: {len(successes)}")


if __name__ == "__main__":
    main()
