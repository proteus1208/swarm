# RL/train_RL.py
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

_RL_DIR = Path(__file__).resolve().parent
if str(_RL_DIR) not in sys.path:
    sys.path.insert(0, str(_RL_DIR))

from train_env_worker import make_training_env


def linear_schedule(initial_value: float, final_value: float = 0.0):
    def schedule(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)

    return schedule


def main():
    import torch

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    parser = argparse.ArgumentParser(description="Train PPO model for Swarm subnet")

    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model path. Default: swarm/submission_template/ppo_policy.zip",
    )

    args = parser.parse_args()

    # Fixed settings
    timesteps = 1_000_000
    seed = int(time.time() % 1000)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.set_num_threads(4)
    set_random_seed(seed)

    learning_rate = linear_schedule(3e-4, 5e-5)
    clip_range = linear_schedule(0.20, 0.08)

    n_steps = 2048
    batch_size = 512
    n_epochs = 10

    gamma = 0.99
    gae_lambda = 0.95
    ent_coef = 0.005
    vf_coef = 0.5
    max_grad_norm = 0.5
    target_kl = 0.02

    output_dir = Path(__file__).parent.parent / "swarm" / "submission_template"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model) if args.model else output_dir / "ppo_policy.zip"

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.gui and args.n_envs > 1:
        print("Note: --gui only runs on env index 0.")
        print("On Windows, GUI uses DummyVecEnv to avoid Qt/SubprocVecEnv crashes.")

    env_fns = [
        make_training_env(
            seed + i,
            gui=args.gui and i == 0,
            path_monitor=args.gui and i == 0,
        )
        for i in range(args.n_envs)
    ]

    # Important:
    # - DummyVecEnv is safer for GUI and Windows debugging.
    # - SubprocVecEnv is faster for many non-GUI envs.
    if args.gui or args.n_envs == 1:
        env = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns, start_method="spawn")

    env = VecMonitor(env)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000 // args.n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_policy",
        save_replay_buffer=False,
    )

    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],
            vf=[256, 256],
        ),
        activation_fn=torch.nn.Tanh,
    )

    if args.init:
        print("Creating new PPO model")

        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            device=device,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./ppo_logs/",
            stats_window_size=100,
        )

        reset_num_timesteps = True

    else:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading model from: {model_path}")

        model = PPO.load(
            str(model_path),
            env=env,
            device=device,
            tensorboard_log="./ppo_logs/",
            custom_objects={
                "learning_rate": learning_rate,
                "clip_range": clip_range,
            },
        )

        model.learning_rate = learning_rate
        model.clip_range = clip_range
        model.ent_coef = ent_coef
        model.target_kl = target_kl

        reset_num_timesteps = False

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name="PPO",
        )

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")

    except Exception as e:
        print(f"\nError during training: {e}")
        traceback.print_exc()
        print("\nSaving model before exit...")

    finally:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))

        try:
            env.close()
        except (EOFError, BrokenPipeError, OSError, ConnectionError):
            pass

    print(f"\n✅ Final model saved to: {model_path}")
    print(f"✅ Checkpoints saved in: {checkpoint_dir}")


if __name__ == "__main__":
    main()