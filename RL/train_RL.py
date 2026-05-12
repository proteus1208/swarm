# #!/usr/bin/env python3

# import argparse
# import sys
# from pathlib import Path

# _RL_DIR = Path(__file__).resolve().parent
# if str(_RL_DIR) not in sys.path:
#     sys.path.insert(0, str(_RL_DIR))

# from train_env_worker import make_training_env


# def main():
#     # SB3/torch only in the trainer process — not in SubprocVecEnv worker imports.
#     from stable_baselines3 import PPO
#     from stable_baselines3.common.vec_env import SubprocVecEnv

#     parser = argparse.ArgumentParser(description="Train PPO model for Swarm subnet")
#     parser.add_argument("--timesteps", type=int, default=1000000)
#     parser.add_argument("--continuous", action="store_true")
#     parser.add_argument("--n-envs", type=int, default=4)
#     args = parser.parse_args()

#     n_envs = args.n_envs

#     env = SubprocVecEnv([make_training_env(1000 + i) for i in range(n_envs)])

#     output_dir = Path(__file__).parent.parent / "swarm" / "submission_template"
#     output_dir.mkdir(parents=True, exist_ok=True)
#     model_path = output_dir / "ppo_policy.zip"

#     if args.continuous:
#         model = PPO.load(
#             str(model_path),
#             env=env,
#             device="cuda",
#             tensorboard_log="./ppo_logs/",
#         )
#     else:
#         model = PPO(
#             "MultiInputPolicy",
#             env,
#             verbose=1,
#             device="cuda",
#             learning_rate=3e-4,
#             n_steps=2048,
#             batch_size=512,
#             max_grad_norm=0.5,
#             tensorboard_log="./ppo_logs/",
#         )

#     try:
#         model.learn(total_timesteps=args.timesteps)
#     except KeyboardInterrupt:
#         print("Training interrupted, saving model...")
#     except Exception as e:
#         print(f"Error: {e}")

#     model.save(str(model_path))
#     try:
#         env.close()
#     except (EOFError, BrokenPipeError, OSError, ConnectionError):
#         # After KeyboardInterrupt, SubprocVecEnv workers on Windows may already be gone.
#         pass

#     print(f"\n✅ Model saved to: {model_path}")
#     print("\n📋 Next steps:")
#     print("   1. Test: python tests/test_rpc.py swarm/submission_template/ --zip")
#     print("   2. Run miner")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3

#!/usr/bin/env python3

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
    """
    SB3 passes progress_remaining:
        1.0 at start
        0.0 at end
    """
    def schedule(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)

    return schedule


def main():
    # SB3/torch only in the trainer process — not in SubprocVecEnv worker imports.
    import torch

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

    class SaveVecNormalizeCallback(BaseCallback):
        """
        Saves VecNormalize stats when EvalCallback finds a new best model.
        Useful only when VecNormalize is enabled.
        """

        def __init__(self, save_path: Path, verbose: int = 0):
            super().__init__(verbose)
            self.save_path = save_path

        def _on_step(self) -> bool:
            vec_normalize_env = self.model.get_vec_normalize_env()

            if vec_normalize_env is not None:
                self.save_path.parent.mkdir(parents=True, exist_ok=True)
                vec_normalize_env.save(str(self.save_path))

                if self.verbose:
                    print(f"Saved VecNormalize stats to: {self.save_path}")

            return True

    class TensorboardPathProgressCallback(BaseCallback):
        """
        Logs mission path progress each training iteration (same cadence as PPO
        rollouts / inner epochs). ``path_progress_01`` is 0 at start and 1 at
        goal along the task chord; per-step ``path_progress_delta`` is negative
        when the projection moves backward.
        """

        def __init__(self, verbose: int = 0):
            super().__init__(verbose)
            self._pp: list[float] = []
            self._pp_delta: list[float] = []

        def _on_step(self) -> bool:
            infos = self.locals.get("infos")
            if not infos:
                return True
            for info in infos:
                if not isinstance(info, dict):
                    continue
                if "path_progress_01" in info:
                    self._pp.append(float(info["path_progress_01"]))
                if "path_progress_delta" in info:
                    self._pp_delta.append(float(info["path_progress_delta"]))
            return True

        def _on_rollout_end(self) -> bool:
            import numpy as np

            if self._pp:
                xs = np.asarray(self._pp, dtype=np.float64)
                self.logger.record("custom/path_progress_01_mean", float(xs.mean()))
                self.logger.record("custom/path_progress_01_min", float(xs.min()))
                self.logger.record("custom/path_progress_01_max", float(xs.max()))
            if self._pp_delta:
                d = np.asarray(self._pp_delta, dtype=np.float64)
                self.logger.record("custom/path_progress_delta_mean", float(d.mean()))
                self.logger.record("custom/path_progress_delta_min", float(d.min()))
                self.logger.record("custom/path_progress_delta_max", float(d.max()))
            self._pp.clear()
            self._pp_delta.clear()
            return True

    parser = argparse.ArgumentParser(description="Train PPO model for Swarm subnet")

    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--eval-n-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=int(time.time()%1000))

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)

    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--learning-rate-final", type=float, default=5e-5)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.20)
    parser.add_argument("--clip-range-final", type=float, default=0.08)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.02)

    # Saving / eval
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--log-dir", type=str, default="./ppo_logs/")
    parser.add_argument("--run-name", type=str, default="PPO")

    # Optional normalization
    parser.add_argument("--normalize-obs", action="store_true")
    parser.add_argument("--normalize-reward", action="store_true")

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Enable PyBullet viewer and Qt path monitor on the first env only (avoid with n-envs>1).",
    )

    args = parser.parse_args()

    if args.gui and args.n_envs > 1:
        print(
            "Note: --gui applies PyBullet viewer and path monitor only to subprocess env index 0."
        )

    set_random_seed(args.seed)

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        device = "cpu"

    n_envs = args.n_envs

    output_dir = Path(__file__).parent.parent / "swarm" / "submission_template"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "ppo_policy.zip"
    vecnormalize_path = output_dir / "vecnormalize.pkl"

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_model_dir = output_dir / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)

    eval_log_dir = output_dir / "eval_logs"
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    use_existing_vecnormalize = not args.init and vecnormalize_path.exists()
    use_vecnormalize = args.normalize_obs or args.normalize_reward or use_existing_vecnormalize

    def build_vec_env(
        num_envs: int,
        seed_start: int,
        training: bool,
        load_vecnormalize: bool,
        *,
        gui: bool = False,
        path_monitor: bool = False,
    ):
        env = SubprocVecEnv(
            [
                make_training_env(
                    seed_start + i,
                    gui=gui and i == 0,
                    path_monitor=path_monitor and i == 0,
                )
                for i in range(num_envs)
            ]
        )

        env = VecMonitor(env)

        if use_vecnormalize:
            if load_vecnormalize and vecnormalize_path.exists():
                env = VecNormalize.load(str(vecnormalize_path), env)
                print(f"Loaded VecNormalize stats from: {vecnormalize_path}")
            else:
                env = VecNormalize(
                    env,
                    norm_obs=args.normalize_obs,
                    norm_reward=args.normalize_reward,
                    gamma=args.gamma,
                    clip_obs=10.0,
                    clip_reward=10.0,
                )

            env.training = training

            # During evaluation, do not normalize reward.
            # We want real evaluation reward.
            if not training:
                env.norm_reward = False

        return env

    env = build_vec_env(
        num_envs=n_envs,
        seed_start=args.seed,
        training=True,
        load_vecnormalize=use_existing_vecnormalize,
        gui=False,
        path_monitor=args.gui,
    )

    eval_env = None
    if args.eval_freq > 0:
        eval_env = build_vec_env(
            num_envs=args.eval_n_envs,
            seed_start=args.seed + 100_000,
            training=False,
            load_vecnormalize=use_existing_vecnormalize,
            gui=False,
            path_monitor=False,
        )

    rollout_size = args.n_steps * n_envs
    if rollout_size % args.batch_size != 0:
        print(
            f"Warning: n_steps * n_envs = {rollout_size}, "
            f"but batch_size = {args.batch_size}. "
            "It is usually better when rollout_size is divisible by batch_size."
        )

    lr_schedule = linear_schedule(
        initial_value=args.learning_rate,
        final_value=args.learning_rate_final,
    )

    clip_schedule = linear_schedule(
        initial_value=args.clip_range,
        final_value=args.clip_range_final,
    )

    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],
            vf=[256, 256],
        ),
        activation_fn=torch.nn.Tanh,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_policy",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    path_progress_tb_callback = TensorboardPathProgressCallback(verbose=0)

    callbacks = [checkpoint_callback, path_progress_tb_callback]

    if eval_env is not None:
        save_vecnormalize_on_best = SaveVecNormalizeCallback(
            save_path=best_model_dir / "vecnormalize.pkl",
            verbose=1,
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(best_model_dir),
            log_path=str(eval_log_dir),
            eval_freq=max(args.eval_freq // n_envs, 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            render=False,
            callback_on_new_best=save_vecnormalize_on_best,
            verbose=1,
        )

        callbacks.append(eval_callback)

    callback = CallbackList(callbacks)

    if not args.init:
        if not model_path.exists():
            raise FileNotFoundError(f"Cannot continue. Model not found: {model_path}")

        print(f"Loading model from: {model_path}")

        model = PPO.load(
            str(model_path),
            env=env,
            device=device,
            tensorboard_log=args.log_dir,
            custom_objects={
                "learning_rate": lr_schedule,
                "clip_range": clip_schedule,
            },
        )

        # Safe updates for continued training
        model.learning_rate = lr_schedule
        model.clip_range = clip_schedule
        model.ent_coef = args.ent_coef
        model.target_kl = args.target_kl

        reset_num_timesteps = False

    else:
        print("Creating new PPO model")

        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            device=device,
            learning_rate=lr_schedule,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=clip_schedule,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            policy_kwargs=policy_kwargs,
            tensorboard_log=args.log_dir,
            stats_window_size=100,
        )

        reset_num_timesteps = True

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=args.run_name,
        )

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")

    except Exception as e:
        print(f"\nError during training: {e}")
        traceback.print_exc()
        print("\nSaving current model before exit...")

    finally:
        model.save(str(model_path))

        vec_normalize_env = model.get_vec_normalize_env()
        if vec_normalize_env is not None:
            vec_normalize_env.save(str(vecnormalize_path))
            print(f"VecNormalize stats saved to: {vecnormalize_path}")

        try:
            env.close()
        except (EOFError, BrokenPipeError, OSError, ConnectionError):
            pass

        if eval_env is not None:
            try:
                eval_env.close()
            except (EOFError, BrokenPipeError, OSError, ConnectionError):
                pass

    print(f"\n✅ Final model saved to: {model_path}")
    print(f"✅ Checkpoints saved in: {checkpoint_dir}")
    print(f"✅ Best model saved in: {best_model_dir}")

    print("\n📋 Next steps:")
    print("   1. Test: python tests/test_rpc.py swarm/submission_template/ --zip")
    print("   2. Run miner")


if __name__ == "__main__":
    main()