import os
import yaml
import pickle
from pathlib import Path
from typing import List, Dict
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import torch


class TensorboardCallback(BaseCallback):

    def __init__(self, verbose=0, log_freq=100, total_timesteps=10000):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.log_freq = log_freq
        self.n_calls = 0
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        self.n_calls += 1

        if 'episode' in self.locals.get('infos', [{}])[0]:
            info = self.locals['infos'][0]['episode']
            ep_reward = info['r']
            ep_length = info['l']

            self.episode_rewards.append(ep_reward)
            self.episode_lengths.append(ep_length)

            recent_mean = np.mean(self.episode_rewards[-5:]) if len(self.episode_rewards) >= 5 else ep_reward
            all_mean = np.mean(self.episode_rewards)

            print(f"\n{'=' * 70}")
            print(f"EPISODE {len(self.episode_rewards)} COMPLETE")
            print(f"{'=' * 70}")
            print(f"  Reward: {ep_reward:.2f}")
            print(f"  Length: {ep_length}")
            print(f"  Avg reward (last 5): {recent_mean:.2f}")
            print(f"  Avg reward (all): {all_mean:.2f}")
            print(f"  Total timesteps: {self.n_calls}")
            print(f"{'=' * 70}\n")

            self.logger.record('rollout/ep_rew_mean', all_mean)
            self.logger.record('rollout/ep_len_mean', np.mean(self.episode_lengths))

        if self.n_calls % self.log_freq == 0:
            progress_pct = (self.n_calls / self.total_timesteps) * 100
            print(f"[Timestep {self.n_calls}/{self.total_timesteps}] Progress: {progress_pct:.1f}%", flush=True)

        return True


class RLAgent:

    def __init__(self,
                 env,
                 learning_rate: float = 3e-4,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 ent_coef: float = 0.01,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 tensorboard_log: str = None,
                 device: str = 'auto',
                 verbose: int = 1):

        self.env = env

        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.ReLU
        )

        self.model = PPO(
            policy='MlpPolicy',
            env=env,
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
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            tensorboard_log=tensorboard_log
        )

    def train(self,
              total_timesteps: int,
              save_path: str,
              save_freq: int = 1000,
              callback=None):

        os.makedirs(save_path, exist_ok=True)

        tensorboard_callback = TensorboardCallback(verbose=1, log_freq=500, total_timesteps=total_timesteps)

        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=save_path,
            name_prefix="rl_model",
            save_replay_buffer=False,
            save_vecnormalize=False
        )

        from stable_baselines3.common.callbacks import CallbackList
        if callback is None:
            callback = CallbackList([tensorboard_callback, checkpoint_callback])
        else:
            callback = CallbackList([callback, tensorboard_callback, checkpoint_callback])

        print("\n[TRAINING STARTED] This will take a while...")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Checkpoints every {save_freq} steps in: {save_path}")
        print("=" * 70)
        print("\nNOTE: Progress updates every 500 steps and every 5 episodes")
        print("=" * 70 + "\n")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False
        )

        print("\n" + "=" * 70)
        print("[TRAINING FINISHED] Saving model...")

        final_model_path = os.path.join(save_path, "ppo_adaptive_ea_final")
        self.model.save(final_model_path)

        print(f"Model saved to: {final_model_path}")

        return self.model

    def save(self, path: str):

        self.model.save(path)

    def load(self, path: str):

        self.model = PPO.load(path, env=self.env)

    def predict(self, obs, deterministic: bool = True):

        action, _states = self.model.predict(obs, deterministic=deterministic)
        return action

    def evaluate(self, n_eval_episodes: int = 10):

        episode_rewards = []
        episode_lengths = []

        for _ in range(n_eval_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                action = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths)
        }


def create_training_env(ea_class, instance_pool, config, n_envs: int = 1):
    def make_env(rank: int):
        def _init():
            env = Monitor(ea_class(
                ea_class=ea_class,
                instance_pool=instance_pool,
                base_mu=config['mu'],
                base_generations=config['generations'],
                n_check_gen=config.get('n_check_gen', 20),
                plus_selection=config['plus_selection'],
                early_stopping_patience=config['early_stopping_patience'],
                seed=config['experiment']['seed_start'] + rank
            ))
            return env

        return _init

    if n_envs > 1:
        return SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        return DummyVecEnv([make_env(0)])


def load_training_instances(cache_dir: Path, max_instances: int = 10) -> List[Dict]:
    instance_files = sorted(cache_dir.glob("*.pkl"))[:max_instances]
    instances = []

    for pkl_file in instance_files:
        with open(pkl_file, 'rb') as f:
            instance_data = pickle.load(f)
            instances.append(instance_data)

    return instances


def save_training_config(config: Dict, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_dir / "rl_training_config.yaml"

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)