import sys
import yaml
import torch
from pathlib import Path
from typing import Dict
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.instance_loader import load_cached_instance
from src.ea.evolutionary_algorithm import EvolutionaryAlgorithm
from src.roots import get_project_root
from src.frameworks.rl_param_optimizer.env import AdaptiveEAEnv
from src.frameworks.rl_param_optimizer.agent import RLAgent, load_training_instances, save_training_config


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def make_env(rank: int, config: Dict, instance_pool: list, ea_class):
    def _init():
        env = AdaptiveEAEnv(
            ea_class=ea_class,
            instance_pool=instance_pool,
            base_mu=config['mu'],
            base_generations=config['generations'],
            n_check_gen=config['n_check_gen'],
            plus_selection=config['plus_selection'],
            early_stopping_patience=config['early_stopping_patience'],
            seed=config['experiment']['seed_start'] + rank
        )
        return Monitor(env)

    return _init


def main():
    root = get_project_root()
    config_path = root / "src" / "frameworks" / "rl_param_optimizer" / "config_rl.yaml"
    config = load_config(config_path)

    cache_dir = root / "data" / "processed" / "cached_instances"
    max_instances = config['rl']['max_training_instances']
    instance_pool = load_training_instances(cache_dir, max_instances=max_instances)

    n_envs = config['rl']['n_parallel_envs']

    print(f"Initializing {n_envs} parallel environments...")
    train_env = SubprocVecEnv([
        make_env(i, config, instance_pool, EvolutionaryAlgorithm)
        for i in range(n_envs)
    ])

    models_dir = root / "src" / "frameworks" / "rl_param_optimizer" / "trained_models"
    logs_dir = root / "src" / "frameworks" / "rl_param_optimizer" / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    agent = RLAgent(
        env=train_env,
        learning_rate=config['rl']['learning_rate'],
        n_steps=config['rl']['n_steps'],
        batch_size=config['rl']['batch_size'],
        n_epochs=config['rl']['n_epochs'],
        gamma=config['rl']['gamma'],
        gae_lambda=config['rl']['gae_lambda'],
        clip_range=config['rl']['clip_range'],
        ent_coef=config['rl']['ent_coef'],
        vf_coef=config['rl']['vf_coef'],
        max_grad_norm=config['rl']['max_grad_norm'],
        tensorboard_log=str(logs_dir),
        verbose=2
    )

    save_training_config(config, models_dir)

    agent.train(
        total_timesteps=config['rl']['total_timesteps'],
        save_path=str(models_dir),
        save_freq=config['rl']['save_freq']
    )

    print("Training finished. Running final evaluation...")
    eval_env = Monitor(AdaptiveEAEnv(
        ea_class=EvolutionaryAlgorithm,
        instance_pool=instance_pool,
        base_mu=config['mu'],
        base_generations=config['generations'],
        n_check_gen=config['n_check_gen'],
        plus_selection=config['plus_selection'],
        early_stopping_patience=config['early_stopping_patience'],
        seed=999
    ))

    agent.env = eval_env
    eval_results = agent.evaluate(n_eval_episodes=5)
    print(f"Mean Reward: {eval_results['mean_reward']:.2f}")


if __name__ == "__main__":
    main()