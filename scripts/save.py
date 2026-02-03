import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.roots import get_project_root

print("=" * 70)
print("EMERGENCY MODEL SAVE")
print("=" * 70)

root = get_project_root()
logs_dir = root / "src" / "frameworks" / "rl_param_optimizer" / "logs"
models_dir = root / "src" / "frameworks" / "rl_param_optimizer" / "trained_models"

models_dir.mkdir(parents=True, exist_ok=True)

ppo_dirs = list(logs_dir.glob("PPO_*"))

if not ppo_dirs:
    print("\nERROR: No PPO logs found!")
    print(f"Looked in: {logs_dir}")
    sys.exit(1)

latest_ppo_dir = max(ppo_dirs, key=lambda p: p.stat().st_mtime)
print(f"\nFound latest PPO run: {latest_ppo_dir.name}")

rl_model_files = list(latest_ppo_dir.glob("rl_model_*.zip"))

if not rl_model_files:
    print("\nERROR: No model checkpoints found in logs!")
    print("The model was only in memory and wasn't saved :(")
    print("\nYou'll need to re-run training with a shorter timestep limit.")
    sys.exit(1)

latest_checkpoint = max(rl_model_files, key=lambda p: int(p.stem.split('_')[-1]))
print(f"Found latest checkpoint: {latest_checkpoint.name}")

checkpoint_step = int(latest_checkpoint.stem.split('_')[-1])
print(f"Checkpoint at step: {checkpoint_step}")

import shutil
dest_path = models_dir / "ppo_adaptive_ea_final.zip"
shutil.copy(str(latest_checkpoint), str(dest_path))

print(f"\n✅ MODEL SAVED TO: {dest_path}")
print(f"✅ Trained for {checkpoint_step} steps")
print("\nYou can now run run_exp02.py to test it!")
print("=" * 70)