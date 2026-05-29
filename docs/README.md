# Dog Lab

Standalone Isaac Lab extension for Go2W-Piper assets, environments, and training configuration.

## Layout

- `dog_lab/assets`: robot asset configurations.
- `dog_lab/tasks/go2w_piper`: Gym registration plus the stage-1 locomotion configs.
- `data`: MJCF, URDF, and mesh assets.
- `scripts`: runnable environment and RSL-RL scripts.

## Examples

From a standalone checkout next to `isaaclab_dog-main`:

```bash
./run_dog_lab.sh -p scripts/environments/random_agent.py \
  --task DogLab-Go2W-Piper-Flat-Play-v0 --num_envs 1

./run_dog_lab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task DogLab-Go2W-Piper-Flat-v0
```

If Isaac Lab is elsewhere:

```bash
ISAACLAB_PATH=/path/to/IsaacLab ./run_dog_lab.sh -p scripts/environments/random_agent.py \
  --task DogLab-Go2W-Piper-Flat-Play-v0 --num_envs 1
```

`run_dog_lab.sh` starts Isaac Lab in a clean environment and adds this project to `PYTHONPATH`.
This keeps the external `dog_lab` project visible while avoiding common Isaac Sim GUI/XCB conflicts
from the current shell environment. By default caches are written to `../.cache_isaac`; override with:

```bash
DOG_LAB_CACHE_ROOT=/path/to/cache ./run_dog_lab.sh -p scripts/environments/random_agent.py \
  --task DogLab-Go2W-Piper-Flat-Play-v0 --num_envs 1
```

The older `DogLab-Velocity-...` task ids are still registered for compatibility.

See `docs/stage1_locomotion.md` for the first-stage training flow and the files worth reading first.
