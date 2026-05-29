# Go2W-Piper Stage 1 Locomotion

Stage 1 trains the chassis while the Piper arm stays fixed at its default pose.
The policy action is 16-D: 12 leg position targets and 4 wheel velocity targets.

## Run

```bash
./run_dog_lab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task DogLab-Go2W-Piper-Flat-v0
```

For a small smoke test or visualization:

```bash
./run_dog_lab.sh -p scripts/environments/random_agent.py \
  --task DogLab-Go2W-Piper-Flat-Play-v0 --num_envs 1
```

## Code Path

1. `dog_lab/__init__.py` imports `dog_lab.tasks`.
2. `dog_lab/tasks/__init__.py` imports `dog_lab.tasks.go2w_piper`.
3. `dog_lab/tasks/go2w_piper/__init__.py` registers Gym task ids.
4. `flat_env_cfg.py` or `rough_env_cfg.py` builds the Isaac Lab environment config.
5. `agents/rsl_rl_ppo_cfg.py` provides the RSL-RL PPO runner config.
6. `scripts/reinforcement_learning/rsl_rl/train.py` launches Isaac Sim, creates the Gym env, wraps it for RSL-RL, and starts PPO.

## Important Places

- `assets/robots/go2w_piper.py`: USD path, default joint pose, PD gains, effort and velocity limits.
- `tasks/go2w_piper/rough_env_cfg.py`: action split, command ranges, randomization, rewards, terminations.
- `tasks/go2w_piper/actions.py`: the fixed-arm action term. It consumes 0 policy dimensions and sends default joint targets to `joint1` through `joint6`.
- `tasks/go2w_piper/mdp.py`: chassis-only custom reward helpers, especially action-rate, joint-power, and mirror penalties.
- `tasks/go2w_piper/agents/rsl_rl_ppo_cfg.py`: PPO horizon, iterations, network size, learning rate, entropy, and batch settings.

## Stage 1 Logic

Commands sample desired base velocity: forward velocity, yaw velocity, and heading.
Lateral velocity is fixed to zero, matching the first-stage Loco-Manipulation setup.

The policy receives observations from Isaac Lab's velocity locomotion task, with joint
position and velocity terms restricted to the chassis joints. The arm is not part of the
action space and is held by a position target at the default joint pose.

Rewards mainly ask the base to track commanded linear and yaw velocity while discouraging
vertical motion, roll/pitch angular velocity, bad orientation, high torque, high
acceleration, fast action changes, low/high base height, joint-limit violations, and
left-right leg asymmetry.

Flat training is the simplest first target because terrain scanning and terrain curriculum
are disabled. Rough training keeps the same chassis control split, but enables the generated
terrain path inherited from Isaac Lab.
