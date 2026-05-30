# DogLab Go2W-Piper MuJoCo Deploy

This folder is a DogLab-local port of the Loco-Manipulation MuJoCo deploy code.
It no longer depends on `legged_gym` paths: XML assets are resolved from this
folder, and policy paths can be passed on the command line.

## Install

Use the same `isaaclab` conda environment if you want one environment for
training, export, and MuJoCo deploy:

```powershell
conda activate isaaclab
pip install mujoco==3.3.3 glfw PyOpenGL
```

If the viewer cannot open on Windows, install the Visual C++ runtime and update
the graphics driver.

## Export From A Training Checkpoint

```powershell
cd F:\DOG\dog_lab
python third_party\deploy_mujoco\export_policy.py logs\rsl_rl\go2w_piper_cost\<run>\model_<iter>.pt
```

By default this writes:

- `third_party/deploy_mujoco/pre_train/go2w_piper_cost/traced_actor.pt`
- `third_party/deploy_mujoco/pre_train/go2w_piper_cost/traced_hist_encoder.pt`

## Run Deploy

```powershell
cd F:\DOG\dog_lab
python third_party\deploy_mujoco\deploy_mujoco.py
```

You can also pass explicit weights:

```powershell
python third_party\deploy_mujoco\deploy_mujoco.py `
  --actor-path path\to\traced_actor.pt `
  --hist-encoder-path path\to\traced_hist_encoder.pt
```

For a quick headless smoke test:

```powershell
python third_party\deploy_mujoco\deploy_mujoco.py --no-viewer --duration 2
```
