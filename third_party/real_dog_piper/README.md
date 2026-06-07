# Real Go2W-Piper Baseline

This folder is a compact real-hardware deployment baseline for the `dog_lab` Go2W-Piper policy.

Main pieces:

- `go2w_api.py`: Unitree Go2W low-level state/control wrapper.
- `piper_api.py`: Piper CAN state/control wrapper using the local `piper_sdk`.
- `observation.py`: builds the 71-D proprio observation used by the trained policy.
- `policy.py`: loads `traced_actor.pt` plus optional `traced_hist_encoder.pt`.
- `deploy_real.py`: ties hardware feedback, policy inference, EE goal sampling, and commands together.

Test without hardware:

```bash
cd dog_lab/third_party
python -m real_dog_piper.deploy_real --dry-run --duration 5
```

Real run example:

```bash
cd dog_lab/third_party
python -m real_dog_piper.deploy_real --iface eth0 --piper-can can0 --cmd 0.15 0.0 0.0
```

The public joint order is the dog_lab policy order. `go2w_api.py` maps it to the Unitree low-level motor order internally.
