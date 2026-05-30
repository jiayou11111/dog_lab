import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn


DEPLOY_MUJOCO_ROOT = Path(__file__).resolve().parent
THIRD_PARTY_ROOT = DEPLOY_MUJOCO_ROOT.parent
LOCO_RSL_RL_ROOT = THIRD_PARTY_ROOT / "loco_rsl_rl"
if str(LOCO_RSL_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCO_RSL_RL_ROOT))

from rsl_rl.modules import ActorCritic  # noqa: E402


class DeployActor(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor_backbone = actor.actor_backbone
        self.actor_leg_control_head = actor.actor_leg_control_head
        self.actor_arm_control_head = actor.actor_arm_control_head

    def forward(self, obs_and_latent):
        backbone_output = self.actor_backbone(obs_and_latent)
        leg_output = self.actor_leg_control_head(backbone_output)
        arm_output = self.actor_arm_control_head(backbone_output)
        return torch.cat([leg_output, arm_output], dim=-1)


class DeployHistoryEncoder(nn.Module):
    def __init__(self, history_encoder, history_len, num_proprio):
        super().__init__()
        self.history_encoder = history_encoder
        self.history_len = history_len
        self.num_proprio = num_proprio

    def forward(self, obs_history):
        return self.history_encoder(obs_history.view(-1, self.history_len, self.num_proprio))


def build_actor_critic():
    return ActorCritic(
        71,
        71,
        22,
        actor_hidden_dims=[128],
        critic_hidden_dims=[128],
        activation="elu",
        init_std=[[1.0, 1.0, 1.0, 1.0] * 4 + [1.0] * 6],
        leg_control_head_hidden_dims=[128, 128],
        arm_control_head_hidden_dims=[128, 128],
        priv_encoder_dims=[64, 20],
        cost_hidden_dims=[128, 128, 128],
        num_leg_actions=16,
        num_arm_actions=6,
        num_priv=22,
        num_hist=10,
        num_prop=71,
        num_costs=2,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Export a DogLab Go2W-Piper checkpoint for MuJoCo deploy.")
    parser.add_argument("checkpoint", type=str, help="Path to an RSL-RL checkpoint, for example model_500.pt.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEPLOY_MUJOCO_ROOT / "pre_train" / "go2w_piper_cost"),
        help="Directory for traced_actor.pt and traced_hist_encoder.pt.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    actor_critic = build_actor_critic()
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    actor_critic.load_state_dict(state_dict)
    actor_critic.eval()

    deploy_actor = DeployActor(actor_critic.actor).eval()
    deploy_hist_encoder = DeployHistoryEncoder(actor_critic.actor.history_encoder, 10, 71).eval()

    traced_actor = torch.jit.trace(deploy_actor, torch.zeros(1, 91))
    traced_hist_encoder = torch.jit.trace(deploy_hist_encoder, torch.zeros(1, 710))

    actor_path = output_dir / "traced_actor.pt"
    hist_encoder_path = output_dir / "traced_hist_encoder.pt"
    traced_actor.save(str(actor_path))
    traced_hist_encoder.save(str(hist_encoder_path))

    print(f"Exported actor: {actor_path}")
    print(f"Exported history encoder: {hist_encoder_path}")


if __name__ == "__main__":
    main()
