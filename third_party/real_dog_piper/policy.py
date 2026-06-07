from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import PolicyConfig


class PolicyRunner:
    """TorchScript actor wrapper with optional Loco history encoder."""

    def __init__(self, cfg: PolicyConfig, actor_path: str | Path | None = None, hist_encoder_path: str | Path | None = None):
        self.cfg = cfg
        self.actor_path = Path(actor_path or cfg.actor_path).expanduser().resolve()
        hist_path = hist_encoder_path if hist_encoder_path is not None else cfg.hist_encoder_path
        self.hist_encoder_path = Path(hist_path).expanduser().resolve() if hist_path else None
        if not self.actor_path.exists():
            raise FileNotFoundError(f"Actor policy not found: {self.actor_path}")
        if self.hist_encoder_path and not self.hist_encoder_path.exists():
            raise FileNotFoundError(f"History encoder not found: {self.hist_encoder_path}")

        self.actor = torch.jit.load(str(self.actor_path), map_location="cpu")
        self.actor.eval()
        self.hist_encoder = None
        if self.hist_encoder_path is not None:
            self.hist_encoder = torch.jit.load(str(self.hist_encoder_path), map_location="cpu")
            self.hist_encoder.eval()

    def act(self, obs: np.ndarray, history: np.ndarray) -> np.ndarray:
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
        with torch.inference_mode():
            if self.hist_encoder is not None:
                hist_tensor = torch.from_numpy(history.reshape(-1)).unsqueeze(0).float()
                latent = self.hist_encoder(hist_tensor)
                actor_input = torch.cat([obs_tensor, latent], dim=1)
            else:
                actor_input = obs_tensor
            actions = self.actor(actor_input).detach().cpu().numpy().squeeze()
        return np.clip(actions, -self.cfg.clip_actions, self.cfg.clip_actions).astype(np.float32)
