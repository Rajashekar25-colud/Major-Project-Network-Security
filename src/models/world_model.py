"""Temporal world model: LSTM + temporal self-attention with K-step forecasting heads.

The network learns network-state *dynamics*:
  * ``state``  head - the next H network-state vectors (state-transition learning),
  * ``stage``  head - the ATT&CK stage of each of the next H windows,
  * ``attack`` head - the attack probability of each of the next H windows.
``rollout`` additionally unrolls the learned dynamics autoregressively (predicted
state fed back as input) to imagine a longer future trajectory.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class WorldModel(nn.Module):
    def __init__(self, input_size: int, num_stages: int, horizon: int, hidden_size: int = 128,
                 layers: int = 2, dropout: float = 0.2, heads: int = 4):
        super().__init__()
        self.input_size, self.num_stages, self.horizon = input_size, num_stages, horizon
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU())
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.attn = nn.MultiheadAttention(hidden_size, heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.state_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(),
                                        nn.Linear(hidden_size, horizon * input_size))
        self.stage_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(),
                                        nn.Linear(hidden_size, horizon * num_stages))
        self.attack_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.GELU(),
                                         nn.Linear(hidden_size // 2, horizon))

    def forward(self, x: torch.Tensor, need_attention: bool = False) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        h, _ = self.lstm(z)
        a, w = self.attn(h, h, h, need_weights=need_attention, average_attn_weights=False)
        h = self.norm(h + self.drop(a))
        ctx = h[:, -1]
        B = x.shape[0]
        out = {
            "state": self.state_head(ctx).view(B, self.horizon, self.input_size),
            "stage_logits": self.stage_head(ctx).view(B, self.horizon, self.num_stages),
            "attack_logit": self.attack_head(ctx),
        }
        if need_attention:
            out["attention"] = w          # (B, heads, L, L)
        return out

    @torch.no_grad()
    def rollout(self, x: torch.Tensor, steps: int) -> dict[str, torch.Tensor]:
        """Autoregressive imagination: feed the predicted next state back as input."""
        self.eval()
        seq = x.clone()
        states, probs, stages = [], [], []
        for _ in range(steps):
            o = self.forward(seq)
            nxt = o["state"][:, 0]
            states.append(nxt); probs.append(torch.sigmoid(o["attack_logit"][:, 0]))
            stages.append(torch.softmax(o["stage_logits"][:, 0], -1))
            seq = torch.cat([seq[:, 1:], nxt.unsqueeze(1)], dim=1)
        return {"state": torch.stack(states, 1), "attack_prob": torch.stack(probs, 1), "stage_prob": torch.stack(stages, 1)}


def multitask_loss(out, y_state, y_stage, y_attack, active_mask, class_weight, pos_weight,
                   w_state=0.3, w_stage=0.7, w_attack=1.0):
    st = F.smooth_l1_loss(out["state"] * active_mask, y_state * active_mask)
    B, H, S = out["stage_logits"].shape
    valid = y_stage >= 0
    if valid.any():
        sl = F.cross_entropy(out["stage_logits"].reshape(B * H, S)[valid.reshape(-1)],
                             y_stage.reshape(-1)[valid.reshape(-1)], weight=class_weight)
    else:
        sl = out["stage_logits"].sum() * 0.0
    at = F.binary_cross_entropy_with_logits(out["attack_logit"], y_attack, pos_weight=pos_weight)
    total = w_state * st + w_stage * sl + w_attack * at
    return total, {"state": st.item(), "stage": sl.item(), "attack": at.item()}


class TransformerWorldModel(WorldModel):
    """Transformer encoder variant with the same forecasting interface."""

    def __init__(self, input_size: int, num_stages: int, horizon: int, hidden_size: int = 128,
             layers: int = 2, dropout: float = 0.2, heads: int = 4):
        super().__init__(input_size, num_stages, horizon, hidden_size, layers, dropout, heads)
        del self.lstm
        self.encoder_stack = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, heads, hidden_size * 4, dropout,
                                  batch_first=True, norm_first=True),
            num_layers=layers,
        )

    def forward(self, x: torch.Tensor, need_attention: bool = False) -> dict[str, torch.Tensor]:
        z = self.encoder_stack(self.encoder(x))
        ctx = self.norm(z[:, -1])
        B = x.shape[0]
        out = {
            "state": self.state_head(ctx).view(B, self.horizon, self.input_size),
            "stage_logits": self.stage_head(ctx).view(B, self.horizon, self.num_stages),
            "attack_logit": self.attack_head(ctx),
        }
        if need_attention:
            # The encoder does not expose weights without a custom layer; retain the
            # documented shape and provide a neutral attention map for explanations.
            out["attention"] = torch.zeros(B, 1, x.shape[1], x.shape[1], device=x.device)
        return out


def build_model(kind: str, input_size: int, num_stages: int, horizon: int,
                hp: dict) -> WorldModel:
    cls = TransformerWorldModel if str(kind).lower() == "transformer" else WorldModel
    return cls(input_size, num_stages, horizon, hidden_size=int(hp["hidden_size"]),
               layers=int(hp["layers"]), dropout=float(hp["dropout"]),
               heads=int(hp.get("attention_heads", hp.get("heads", 4))))
