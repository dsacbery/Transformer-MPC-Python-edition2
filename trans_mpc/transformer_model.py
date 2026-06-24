from __future__ import annotations

import math

import torch
from torch import nn


class RiskTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 48,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.05,
        k_v_min: float = 0.45,
    ):
        super().__init__()
        self.k_v_min = k_v_min
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 3,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        z = z + _sinusoidal_position_encoding(z.size(1), z.size(2), z.device, z.dtype).unsqueeze(0)
        encoded = self.encoder(z)
        raw = self.head(encoded[:, -1, :])
        risk = torch.sigmoid(raw[:, :3])
        k_v = self.k_v_min + (1.0 - self.k_v_min) * torch.sigmoid(raw[:, 3:4])
        return torch.cat([risk, k_v], dim=1)


def _sinusoidal_position_encoding(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div_terms = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(positions * div_terms)
    if dim > 1:
        pe[:, 1::2] = torch.cos(positions * div_terms[: pe[:, 1::2].shape[1]])
    return pe
