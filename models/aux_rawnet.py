# models/aux_rawnet.py
"""
Lightweight Auxiliary RawNet-style encoder for AASIST integration.

Input: waveform tensor shape (B, N)  -> converted to (B,1,N)
Output: embedding tensor shape (B, emb_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.selu = nn.SELU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        if in_ch != out_ch:
            self.down = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        else:
            self.down = None
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.selu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.down is not None:
            identity = self.down(identity)
        out = out + identity
        out = self.pool(out)
        return out


class AuxiliaryRawNet(nn.Module):
    def __init__(self, embedding_dim: int = 128, in_channels: int = 1):
        """
        embedding_dim: final output embedding size (e.g., 128)
        """
        super().__init__()
        # small front-end conv stack (acts like a lightweight SincConv front-end)
        self.front = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.SELU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SELU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )

        # a few lightweight residual blocks
        self.res1 = _ResBlock1D(64, 128)
        self.res2 = _ResBlock1D(128, 128)

        # statistics pooling (mean + std)
        self.stat_pool = _StatsPooling()

        # projection to embedding
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 256),
            nn.SELU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        """
        x: (B, N) waveform -> convert to (B,1,N)
        returns: (B, embedding_dim)
        """
        # ensure float tensor
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B,1,N)
        out = self.front(x)     # (B, C, T)
        out = self.res1(out)
        out = self.res2(out)
        # stats pooling returns (B, C*2) where we concat mean and std
        pooled = self.stat_pool(out)
        emb = self.fc(pooled)
        return emb


class _StatsPooling(nn.Module):
    """Compute mean and std across time dimension -> concat"""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x: (B, C, T)
        mean = torch.mean(x, dim=2)
        std = torch.sqrt(torch.var(x, dim=2) + self.eps)
        return torch.cat([mean, std], dim=1)
