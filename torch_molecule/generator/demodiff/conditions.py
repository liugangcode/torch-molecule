import torch
import torch.nn as nn
import math


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t = t.view(-1)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)

        return t_emb


class ScoreEmbedder(nn.Module):
    """Embed similarity scores: target 1, demonstrations between 0 and 1."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def score_embedding(score, dim, max_period=10000, factor=500):
        """Create sinusoidal embeddings from per-node scores shaped (batch, nodes, 1)."""
        bs, n, _ = score.size()
        score = score.view(-1)
        score = factor * score
        freqs = 1.0 / (max_period ** (torch.arange(0, dim, 2, device=score.device).float() / dim))
        freqs = torch.outer(score, freqs)
        emb = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1).view(bs, n, -1)
        return emb

    def forward(self, score):
        score_freq = self.score_embedding(score, self.frequency_embedding_size, factor=500)
        return self.mlp(score_freq)


class InputEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        h = self.in_layer(x)
        h = self.softmax(h)
        h = self.out_layer(h)
        return h
