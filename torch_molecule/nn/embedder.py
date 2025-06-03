import torch
import torch.nn as nn
import math

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations using a sinusoidal embedding
    followed by a multilayer perceptron (MLP).
    
    Parameters
    ----------
    hidden_size : int
        Output dimension of the MLP embedding.
    frequency_embedding_size : int, optional
        Size of the input frequency embedding, by default 256.
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
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
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
        """
        Forward pass for timestep embedding.

        Parameters
        ----------
        t : torch.Tensor
            1D tensor of scalar timesteps.

        Returns
        -------
        torch.Tensor
            The final embedded representation of shape (N, hidden_size).
        """
        t = t.view(-1)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class CategoricalEmbedder(nn.Module):
    """
    Embeds categorical conditions (e.g., data source labels) into vector representations.
    Supports label dropout for classifier-free guidance.

    Parameters
    ----------
    num_classes : int
        Number of distinct label categories.
    hidden_size : int
        Size of the embedding vectors.
    dropout_prob : float
        Probability of label dropout.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels, train, force_drop_ids=None):
        """
        Forward pass for categorical embedding with optional label dropout.

        Parameters
        ----------
        labels : torch.Tensor
            Tensor of categorical labels.
        train : bool
            Whether the model is in training mode.
        force_drop_ids : torch.Tensor or None, optional
            Explicit mask for which labels to drop.

        Returns
        -------
        torch.Tensor
            Embedded label representations, with optional noise added during training.
        """
        labels = labels.long().view(-1)

        use_dropout = self.dropout_prob > 0
        drop_ids = force_drop_ids == 1

        if (train and use_dropout):
            drop_ids_rand = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            if force_drop_ids is not None:
                drop_ids = torch.logical_or(drop_ids, drop_ids_rand)
            else:
                drop_ids = drop_ids_rand
        
        if use_dropout:
            labels = torch.where(drop_ids, self.num_classes, labels)
        embeddings = self.embedding_table(labels)
        if train:
            noise = torch.randn_like(embeddings)
            embeddings = embeddings + noise
        return embeddings
    
class ClusterContinuousEmbedder(nn.Module):
    """
    Embeds continuous input features into vector representations using a multilayer perceptron (MLP).
    Supports optional embedding dropout for classifier-free guidance.

    Parameters
    ----------
    input_size : int
        The size of the input features.
    hidden_size : int
        The size of the output embedding vectors.
    dropout_prob : float
        Probability of embedding dropout, used for classifier-free guidance.

    """
    def __init__(self, input_size, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0

        if use_cfg_embedding:
            self.embedding_drop = nn.Embedding(1, hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.Softmax(dim=1),
            nn.Linear(hidden_size, hidden_size, bias=False)
        )
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if force_drop_ids is not None:
            drop_ids = force_drop_ids == 1
        else:
            drop_ids = None
        if not use_dropout:
            drop_ids = None

        if (train and use_dropout):
            drop_ids_rand = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            if force_drop_ids is not None:
                drop_ids = torch.logical_or(drop_ids, drop_ids_rand)
            else:
                drop_ids = drop_ids_rand
        
        if drop_ids is not None:
            embeddings = torch.zeros((labels.shape[0], self.hidden_size), device=labels.device)
            embeddings[~drop_ids] = self.mlp(labels[~drop_ids])
            embeddings[drop_ids] += self.embedding_drop.weight[0]
        else:
            embeddings = self.mlp(labels)

        if train:
            noise = torch.randn_like(embeddings)
            embeddings = embeddings + noise
        return embeddings