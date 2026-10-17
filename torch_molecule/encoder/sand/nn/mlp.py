"""
MLP module
"""

from typing import Callable, Dict, Literal, Optional, Union

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multilayer perceptron block

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        hidden_dim (int): Dimension of hidden layers.
        num_layers (int): Number of layers. Defaults to 3.
        activation (Callable, optional): Activation function.
            Defaults to nn.functional.silu.
        norm (Optional[str], optional): Normalization type.
            Defaults to None.
        dropout_p (float, optional): Dropout probability.
            Defaults to 0.0.
        feature_key (str, optional): Key of features in input dictionary.
            Defaults to None.
        raw_last_layer (bool, optional): Whether to not apply norm, activation and
            dropout on last layer. Defaults to True.
    """

    _NORM_TYPE = Optional[Literal["batch", "layer"]]

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        activation: Callable = nn.functional.silu,
        norm: _NORM_TYPE = None,
        dropout_p: float = 0.0,
        feature_key: Optional[str] = None,
        raw_last_layer: bool = True,
        out_norm: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        if num_layers < 3:
            raise ValueError(
                "Number of layers must be at least 3. Use nn.Linear instead."
            )
        self.in_layer = nn.Linear(in_dim, hidden_dim)
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.norm_type = norm
        self.raw_last_layer = raw_last_layer
        self.hidden_layers = self.get_hidden_layers()
        self.norm_layers = self.get_norm_layers()
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else None
        self.out_layer = nn.Linear(hidden_dim, out_dim)
        self.activation = activation
        self.feature_key = feature_key
        self.out_norm = out_norm

    def get_hidden_layers(self) -> nn.ModuleList:
        """Build and return the intermediate hidden linear layers."""
        hidden_layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, self.hidden_dim)
                for _ in range(self.num_layers - 2)
            ]
        )
        return hidden_layers

    def get_norm_layers(self) -> Optional[nn.ModuleList]:
        """Build and return the normalisation layers, or ``None`` if not used."""
        if self.norm_type == "batch":
            nl = nn.BatchNorm1d
        elif self.norm_type == "layer":
            nl = nn.LayerNorm
        elif self.norm_type is None:
            return None
        else:
            raise ValueError("Normalization must be one of ['batch', 'layer']")
        norm_layers = nn.ModuleList(
            [nl(self.hidden_dim) for _ in range(self.num_layers - 1)]
        )
        if not self.raw_last_layer:
            norm_layers += [nl(self.out_dim)]
        return norm_layers

    def apply_norm_activation_dropout(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """Apply normalisation, activation, and dropout at layer index ``i``."""
        if self.norm_layers is not None:
            x = self.norm_layers[i](x)
        x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
        return x

    def forward(self, x: Union[Dict, torch.Tensor]) -> torch.Tensor:
        """Compute a forward pass through the MLP.

        Args:
            x (Union[Dict, torch.Tensor]): Input tensor of shape
                ``(batch, in_dim)``, or a dictionary from which the tensor is
                extracted using ``self.feature_key``.

        Returns:
            torch.Tensor: Output tensor of shape ``(batch, out_dim)``.
        """
        if self.feature_key:
            x = x[self.feature_key]
        x = self.in_layer(x)
        x = self.apply_norm_activation_dropout(x, 0)

        for i, hidden_layer in enumerate(self.hidden_layers):
            x = hidden_layer(x)
            x = self.apply_norm_activation_dropout(x, i + 1)

        pred = self.out_layer(x)
        if not self.raw_last_layer:
            pred = self.apply_norm_activation_dropout(pred, i + 2)
        if self.out_norm:
            pred = self.out_norm(pred)
        return pred
