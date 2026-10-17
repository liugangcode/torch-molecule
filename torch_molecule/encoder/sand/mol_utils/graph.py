"""Molecular graph storage and batching."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Union


from torch_geometric.data import Data, Batch

import numpy as np
import torch


class MolecularGraph:
    def __init__(self, x: Union[np.ndarray, torch.Tensor], edge_index: Union[np.ndarray, torch.Tensor], edge_attr: Union[np.ndarray, torch.Tensor], pos: Union[np.ndarray, torch.Tensor, None] = None, **kwargs: Any) -> None:
        """Construct a MolecularGraph.

        Args:
            x: Node feature array (numpy or torch).
            edge_index: Edge connectivity, shape ``(2, num_edges)``.
            edge_attr: Edge feature array.
            pos: Optional atom position array.
            **kwargs: Additional attributes stored as instance attributes.
        """
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        if pos is not None:
            self.pos = pos
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_torch(self) -> "MolecularGraph":
        """Convert core arrays to ``torch.Tensor`` in-place.

        Returns:
            ``self`` for chaining.
        """
        self.x = torch.tensor(self.x).long()
        self.edge_index = torch.tensor(self.edge_index).long()
        self.edge_attr = torch.tensor(self.edge_attr).long()
        if hasattr(self, "pos"):
            if self.pos is not None:
                self.pos = torch.tensor(self.pos).float()
        return self

    def to(self, device: Union[str, torch.device]) -> "MolecularGraph":
        """Move all tensor attributes to ``device``.

        Args:
            device: Target device string or :class:`torch.device`.

        Returns:
            ``self`` for chaining.
        """
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                self.__dict__[key] = value.to(device)
        return self

    def cuda(self) -> "MolecularGraph":
        return self.to("cuda")

    def cpu(self) -> "MolecularGraph":
        return self.to("cpu")

    def to_numpy(self) -> None:
        """Convert all tensor attributes to numpy arrays in-place."""
        def to_numpy(x):
            x = x.detach().cpu().numpy()
            if x.dtype in [np.float32, np.float16]:
                x = x.astype(np.float64)
            return x

        self.x = to_numpy(self.x)
        self.edge_index = to_numpy(self.edge_index)
        self.edge_attr = to_numpy(self.edge_attr)
        if hasattr(self, "pos"):
            self.pos = to_numpy(self.pos)
        if hasattr(self, "batch"):
            self.batch = to_numpy(self.batch)

    def __str__(self) -> str:
        """Return a human-readable summary showing each attribute's shape."""
        string = "Graph("
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                string += f"{key}={value.shape}, "
            elif isinstance(value, torch.Tensor):
                s = str(value.shape)[11:-1]
                string += f"{key}={s}, "
        string = string[:-2] + ")"
        return string

    def __getitem__(self, key: str) -> Any:
        """Return the attribute named ``key``.

        Args:
            key: Attribute name.

        Raises:
            KeyError: If no attribute with that name exists.
        """
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Key {key} not found in Graph object.")

    def __setitem__(self, key: str, value: Any) -> None:
        """Set or add the attribute named ``key`` to ``value``.

        Args:
            key: Attribute name.
            value: Value to assign.
        """
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.__dict__[key] = value

    def __repr__(self) -> str:
        return self.__str__()

    def get(self, key: str, default: Any = None) -> Any:
        """Return the attribute named ``key``, or ``default`` if missing.

        Args:
            key: Attribute name.
            default: Fallback value if the attribute does not exist.

        Returns:
            The attribute value, or ``default``.
        """
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return default

    @classmethod
    def from_dict(cls, data: Dict[str, Union[np.ndarray, torch.Tensor]]) -> "MolecularGraph":
        """Construct a :class:`MolecularGraph` from a dict of array/tensor values.

        Args:
            data: Mapping from attribute name to array or tensor.

        Returns:
            A new :class:`MolecularGraph`.
        """
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the Graph object to a dictionary representation.
        """
        data = {}
        for key, value in self.__dict__.items():
            data[key] = value

        return data

    def to_pyg(self) -> Data:
        """Convert to a :class:`torch_geometric.data.Data` object.

        Returns:
            A :class:`torch_geometric.data.Data` with the same attributes.
        """
        if isinstance(self.x, np.ndarray):
            self.to_torch()
        return Data.from_dict(self.to_dict())

    @classmethod
    def from_graph_list(
        cls, graph_list: Iterable["MolecularGraph"]
    ) -> "MolecularGraph":
        """
        Creates a large disconnected MolecularGraph from a list of MolecularGraphs.
        """
        # use pytorch geometric for the batching
        pyg_graph_list = [g.to_pyg() for g in graph_list]
        pyg_batch = Batch.from_data_list(pyg_graph_list)
        pyg_store: Dict[str, Any] = pyg_batch.__dict__["_store"]
        pyg_store.pop("ptr", None)
        pyg_store["batch_size"] = len(graph_list)
        graph = cls.from_dict(pyg_store)
        return graph

    def select_atoms(self, atom_indices: Iterable) -> "MolecularGraph":
        """
        Selects atoms from the graph based on the provided indices.
        Returns a new Graph object with only the selected atoms and their corresponding edges.
        """
        if isinstance(self.x, np.ndarray):
            if isinstance(atom_indices, torch.Tensor):
                atom_indices = atom_indices.numpy()
            elif isinstance(atom_indices, list):
                atom_indices = np.array(atom_indices)
            edge_idxs = np.isin(self.edge_index, atom_indices).all(axis=0)
        elif isinstance(self.x, torch.Tensor):
            if isinstance(atom_indices, (np.ndarray, list)):
                atom_indices = torch.tensor(atom_indices)
            edge_idxs = torch.isin(self.edge_index, atom_indices).all(axis=0)
        selected_x = self.x[atom_indices]
        selected_edge_index = self.edge_index[:, edge_idxs]
        selected_edge_attr = self.edge_attr[edge_idxs]

        selected_edge_index = selected_edge_index - atom_indices.min()
        assert (
            atom_indices.max() - atom_indices.min() == selected_x.shape[0] - 1
        ), f"Atom indices {atom_indices} do not match selected_x shape {selected_x.shape}. Are atom_indices contiguous?"

        return MolecularGraph(
            x=selected_x,
            edge_index=selected_edge_index,
            edge_attr=selected_edge_attr,
            pos=self.pos[atom_indices] if hasattr(self, "pos") else None,
        )

    def select_graph(
        self, batch_idx: int, batch_key: str = "batch"
    ) -> "MolecularGraph":
        """Extract a single graph from a batched :class:`MolecularGraph`.

        Args:
            batch_idx: Index of the graph to extract from the batch.
            batch_key: Name of the batch-assignment attribute.

        Returns:
            A new :class:`MolecularGraph` containing only atoms belonging to
            ``batch_idx``.
        """
        assert hasattr(
            self, batch_key
        ), f"Graph does not have a '{batch_key}' attribute."
        if isinstance(self.x, np.ndarray):
            atom_indices = np.where(self[batch_key] == batch_idx)[0]
            if atom_indices.size == 0:
                raise ValueError(f"No atoms found for batch idx {batch_idx}.")
        elif isinstance(self.x, torch.Tensor):
            atom_indices = torch.where(self[batch_key] == batch_idx)[0]
            if atom_indices.numel() == 0:
                raise ValueError(f"No atoms found for batch idx {batch_idx}.")
        else:
            raise TypeError(
                f"Unsupported type for x: {type(self.x)}. Expected np.ndarray or torch.Tensor."
            )
        return self.select_atoms(atom_indices)

    @staticmethod
    def _values_equal(a: Union[np.ndarray, torch.Tensor], b: Union[np.ndarray, torch.Tensor]) -> bool:
        """
        Compares two array-like values (np.ndarray or torch.Tensor) for equality.
        Uses a numeric tolerance for floating point dtypes and exact equality
        otherwise. Raises ValueError if the shapes do not match.
        """
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        if isinstance(b, torch.Tensor):
            b = b.detach().cpu().numpy()
        a = np.asarray(a)
        b = np.asarray(b)

        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} != {b.shape}")

        if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
            return bool(np.allclose(a, b))
        return bool(np.array_equal(a, b))

    def __eq__(self, other: object) -> bool:
        """
        Checks whether two MolecularGraph objects represent the same graph by
        comparing x, edge_index, edge_attr and pos (if present on either graph).
        """
        if not isinstance(other, MolecularGraph):
            return NotImplemented

        has_pos_self = hasattr(self, "pos")
        has_pos_other = hasattr(other, "pos")
        if has_pos_self != has_pos_other:
            raise ValueError(
                "Cannot compare graphs: one graph has 'pos' and the other does not."
            )

        if not self._values_equal(self.x, other.x):
            return False
        if not self._values_equal(self.edge_index, other.edge_index):
            return False
        if not self._values_equal(self.edge_attr, other.edge_attr):
            return False
        if has_pos_self and not self._values_equal(self.pos, other.pos):
            return False

        return True
