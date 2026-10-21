import torch
from torch_geometric.utils import to_dense_adj, to_dense_batch


def to_dense(x, edge_index, edge_attr, batch, y=None, max_num_nodes=None, edge_pos=None):
    X, node_mask = to_dense_batch(x=x, batch=batch, max_num_nodes=max_num_nodes)
    if max_num_nodes is None:
        max_num_nodes = X.size(1)
    E = to_dense_adj(
        edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes
    )
    E = encode_no_edge(E)

    if edge_pos is not None:
        pos = to_dense_adj(
            edge_index=edge_index, batch=batch, edge_attr=edge_pos, max_num_nodes=max_num_nodes
        )
        pos = encode_no_edge(pos)
    else:
        pos = None
    dense_data = PlaceHolder(X=X, E=E, pos=pos, y=y)
    return dense_data, node_mask


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


class PlaceHolder:
    def __init__(self, X, E, y=None, pos=None):
        self.X = X
        self.E = E
        self.y = y
        self.pos = pos

    def type_as(self, x: torch.Tensor):
        """Changes the device and dtype of X, E, y."""
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        return self

    def to(self, x: str):
        """Changes the device and dtype of X, E, y."""
        self.X = self.X.to(x)
        self.E = self.E.to(x)
        if self.y is not None:
            self.y = self.y.to(x)
        if self.pos is not None:
            self.pos = self.pos.to(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = -1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
            if self.pos is not None:
                self.pos = torch.argmax(self.pos, dim=-1)
                self.pos[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            if self.pos is not None:
                self.pos = self.pos * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self


def select_first_indices(dense_data, node_mask, indicator):
    """
    Get first occurrence indices of sequential values (0,1,2...) from indicator tensor
    and select corresponding elements from dense_data.
    """
    unique_values = torch.unique(indicator)
    first_indices = torch.tensor(
        [torch.where(indicator == value)[0][0].item() for value in unique_values],
        device=indicator.device,
    )

    selected_mask = node_mask[first_indices]
    selected_X = dense_data.X[first_indices]
    selected_E = dense_data.E[first_indices]
    selected_pos = dense_data.pos[first_indices]
    selected_y = dense_data.y[first_indices] if dense_data.y is not None else None

    return PlaceHolder(X=selected_X, E=selected_E, y=selected_y, pos=selected_pos), selected_mask


def replace_first_indices(dense_data, indicator, noisy_data):
    """
    Get first occurrence indices of sequential values (0,1,2...) from indicator tensor
    and replace corresponding elements from noisy_data.
    """
    unique_values = torch.unique(indicator)
    first_indices = torch.tensor(
        [torch.where(indicator == value)[0][0].item() for value in unique_values],
        device=indicator.device,
    )

    new_X = dense_data.X.clone()
    new_X[first_indices] = noisy_data["X_t"].to(new_X.dtype)

    new_E = dense_data.E.clone()
    new_E[first_indices] = noisy_data["E_t"].to(new_E.dtype)

    new_pos = dense_data.pos.clone()
    new_pos[first_indices] = noisy_data["pos_t"].to(new_pos.dtype)
    new_y = dense_data.y.clone() if dense_data.y is not None else None
    return PlaceHolder(X=new_X, E=new_E, y=new_y, pos=new_pos)
