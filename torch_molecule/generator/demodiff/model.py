import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from . import utils
from . import diffusion as diffusion_utils
from .noise_schedule import NoiseScheduleDiscrete, MarginalTransition
from .transformer import Transformer


class DemoDiffModel(torch.nn.Module):
    """Graph diffusion with clean molecular demonstrations as context."""

    def __init__(
        self,
        num_layer,
        hidden_size,
        num_head,
        mlp_ratio,
        diffusion_steps,
        drop_context,
        noise_weight,
        lambda_train,
        dataset_info,
        diffusion_noise_schedule="cosine",
        guide_scale=2.0,
        no_context_train=False,
    ):
        super().__init__()
        self.T = diffusion_steps
        self.guide_scale = guide_scale
        self.no_context_train = no_context_train
        self.drop_context = drop_context
        self.noise_weight = noise_weight
        self.lambda_train = lambda_train
        self.max_n_nodes = dataset_info["max_n_nodes"]
        self.max_node_type = len(dataset_info["node_types"])
        self.max_bond_type = len(dataset_info["edge_types"])
        self.max_position_type = len(dataset_info["pos_types"])
        for name, key in (
            ("x_marginals", "node_types"),
            ("e_marginals", "edge_types"),
            ("pos_marginals", "pos_types"),
            ("node_counts", "node_counts"),
        ):
            values = torch.as_tensor(dataset_info[key], dtype=torch.float32)
            self.register_buffer(name, values / values.sum())

        self.model = Transformer(
            max_n_nodes=self.max_n_nodes,
            hidden_size=hidden_size,
            depth=num_layer,
            num_heads=num_head,
            mlp_ratio=mlp_ratio,
            X_dim=self.max_node_type,
            E_dim=self.max_bond_type,
            y_dim=None,
            pos_dim=self.max_position_type,
        )
        self.noise_schedule = NoiseScheduleDiscrete(diffusion_noise_schedule, diffusion_steps)
        self.transition_model = MarginalTransition(
            self.x_marginals,
            self.e_marginals,
            torch.as_tensor(dataset_info["xe_conditions"], dtype=torch.float32),
            torch.as_tensor(dataset_info["ex_conditions"], dtype=torch.float32),
            self.pos_marginals,
            self.max_n_nodes,
        )
        self.limit_dist = utils.PlaceHolder(
            X=self.x_marginals, E=self.e_marginals, pos=self.pos_marginals
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def model_dtype(self):
        return next(self.parameters()).dtype

    def denoising_loss(self, pred, target, noise_weight=None):
        """Mean cross-entropy for fragment, bond, and attachment-position labels."""
        losses = []
        for prediction, truth in ((pred.X, target.X), (pred.E, target.E), (pred.pos, target.pos)):
            valid = truth.any(dim=-1)
            # Single-fragment batches have no edge or attachment-position targets.
            if not valid.any():
                losses.append(prediction.sum() * 0)
                continue
            values = F.cross_entropy(prediction[valid], truth[valid].argmax(-1), reduction="none")
            if noise_weight is not None:
                shape = (truth.shape[0],) + (1,) * (truth.ndim - 2)
                weights = noise_weight.reshape(shape).expand(truth.shape[:-1])
                values = values * weights[valid]
            losses.append(values.mean())
        return sum(weight * loss for weight, loss in zip(self.lambda_train, losses))

    def compute_loss(self, batch):
        graphs = batch["molecular_graphs"]
        noisy_target, context = self.batch_data_process(
            graphs, batch["context_indicators"], graphs.batch, batch["score"]
        )
        prediction = self.forward(noisy_target, context)
        if self.noise_weight == "noiser_lower":
            noise_weight = noisy_target["alpha_t_bar"]
        elif self.noise_weight == "noiser_higher":
            noise_weight = 1 - noisy_target["alpha_t_bar"]
        else:
            noise_weight = None
        return self.denoising_loss(prediction, context["target_data"], noise_weight)

    def forward(self, noisy_target, dense_context_dict=None, unconditioned=None):
        if unconditioned is None:
            rand = torch.rand(1).item()
            unconditioned = self.no_context_train or (rand < self.drop_context)

        if dense_context_dict is not None and not unconditioned:
            score = dense_context_dict["score"]
            indicator_node_to_mol = dense_context_dict["indicator_node_to_mol"]
            indicator_mol_to_ctx = dense_context_dict["indicator_mol_to_ctx"]
            node_mask_allmol = dense_context_dict["node_mask_allmol"]
            node_mask_target = dense_context_dict["node_mask_target"]
            dense_data = dense_context_dict["dense_data"]

            noisy_data = utils.replace_first_indices(dense_data, indicator_mol_to_ctx, noisy_target)
            indicator_node_to_ctx = indicator_mol_to_ctx[indicator_node_to_mol]
            score = score[indicator_node_to_mol].view(-1, 1)

            noisy_X, node_mask_context = to_dense_batch(
                x=noisy_data.X[node_mask_allmol], batch=indicator_node_to_ctx
            )
            bs, n, _, de = noisy_data.E.size()
            noisy_E = noisy_data.E.view(bs, n, n * self.max_bond_type)[node_mask_allmol]
            noisy_E, _ = to_dense_batch(x=noisy_E, batch=indicator_node_to_ctx)
            noisy_pos = noisy_data.pos.view(bs, n, n * self.max_position_type)[node_mask_allmol]
            noisy_pos, _ = to_dense_batch(x=noisy_pos, batch=indicator_node_to_ctx)

            score, _ = to_dense_batch(x=score, batch=indicator_node_to_ctx)
            noisy_X, noisy_E, noisy_pos = noisy_X.float(), noisy_E.float(), noisy_pos.float()
            node_mask_dict = {"target": node_mask_target, "all": node_mask_context}
            batch_max_n_nodes = n
        else:
            noisy_X, noisy_E, noisy_pos = (
                noisy_target["X_t"].to(self.model_dtype),
                noisy_target["E_t"].to(self.model_dtype),
                noisy_target["pos_t"].to(self.model_dtype),
            )
            bs, n, _, de = noisy_E.size()
            noisy_E = noisy_E.view(bs, n, n * self.max_bond_type)
            noisy_pos = noisy_pos.view(bs, n, n * self.max_position_type)
            node_mask_dict = {"target": noisy_target["node_mask"], "all": noisy_target["node_mask"]}
            score = torch.ones(noisy_X.size(0), noisy_X.size(1), 1).type_as(noisy_X)
            batch_max_n_nodes = n

        target_n_nodes = noisy_target["X_t"].size(1)
        t = noisy_target["t"] * self.T
        score[~node_mask_dict["all"]] = 0

        pred_X, pred_E, pred_pos = self.model(
            noisy_X, noisy_E, noisy_pos, score, node_mask_dict, target_n_nodes, t=t
        )
        pred = utils.PlaceHolder(X=pred_X, E=pred_E, y=None, pos=pred_pos)

        return pred

    def batch_data_process(
        self,
        mol_data,
        indicator_mol_to_ctx,
        indicator_node_to_mol,
        score,
        apply_noise=True,
    ):
        mol_x = F.one_hot(mol_data.x, num_classes=self.max_node_type).to(self.device)
        mol_edge_attr = F.one_hot(mol_data.edge_attr, num_classes=self.max_bond_type).to(
            self.device
        )
        mol_edge_pos = F.one_hot(mol_data.edge_pos, num_classes=self.max_position_type).to(
            self.device
        )
        mol_edge_index = mol_data.edge_index.to(self.device)
        mol_y = mol_data.y.to(self.device) if mol_data.y is not None else None
        indicator_mol_to_ctx = indicator_mol_to_ctx.to(self.device)
        indicator_node_to_mol = indicator_node_to_mol.to(self.device)
        score = score.to(self.device)

        dense_data, node_mask_allmol = utils.to_dense(
            mol_x,
            mol_edge_index,
            mol_edge_attr,
            indicator_node_to_mol,
            y=mol_y,
            edge_pos=mol_edge_pos,
        )
        dense_data = dense_data.mask(node_mask_allmol)
        target_data, node_mask_target = utils.select_first_indices(
            dense_data, node_mask_allmol, indicator_mol_to_ctx
        )

        dense_context_dict = {
            "indicator_node_to_mol": indicator_node_to_mol,
            "indicator_mol_to_ctx": indicator_mol_to_ctx,
            "score": score,
            "node_mask_allmol": node_mask_allmol,
            "dense_data": dense_data,
            "node_mask_target": node_mask_target,
            "target_data": target_data,
        }

        if apply_noise:
            noisy_target = self.apply_noise(
                target_data.X, target_data.E, target_data.y, target_data.pos, node_mask_target
            )
            return noisy_target, dense_context_dict
        else:
            return None, dense_context_dict

    def apply_noise(self, X, E, y, pos, node_mask):
        """Sample noise and apply it to the data."""
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device
        ).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        beta_t = self.noise_schedule(t_normalized=t_float)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)  # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(
            alpha_t_bar, self.device, n_nodes=X.size(1)
        )  # (bs, dx_in, dx_out), (bs, de_in, de_out)

        bs, n, d = X.shape
        X_all = torch.cat([X, E.reshape(bs, n, -1)], dim=-1).to(self.model_dtype)
        prob_all = X_all @ Qtb.X
        probX = prob_all[:, :, : self.max_node_type]
        probE = prob_all[:, :, self.max_node_type :].reshape(bs, n, n, -1)
        prob_pos = pos.to(probE.dtype) @ Qtb.pos.unsqueeze(1)

        sampled_t = diffusion_utils.sample_discrete_features(
            probX=probX, probE=probE, prob_pos=prob_pos, node_mask=node_mask
        )

        X_t = F.one_hot(sampled_t.X, num_classes=self.max_node_type)
        E_t = F.one_hot(sampled_t.E, num_classes=self.max_bond_type)
        pos_t = F.one_hot(sampled_t.pos, num_classes=self.max_position_type)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        y_t = y
        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y_t, pos=pos_t).type_as(X_t).mask(node_mask)

        noisy_data = {
            "t_int": t_int,
            "t": t_float,
            "beta_t": beta_t,
            "alpha_s_bar": alpha_s_bar,
            "alpha_t_bar": alpha_t_bar,
            "X_t": z_t.X,
            "E_t": z_t.E,
            "y_t": z_t.y,
            "pos_t": z_t.pos,
            "node_mask": node_mask,
        }
        return noisy_data

    @torch.no_grad()
    def sample_batch(self, batch_data, init_node="reconstruction"):
        """Return generated and target fragment, bond, and attachment-position graphs."""
        mol_data = batch_data["molecular_graphs"]
        indicator_mol_to_ctx = batch_data["context_indicators"]
        indicator_node_to_mol = mol_data.batch
        score = batch_data["score"]
        _, dense_context_dict = self.batch_data_process(
            mol_data,
            indicator_mol_to_ctx,
            indicator_node_to_mol,
            score,
            apply_noise=False,
        )
        batch_size = indicator_mol_to_ctx[-1] + 1

        if init_node == "reconstruction" or init_node is None:
            node_mask_generated = dense_context_dict["node_mask_target"]
            n_nodes_generated = node_mask_generated.sum(-1)
            node_mask_true = dense_context_dict["node_mask_target"]
        elif init_node == "random" or type(init_node) in [int, torch.Tensor]:
            if init_node == "random":
                n_nodes_generated = torch.multinomial(
                    self.node_counts, int(batch_size), replacement=True
                )
            elif type(init_node) == int:
                n_nodes_generated = init_node * torch.ones(
                    batch_size, device=self.device, dtype=torch.int
                )
            else:
                assert isinstance(init_node, torch.Tensor)
                n_nodes_generated = init_node
            arange = (
                torch.arange(self.max_n_nodes, device=self.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            node_mask_generated = arange < n_nodes_generated.unsqueeze(1)
            dense_context_dict["node_mask_target"] = node_mask_generated
            node_mask_true = dense_context_dict["node_mask_target"]

        n_nodes_true = node_mask_true.sum(-1)

        limit_dist = self.limit_dist.to(self.device)
        sampled_s = diffusion_utils.sample_discrete_feature_noise(
            limit_dist=limit_dist, node_mask=node_mask_generated
        )
        assert (sampled_s.E == torch.transpose(sampled_s.E, 1, 2)).all()

        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).to(self.device)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            sampled_s = self.sample_p_zs_given_zt(
                s_norm,
                t_norm,
                sampled_s.X,
                sampled_s.E,
                sampled_s.pos,
                node_mask_generated,
                dense_context_dict,
            )

        sampled_s = sampled_s.mask(node_mask_generated, collapse=True)
        molecule_list = []
        for i in range(batch_size):
            n = n_nodes_generated[i]
            atom_types = sampled_s.X[i, :n].cpu()
            edge_types = sampled_s.E[i, :n, :n].cpu()
            positions = sampled_s.pos[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types, positions])

        target_data = dense_context_dict["target_data"]
        target_list = []
        target_data = target_data.mask(node_mask_true, collapse=True)
        for i in range(batch_size):
            n = n_nodes_true[i]
            atom_types = target_data.X[i, :n].cpu()
            edge_types = target_data.E[i, :n, :n].cpu()
            positions = target_data.pos[i, :n, :n].cpu()
            target_list.append([atom_types, edge_types, positions])

        return molecule_list, target_list

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, pos_t, node_mask, dense_context_dict):
        """Sample one reverse diffusion step, with optional demonstration guidance."""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        noisy_data = {"X_t": X_t, "E_t": E_t, "pos_t": pos_t, "t": t, "node_mask": node_mask}

        def get_prob(noisy_data, unconditioned=False):
            pred = self.forward(noisy_data, dense_context_dict, unconditioned=unconditioned)

            pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
            pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0
            pred_pos = F.softmax(pred.pos, dim=-1)  # bs, n, n, dpos

            n_nodes = pred_X.size(1)
            Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device, n_nodes=n_nodes)
            Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device, n_nodes=n_nodes)
            Qt = self.transition_model.get_Qt(beta_t, self.device, n_nodes=n_nodes)

            Xt_all = torch.cat([X_t, E_t.reshape(bs, n, -1)], dim=-1)
            pred_all = torch.cat([pred_X, pred_E.reshape(bs, n, -1)], dim=-1)

            unnormalized_probX_all = diffusion_utils.reverse_diffusion(
                predX_0=pred_all, X_t=Xt_all, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X
            )
            unnormalized_prob_X = unnormalized_probX_all[:, :, : self.max_node_type]
            unnormalized_prob_E = unnormalized_probX_all[:, :, self.max_node_type :].reshape(
                bs, n * n, -1
            )
            unnormalized_prob_pos = diffusion_utils.reverse_diffusion(
                predX_0=pred_pos.flatten(start_dim=1, end_dim=-2),
                X_t=pos_t.flatten(start_dim=1, end_dim=-2),
                Qt=Qt.pos,
                Qsb=Qsb.pos,
                Qtb=Qtb.pos,
            )
            unnormalized_prob_pos = unnormalized_prob_pos.reshape(bs, n, n, pred_pos.shape[-1])

            unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
            unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
            unnormalized_prob_pos[torch.sum(unnormalized_prob_pos, dim=-1) == 0] = 1e-5

            prob_X = unnormalized_prob_X / torch.sum(
                unnormalized_prob_X, dim=-1, keepdim=True
            )  # bs, n, d_t-1
            prob_E = unnormalized_prob_E / torch.sum(
                unnormalized_prob_E, dim=-1, keepdim=True
            )  # bs, n, d_t-1
            prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])
            prob_pos = unnormalized_prob_pos / torch.sum(
                unnormalized_prob_pos, dim=-1, keepdim=True
            )
            prob_pos = prob_pos.reshape(bs, n, n, pred_pos.shape[-1])
            return prob_X, prob_E, prob_pos

        prob_X, prob_E, prob_pos = get_prob(noisy_data, unconditioned=False)

        if self.guide_scale != 1:
            uncon_prob_X, uncon_prob_E, uncon_prob_pos = get_prob(noisy_data, unconditioned=True)
            prob_X = uncon_prob_X * (prob_X / uncon_prob_X.clamp_min(1e-10)) ** self.guide_scale
            prob_E = uncon_prob_E * (prob_E / uncon_prob_E.clamp_min(1e-10)) ** self.guide_scale
            prob_pos = (
                uncon_prob_pos * (prob_pos / uncon_prob_pos.clamp_min(1e-10)) ** self.guide_scale
            )
            prob_X = prob_X / prob_X.sum(dim=-1, keepdim=True).clamp_min(1e-10)
            prob_E = prob_E / prob_E.sum(dim=-1, keepdim=True).clamp_min(1e-10)
            prob_pos = prob_pos / prob_pos.sum(dim=-1, keepdim=True).clamp_min(1e-10)

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_pos.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(
            prob_X, prob_E, prob_pos, node_mask=node_mask
        )

        X_s = F.one_hot(sampled_s.X, num_classes=self.max_node_type).to(self.model_dtype)
        E_s = F.one_hot(sampled_s.E, num_classes=self.max_bond_type).to(self.model_dtype)
        pos_s = F.one_hot(sampled_s.pos, num_classes=self.max_position_type).to(self.model_dtype)

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=None, pos=pos_s)
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=None, pos=pos_s)

        return out_one_hot.mask(node_mask)
