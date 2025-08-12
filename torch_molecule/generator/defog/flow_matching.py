import torch
import numpy as np
from scipy.stats import norm
from scipy.special import beta as beta_func, betaln
from scipy.stats import beta as sp_beta
from scipy.interpolate import interp1d
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
import math

from .utils import PlaceHolder

class NoiseDistribution:
    """Discrete noise/limit distribution for DeFoG.

    Parameters
    ----------
    model_transition : str
        Transition type controlling the class limit distributions and potential
        addition of virtual absorbing classes. Supported options:
        - 'uniform': uniform limit over classes
        - 'absorbfirst': absorb into the first class only
        - 'argmax': absorb into the most frequent class (per dataset)
        - 'absorbing': add a virtual absorbing class (last index)
        - 'marginal': dataset marginal distributions (default)
        - 'edge_marginal': uniform X, marginal E
        - 'node_marginal': marginal X, uniform E
    """
    def __init__(self, model_transition, dataset_infos):

        self.x_num_classes = dataset_infos.output_dims["X"]
        self.e_num_classes = dataset_infos.output_dims["E"]
        self.y_num_classes = dataset_infos.output_dims["y"]
        self.x_added_classes = 0
        self.e_added_classes = 0
        self.y_added_classes = 0
        self.transition = model_transition

        if model_transition == "uniform":
            x_limit = torch.ones(self.x_num_classes) / self.x_num_classes
            e_limit = torch.ones(self.e_num_classes) / self.e_num_classes

        elif model_transition == "absorbfirst":
            x_limit = torch.zeros(self.x_num_classes)
            x_limit[0] = 1
            e_limit = torch.zeros(self.e_num_classes)
            e_limit[0] = 1

        elif model_transition == "argmax":
            node_types = dataset_infos.node_types.float()
            x_marginals = node_types / torch.sum(node_types)

            edge_types = dataset_infos.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)

            x_max_dim = torch.argmax(x_marginals)
            e_max_dim = torch.argmax(e_marginals)
            x_limit = torch.zeros(self.x_num_classes)
            x_limit[x_max_dim] = 1
            e_limit = torch.zeros(self.e_num_classes)
            e_limit[e_max_dim] = 1

        elif model_transition == "absorbing":
            # only add virtual classes when there are several
            if self.x_num_classes > 1:
                # if self.x_num_classes >= 1:
                self.x_num_classes += 1
                self.x_added_classes = 1
            if self.e_num_classes > 1:
                self.e_num_classes += 1
                self.e_added_classes = 1

            x_limit = torch.zeros(self.x_num_classes)
            x_limit[-1] = 1
            e_limit = torch.zeros(self.e_num_classes)
            e_limit[-1] = 1

        elif model_transition == "marginal":

            node_types = dataset_infos.node_types.float()
            x_limit = node_types / torch.sum(node_types)

            edge_types = dataset_infos.edge_types.float()
            e_limit = edge_types / torch.sum(edge_types)

        elif model_transition == "edge_marginal":
            x_limit = torch.ones(self.x_num_classes) / self.x_num_classes

            edge_types = dataset_infos.edge_types.float()
            e_limit = edge_types / torch.sum(edge_types)

        elif model_transition == "node_marginal":
            e_limit = torch.ones(self.e_num_classes) / self.e_num_classes

            node_types = dataset_infos.node_types.float()
            x_limit = node_types / torch.sum(node_types)

        else:
            raise ValueError(f"Unknown transition model: {model_transition}")

        y_limit = torch.ones(self.y_num_classes) / self.y_num_classes  # typically dummy
        print(
            f"Limit distribution of the classes | Nodes: {x_limit} | Edges: {e_limit}"
        )
        self.limit_dist = PlaceHolder(X=x_limit, E=e_limit, y=y_limit)

    def update_input_output_dims(self, input_dims):
        input_dims["X"] += self.x_added_classes
        input_dims["E"] += self.e_added_classes
        input_dims["y"] += self.y_added_classes

    def update_dataset_infos(self, dataset_infos):
        if hasattr(dataset_infos, "atom_decoder"):
            dataset_infos.atom_decoder = (
                dataset_infos.atom_decoder + ["Y"] * self.x_added_classes
            )

    def get_limit_dist(self):
        return self.limit_dist

    def get_noise_dims(self):
        return {
            "X": len(self.limit_dist.X),
            "E": len(self.limit_dist.E),
            "y": len(self.limit_dist.E),
        }

    def ignore_virtual_classes(self, X, E, y=None):
        if self.transition == "absorbing":
            new_X = X[..., : -self.x_added_classes]
            new_E = E[..., : -self.e_added_classes]
            new_y = y[..., : -self.y_added_classes] if y is not None else None
            return new_X, new_E, new_y
        else:
            return X, E, y

    def add_virtual_classes(self, X, E, y=None):
        x_virtual = torch.zeros_like(X[..., :1]).repeat(1, 1, self.x_added_classes)
        new_X = torch.cat([X, x_virtual], dim=-1)

        e_virtual = torch.zeros_like(E[..., :1]).repeat(1, 1, 1, self.e_added_classes)
        new_E = torch.cat([E, e_virtual], dim=-1)

        if y is not None:
            y_virtual = torch.zeros_like(y[..., :1]).repeat(1, self.y_added_classes)
            new_y = torch.cat([y, y_virtual], dim=-1)
        else:
            new_y = None

        return new_X, new_E, new_y



def beta_pdf(x, alpha, beta):
    """Beta distribution PDF."""
    # coeff = np.exp(betaln(alpha, beta))
    # return x ** (alpha - 1) * (1 - x) ** (beta - 1) / coeff
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / beta_func(alpha, beta)


def objective_function(alpha, beta, y, t):
    """Objective function to minimize (mean squared error)."""
    y_pred = beta_pdf(t, alpha, beta)
    regularization = (alpha + beta) + (1 / alpha + 1 / beta)
    error = np.mean((y - y_pred) ** 2)
    error = error + 0.0001 * regularization
    return error


class TimeDistorter:
    """Time distortion schedules used for training/sampling.

    Parameters
    ----------
    train_distortion : str
        Distortion used during training time sampling. Supported options:
        - 'identity': f(t) = t
        - 'cos': f(t) = (1 - cos(pi t)) / 2
        - 'revcos': f(t) = 2 t - (1 - cos(pi t)) / 2
        - 'polyinc': f(t) = t^2
        - 'polydec': f(t) = 2 t - t^2 (default)
        - 'beta' (unsupported for now)
        - 'logitnormal' (unsupported for now)
    sample_distortion : str
        Distortion used during generation time; same options as above.
    alpha, beta : float
        Optional parameters for future schedules (e.g., beta), kept for API compatibility.
    """

    def __init__(
        self,
        train_distortion,
        sample_distortion,
        mu=0,
        sigma=1,
        alpha=1,
        beta=1,
    ):
        self.train_distortion = train_distortion  # used for sample_ft
        self.sample_distortion = sample_distortion  # used for get_ft
        self.alpha = alpha
        self.beta = beta
        print(
            f"TimeDistorter: train_distortion={train_distortion}, sample_distortion={sample_distortion}"
        )
        self.f_inv = None

    def train_ft(self, batch_size, device):
        t_uniform = torch.rand((batch_size, 1), device=device)
        t_distort = self.apply_distortion(t_uniform, self.train_distortion)

        return t_distort

    def sample_ft(self, t, sample_distortion):
        t_distort = self.apply_distortion(t, sample_distortion)
        return t_distort

    def fit(self, difficulty, t_array, learning_rate=0.01, iterations=1000):
        """Fit a beta distribution to data using the method of moments."""
        alpha, beta = self.alpha, self.beta
        t_array = t_array + 1e-6  # Avoid division by zero

        for _ in range(iterations):
            y_pred = beta_pdf(t_array, alpha, beta)

            # Numerical approximation of the gradients
            epsilon = 1e-5
            grad_alpha = (
                objective_function(alpha + epsilon, beta, difficulty, t_array)
                - objective_function(alpha - epsilon, beta, difficulty, t_array)
            ) / (2 * epsilon)
            grad_beta = (
                objective_function(alpha, beta + epsilon, difficulty, t_array)
                - objective_function(alpha, beta - epsilon, difficulty, t_array)
            ) / (2 * epsilon)

            # # Add regularization gradient components
            # grad_alpha += learning_rate * (1 - 1 / alpha**2)
            # grad_beta += learning_rate * (1 + 1 / beta**2)

            # Update parameters
            alpha -= learning_rate * grad_alpha
            beta -= learning_rate * grad_beta

            alpha = min(max(0.3, alpha), 3)
            beta = min(max(0.3, beta), 3)

        y_pred = beta_pdf(t_array, alpha, beta)
        self.approximate_f_inverse(alpha, beta)

        return y_pred, alpha, beta

    def approximate_f_inverse(self, alpha, beta):
        # Generate data points
        t_values = np.linspace(0, 1, 100000)
        f_values = sp_beta.cdf(t_values, alpha, beta)

        # Sort and remove duplicates
        sorted_indices = np.argsort(f_values)
        f_values_sorted = f_values[sorted_indices]
        t_values_sorted = t_values[sorted_indices]

        # Remove duplicates
        _, unique_indices = np.unique(f_values_sorted, return_index=True)
        f_values_unique = f_values_sorted[unique_indices]
        t_values_unique = t_values_sorted[unique_indices]

        # Create the interpolation function for the inverse
        f_inv = interp1d(
            f_values_unique,
            t_values_unique,
            bounds_error=False,
            fill_value="extrapolate",
        )

        self.f_inv = f_inv

    def apply_distortion(self, t, distortion_type):
        assert torch.all((t >= 0) & (t <= 1)), "t must be in the range (0, 1)"

        if distortion_type == "identity":
            ft = t
        elif distortion_type == "cos":
            ft = (1 - torch.cos(t * torch.pi)) / 2
        elif distortion_type == "revcos":
            ft = 2 * t - (1 - torch.cos(t * torch.pi)) / 2
        elif distortion_type == "polyinc":
            ft = t**2
        elif distortion_type == "polydec":
            ft = 2 * t - t**2
        elif distortion_type == "beta":
            raise ValueError(f"Unsupported for now: {distortion_type}")
        elif distortion_type == "logitnormal":
            raise ValueError(f"Unsupported for now: {distortion_type}")
        else:
            raise ValueError(f"Unknown distortion type: {distortion_type}")

        return ft


class RateMatrixDesigner:
    """Design of the discrete rate matrix R(t).

    Parameters
    ----------
    rdb : str
        Design family for R^db. Supported options:
        - 'general': dense stochastic matrix scaled by eta
        - 'marginal': mask guided by dataset marginals
        - 'column': select one target column per (node/edge)
        - 'entry': swap entries based on a single index
    rdb_crit : str
        Criterion used by 'column'/'entry' designs only:
        - 'max_marginal', 'x_t', 'abs_state', 'p_x1_g_xt', 'x_1', 'p_xt_g_x1', 'xhat_t', 'first'
    eta : float
        Stochasticity scaling for R^db (>= 0).
    omega : float
        Target-guidance scaling for R^tg (>= 0).
    limit_dist : PlaceHolder
        Limit distributions for X/E used by various criteria.
    """

    def __init__(self, rdb, rdb_crit, eta, omega, limit_dist):

        self.omega = omega  # target guidance
        self.eta = eta  # stochasticity
        # Different designs of R^db
        self.rdb = rdb
        self.rdb_crit = rdb_crit
        self.limit_dist = limit_dist
        self.num_classes_X = len(self.limit_dist.X)
        self.num_classes_E = len(self.limit_dist.E)

        print(
            f"RateMatrixDesigner: rdb={rdb}, rdb_crit={rdb_crit}, eta={eta}, omega={omega}"
        )

    def compute_graph_rate_matrix(self, t, node_mask, G_t, G_1_pred):

        X_t, E_t = G_t
        X_1_pred, E_1_pred = G_1_pred

        X_t_label = X_t.argmax(-1, keepdim=True)
        E_t_label = E_t.argmax(-1, keepdim=True)
        sampled_G_1 = sample_discrete_features(
            X_1_pred,
            E_1_pred,
            node_mask=node_mask,
        )
        X_1_sampled = sampled_G_1.X
        E_1_sampled = sampled_G_1.E

        dfm_variables = self.compute_dfm_variables(
            t, X_t_label, E_t_label, X_1_sampled, E_1_sampled
        )

        Rstar_t_X, Rstar_t_E = self.compute_Rstar(dfm_variables)

        Rdb_t_X, Rdb_t_E = self.compute_RDB(
            X_t_label,
            E_t_label,
            X_1_pred,
            E_1_pred,
            X_1_sampled,
            E_1_sampled,
            node_mask,
            t,
            dfm_variables,
        )

        Rtg_t_X, Rtg_t_E = self.compute_R_tg(
            X_1_sampled,
            E_1_sampled,
            X_t_label,
            E_t_label,
            dfm_variables,
        )

        # sum to get the final R_t_X and R_t_E
        R_t_X = Rstar_t_X + Rdb_t_X + Rtg_t_X
        R_t_E = Rstar_t_E + Rdb_t_E + Rtg_t_E

        # Stabilize rate matrices
        R_t_X, R_t_E = self.stabilize_rate_matrix(R_t_X, R_t_E, dfm_variables)

        return R_t_X, R_t_E

    def compute_dfm_variables(self, t, X_t_label, E_t_label, X_1_sampled, E_1_sampled):

        dt_p_vals_X, dt_p_vals_E = dt_p_xt_g_x1(
            X_1_sampled,
            E_1_sampled,
            self.limit_dist,
        )  #  (bs, n, dx), (bs, n, n, de)

        dt_p_vals_at_Xt = dt_p_vals_X.gather(-1, X_t_label).squeeze(-1)  # (bs, n, )
        dt_p_vals_at_Et = dt_p_vals_E.gather(-1, E_t_label).squeeze(-1)  # (bs, n, n, )

        pt_vals_X, pt_vals_E = p_xt_g_x1(
            X_1_sampled,
            E_1_sampled,
            t,
            self.limit_dist,
        )  # (bs, n, dx), (bs, n, n, de)

        pt_vals_at_Xt = pt_vals_X.gather(-1, X_t_label).squeeze(-1)  # (bs, n, )
        pt_vals_at_Et = pt_vals_E.gather(-1, E_t_label).squeeze(-1)  # (bs, n, n, )

        Z_t_X = torch.count_nonzero(pt_vals_X, dim=-1)  # (bs, n)
        Z_t_E = torch.count_nonzero(pt_vals_E, dim=-1)  # (bs, n, n)

        dfm_variables = {
            "pt_vals_X": pt_vals_X,
            "pt_vals_E": pt_vals_E,
            "pt_vals_at_Xt": pt_vals_at_Xt,
            "pt_vals_at_Et": pt_vals_at_Et,
            "dt_p_vals_X": dt_p_vals_X,
            "dt_p_vals_E": dt_p_vals_E,
            "dt_p_vals_at_Xt": dt_p_vals_at_Xt,
            "dt_p_vals_at_Et": dt_p_vals_at_Et,
            "Z_t_X": Z_t_X,
            "Z_t_E": Z_t_E,
        }

        return dfm_variables

    def compute_Rstar(self, dfm_variables):

        # Unpack needed variables
        dt_p_vals_X = dfm_variables["dt_p_vals_X"]
        dt_p_vals_E = dfm_variables["dt_p_vals_E"]
        dt_p_vals_at_Xt = dfm_variables["dt_p_vals_at_Xt"]
        dt_p_vals_at_Et = dfm_variables["dt_p_vals_at_Et"]
        pt_vals_at_Xt = dfm_variables["pt_vals_at_Xt"]
        pt_vals_at_Et = dfm_variables["pt_vals_at_Et"]
        Z_t_X = dfm_variables["Z_t_X"]
        Z_t_E = dfm_variables["Z_t_E"]

        # Numerator of R_t^*
        inner_X = dt_p_vals_X - dt_p_vals_at_Xt[:, :, None]
        inner_E = dt_p_vals_E - dt_p_vals_at_Et[:, :, :, None]
        Rstar_t_numer_X = F.relu(inner_X)  # (bs, n, dx)
        Rstar_t_numer_E = F.relu(inner_E)  # (bs, n, n, de)

        # Denominator
        Rstar_t_denom_X = Z_t_X * pt_vals_at_Xt  # (bs, n)
        Rstar_t_denom_E = Z_t_E * pt_vals_at_Et  # (bs, n, n)

        # Final R^\star
        Rstar_t_X = Rstar_t_numer_X / Rstar_t_denom_X[:, :, None]  # (bs, n, dx)
        Rstar_t_E = Rstar_t_numer_E / Rstar_t_denom_E[:, :, :, None]  # (B, n, n, de)

        return Rstar_t_X, Rstar_t_E

    def compute_RDB(
        self,
        X_t_label,
        E_t_label,
        X_1_pred,
        E_1_pred,
        X_1_sampled,
        E_1_sampled,
        node_mask,
        t,
        dfm_variables,
    ):
        # unpack needed variables
        pt_vals_X = dfm_variables["pt_vals_X"]
        pt_vals_E = dfm_variables["pt_vals_E"]

        # dimensions
        dx = pt_vals_X.shape[-1]
        de = pt_vals_E.shape[-1]

        # build mask for Rdb
        if self.rdb == "general":
            x_mask = torch.ones_like(pt_vals_X)
            e_mask = torch.ones_like(pt_vals_E)

        elif self.rdb == "marginal":
            x_limit = self.limit_dist.X
            e_limit = self.limit_dist.E

            Xt_marginal = x_limit[X_t_label]
            Et_marginal = e_limit[E_t_label]

            x_mask = x_limit.repeat(X_t_label.shape[0], X_t_label.shape[1], 1)
            e_mask = e_limit.repeat(
                E_t_label.shape[0], E_t_label.shape[1], E_t_label.shape[2], 1
            )

            x_mask = x_mask > Xt_marginal
            e_mask = e_mask > Et_marginal

        elif self.rdb == "column":
            # Get column idx to pick
            if self.rdb_crit == "max_marginal":
                x_column_idxs = self.limit_dist.X.argmax(keepdim=True).expand(
                    X_t_label.shape
                )
                e_column_idxs = self.limit_dist.E.argmax(keepdim=True).expand(
                    E_t_label.shape
                )
            elif self.rdb_crit == "x_t":
                x_column_idxs = X_t_label
                e_column_idxs = E_t_label
            elif self.rdb_crit == "abs_state":
                x_column_idxs = torch.ones_like(X_t_label) * (dx - 1)
                e_column_idxs = torch.ones_like(E_t_label) * (de - 1)
            elif self.rdb_crit == "p_x1_g_xt":
                x_column_idxs = X_1_pred.argmax(dim=-1, keepdim=True)
                e_column_idxs = E_1_pred.argmax(dim=-1, keepdim=True)
            elif self.rdb_crit == "x_1":  # as in paper, uniform
                x_column_idxs = X_1_sampled.unsqueeze(-1)
                e_column_idxs = E_1_sampled.unsqueeze(-1)
            elif self.rdb_crit == "p_xt_g_x1":
                x_column_idxs = pt_vals_X.argmax(dim=-1, keepdim=True)
                e_column_idxs = pt_vals_E.argmax(dim=-1, keepdim=True)
            elif self.rdb_crit == "xhat_t":
                sampled_1_hat = sample_discrete_features(
                    pt_vals_X,
                    pt_vals_E,
                    node_mask=node_mask,
                )
                x_column_idxs = sampled_1_hat.X.unsqueeze(-1)
                e_column_idxs = sampled_1_hat.E.unsqueeze(-1)
            else:
                raise NotImplementedError

            # create mask based on columns picked
            x_mask = F.one_hot(x_column_idxs.squeeze(-1), num_classes=dx)
            x_mask[(x_column_idxs == X_t_label).squeeze(-1)] = 1.0
            e_mask = F.one_hot(e_column_idxs.squeeze(-1), num_classes=de)
            e_mask[(e_column_idxs == E_t_label).squeeze(-1)] = 1.0

        elif self.rdb == "entry":
            if self.rdb_crit == "abs_state":
                # select last index
                x_masked_idx = torch.tensor(
                    dx - 1  # delete -1 for the last index
                ).to(X_t_label.device)  # leaving this for now, can change later if we want to explore it a bit more
                e_masked_idx = torch.tensor(de - 1).to(E_t_label.device)

                x1_idxs = X_1_sampled.unsqueeze(-1)  # (bs, n, 1)
                e1_idxs = E_1_sampled.unsqueeze(-1)  # (bs, n, n, 1)
            if self.rdb_crit == "first":  # here in all datasets it's the argmax
                # select last index
                x_masked_idx = torch.tensor(0).to(X_t_label.device)  # leaving this for now, can change later if we want to explore it a bit more
                e_masked_idx = torch.tensor(0).to(E_t_label.device)

                x1_idxs = X_1_sampled.unsqueeze(-1)  # (bs, n, 1)
                e1_idxs = E_1_sampled.unsqueeze(-1)  # (bs, n, n, 1)
            else:
                raise NotImplementedError

            # create mask based on columns picked
            # bs, n, _ = X_t_label.shape
            # x_mask = torch.zeros((bs, n, dx), device=self.device)  # (bs, n, dx)
            x_mask = torch.zeros_like(pt_vals_X)  # (bs, n, dx)
            xt_in_x1 = (X_t_label == x1_idxs).squeeze(-1)  # (bs, n, 1)
            x_mask[xt_in_x1] = F.one_hot(x_masked_idx, num_classes=dx).float()
            xt_in_masked = (X_t_label == x_masked_idx).squeeze(-1)
            x_mask[xt_in_masked] = F.one_hot(
                x1_idxs.squeeze(-1), num_classes=dx
            ).float()[xt_in_masked]

            # e_mask = torch.zeros((bs, n, n, de), device=self.device)  # (bs, n, dx)
            e_mask = torch.zeros_like(pt_vals_E)
            et_in_e1 = (E_t_label == e1_idxs).squeeze(-1)
            e_mask[et_in_e1] = F.one_hot(e_masked_idx, num_classes=de).float()
            et_in_masked = (E_t_label == e_masked_idx).squeeze(-1)
            e_mask[et_in_masked] = F.one_hot(
                e1_idxs.squeeze(-1), num_classes=de
            ).float()[et_in_masked]

        else:
            raise NotImplementedError(f"Not implemented rdb type: {self.rdb}")

        # stochastic rate matrix
        Rdb_t_X = pt_vals_X * x_mask * self.eta
        Rdb_t_E = pt_vals_E * e_mask * self.eta

        return Rdb_t_X, Rdb_t_E

    def compute_R_tg(
        self,
        X_1_sampled,
        E_1_sampled,
        X_t_label,
        E_t_label,
        dfm_variables,
    ):
        """Target guidance rate matrix"""

        # Unpack needed variables
        pt_vals_at_Xt = dfm_variables["pt_vals_at_Xt"]
        pt_vals_at_Et = dfm_variables["pt_vals_at_Et"]
        Z_t_X = dfm_variables["Z_t_X"]
        Z_t_E = dfm_variables["Z_t_E"]

        # Numerator
        X1_onehot = F.one_hot(X_1_sampled, num_classes=self.num_classes_X).float()
        E1_onehot = F.one_hot(E_1_sampled, num_classes=self.num_classes_E).float()
        mask_X = X_1_sampled.unsqueeze(-1) != X_t_label
        mask_E = E_1_sampled.unsqueeze(-1) != E_t_label

        Rtg_t_numer_X = X1_onehot * self.omega * mask_X
        Rtg_t_numer_E = E1_onehot * self.omega * mask_E

        # Denominator
        denom_X = Z_t_X * pt_vals_at_Xt  # (bs, n)
        denom_E = Z_t_E * pt_vals_at_Et  # (bs, n, n)

        # Final R^TG
        Rtg_t_X = Rtg_t_numer_X / denom_X[:, :, None]
        Rtg_t_E = Rtg_t_numer_E / denom_E[:, :, :, None]

        return Rtg_t_X, Rtg_t_E

    def stabilize_rate_matrix(self, R_t_X, R_t_E, dfm_variables):

        # Unpack needed variables
        pt_vals_X = dfm_variables["pt_vals_X"]
        pt_vals_E = dfm_variables["pt_vals_E"]
        pt_vals_at_Xt = dfm_variables["pt_vals_at_Xt"]
        pt_vals_at_Et = dfm_variables["pt_vals_at_Et"]

        # protect to avoid NaN and too large values
        R_t_X = torch.nan_to_num(R_t_X, nan=0.0, posinf=0.0, neginf=0.0)
        R_t_E = torch.nan_to_num(R_t_E, nan=0.0, posinf=0.0, neginf=0.0)
        R_t_X[R_t_X > 1e5] = 0.0
        R_t_E[R_t_E > 1e5] = 0.0

        # Set p(x_t | x_1) = 0 or p(j | x_1) = 0 cases to zero, which need to be applied to Rdb too
        dx = R_t_X.shape[-1]
        de = R_t_E.shape[-1]
        R_t_X[(pt_vals_at_Xt == 0.0)[:, :, None].repeat(1, 1, dx)] = 0.0
        R_t_E[(pt_vals_at_Et == 0.0)[:, :, :, None].repeat(1, 1, 1, de)] = 0.0

        # zero-out certain columns of R, which is implied in the computation of Rdb
        # if the probability of a place is 0, then we should not consider it in the R computation
        R_t_X[pt_vals_X == 0.0] = 0.0
        R_t_E[pt_vals_E == 0.0] = 0.0

        return R_t_X, R_t_E
import torch.nn.functional as F


def p_xt_g_x1(X1, E1, t, limit_dist):
    # x1 (B, D)
    # t float
    # returns (B, D, S) for varying x_t value
    device = X1.device
    limit_dist.X = limit_dist.X.to(device)
    limit_dist.E = limit_dist.E.to(device)

    t_time = t.squeeze(-1)[:, None, None]
    X1_onehot = F.one_hot(X1, num_classes=len(limit_dist.X)).float()
    E1_onehot = F.one_hot(E1, num_classes=len(limit_dist.E)).float()

    Xt = t_time * X1_onehot + (1 - t_time) * limit_dist.X[None, None, :]
    Et = (
        t_time[:, None] * E1_onehot
        + (1 - t_time[:, None]) * limit_dist.E[None, None, None, :]
    )

    assert ((Xt.sum(-1) - 1).abs() < 1e-4).all() and (
        (Et.sum(-1) - 1).abs() < 1e-4
    ).all()

    return Xt.clamp(min=0.0, max=1.0), Et.clamp(min=0.0, max=1.0)


def dt_p_xt_g_x1(X1, E1, limit_dist):
    # x1 (B, D)
    # returns (B, D, S) for varying x_t value
    device = X1.device
    limit_dist.X = limit_dist.X.to(device)
    limit_dist.E = limit_dist.E.to(device)

    X1_onehot = F.one_hot(X1, num_classes=len(limit_dist.X)).float()
    E1_onehot = F.one_hot(E1, num_classes=len(limit_dist.E)).float()

    dX = X1_onehot - limit_dist.X[None, None, :]
    dE = E1_onehot - limit_dist.E[None, None, None, :]

    assert (dX.sum(-1).abs() < 1e-4).all() and (dE.sum(-1).abs() < 1e-4).all()

    return dX, dE


def assert_correctly_masked(variable, node_mask):
    assert (
        variable * (1 - node_mask.long())
    ).abs().max().item() < 1e-4, "Variables not masked properly."


def sample_discrete_feature_noise(limit_dist, node_mask):
    """Sample from the limit distribution of the diffusion process"""
    bs, n_max = node_mask.shape
    x_limit = limit_dist.X[None, None, :].expand(bs, n_max, -1)
    e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)
    y_limit = limit_dist.y[None, :].expand(bs, -1)
    U_X = (
        x_limit.flatten(end_dim=-2).multinomial(1, replacement=True).reshape(bs, n_max)
    )
    U_E = (
        e_limit.flatten(end_dim=-2)
        .multinomial(1, replacement=True)
        .reshape(bs, n_max, n_max)
    )
    U_y = torch.empty((bs, 0))

    long_mask = node_mask.long()
    U_X = U_X.type_as(long_mask)
    U_E = U_E.type_as(long_mask)
    U_y = U_y.type_as(long_mask)

    U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).float()
    U_E = F.one_hot(U_E, num_classes=e_limit.shape[-1]).float()

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = torch.zeros_like(U_E)
    indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1

    U_E = U_E * upper_triangular_mask
    U_E = U_E + torch.transpose(U_E, 1, 2)

    assert (U_E == torch.transpose(U_E, 1, 2)).all()

    return PlaceHolder(X=U_X, E=U_E, y=U_y).mask(node_mask)


def sample_discrete_features(probX, probE, node_mask, mask=False):
    """Sample features from multinomial distribution with given probabilities (probX, probE, proby)
    :param probX: bs, n, dx_out        node features
    :param probE: bs, n, n, de_out     edge features
    :param proby: bs, dy_out           global features.
    """
    bs, n, _ = probX.shape
    # Noise X
    # The masked rows should define probability distributions as well
    probX[~node_mask] = 1 / probX.shape[-1]

    # Flatten the probability tensor to sample with multinomial
    probX = probX.reshape(bs * n, -1)  # (bs * n, dx_out)

    # Sample X
    X_t = probX.multinomial(1, replacement=True)  # (bs * n, 1)
    # X_t = Categorical(probs=probX).sample()  # (bs * n, 1)
    X_t = X_t.reshape(bs, n)  # (bs, n)

    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]

    probE = probE.reshape(bs * n * n, -1)  # (bs * n * n, de_out)

    # Sample E
    E_t = probE.multinomial(1, replacement=True).reshape(bs, n, n)  # (bs, n, n)
    # E_t = Categorical(probs=probE).sample().reshape(bs, n, n)  # (bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = E_t + torch.transpose(E_t, 1, 2)

    if mask:
        X_t = X_t * node_mask
        E_t = E_t * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)

    return PlaceHolder(X=X_t, E=E_t, y=torch.zeros(bs, 0).type_as(X_t))


