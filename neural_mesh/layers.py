"""
Core layers for the Neural Mesh architecture.

TriangleMembership  — per-dimension fuzzy antecedent layer
ComplexityLayer     — per-dimension weighted aggregation of memberships
MeshLayer           — tensor-product combination across dimensions
"""

from __future__ import annotations

import itertools
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Activation helper
# ---------------------------------------------------------------------------

_VALID_ACTIVATIONS = ("softplus", "relu", "elu", "tanh", "none")


def _build_activation(name: str) -> nn.Module:
    """Return an activation ``nn.Module`` by name.

    Valid names: ``"softplus"``, ``"relu"``, ``"elu"``, ``"tanh"``,
    ``"none"`` (mapped to ``nn.Identity``).
    """
    _map = {
        "softplus": nn.Softplus(),
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "tanh": nn.Tanh(),
        "none": nn.Identity(),
    }
    if name not in _map:
        raise ValueError(
            f"Unknown activation {name!r}, choose from {list(_map)}"
        )
    return _map[name]


# ---------------------------------------------------------------------------
# Triangle fuzzy membership function
# ---------------------------------------------------------------------------

class TriangleMembership(nn.Module):
    """Compute triangle membership degrees for a single input dimension.

    Supports two modes controlled by the ``convex`` flag:

    **convex=True** (default) — Ruspini partition
        Only the centre positions are learnable.  Centres are kept in
        strict ascending order via a *base + cumulative softplus gaps*
        parameterisation.  Each triangle's left support reaches the
        previous centre and its right support reaches the next centre,
        so the memberships at any point sum to exactly 1::

            ╱╲    ╱╲    ╱╲
           ╱  ╲  ╱  ╲  ╱  ╲       Σ_g μ_g(x) = 1   ∀ x
          ╱    ╲╱    ╲╱    ╲
         c0    c1    c2    c3

        The first and last triangles mirror the gap to their one
        neighbour for the open side.

    **convex=False** — free asymmetric triangles
        Centres, left distances, and right distances are all
        independently learnable (stored in raw space, softplus for
        positivity).  Memberships do *not* necessarily sum to 1.

    Parameters
    ----------
    num_sets : int
        Number of fuzzy sets (*G* for this dimension).
    init_range : tuple[float, float], optional
        Interval for initial centre placement.  Defaults to (0, 1).
    convex : bool
        If ``True``, enforce a Ruspini partition (default ``True``).
    bounds : tuple[float, float] or None
        **Convex mode only.**  If given, the centres are confined to
        ``[lo, hi]`` for the whole of training: the first centre is pinned
        to ``lo``, the last to ``hi``, and the interior centres move freely
        in between (parameterised as softmax-weighted gaps, so they stay
        strictly ordered).  Pass the training-data range of this input so
        that the Ruspini partition always covers the data and no triangle
        can drift outside it.  ``None`` (default) keeps the unbounded
        behaviour, in which centres can leave the data range.
    """

    def __init__(self, num_sets: int, init_range: Tuple[float, float] = (0.0, 1.0),
                 convex: bool = True, min_gap: float = 1e-3,
                 bounds: Optional[Tuple[float, float]] = None):
        super().__init__()
        self.num_sets = num_sets
        self.convex = convex
        self.bounded = bounds is not None
        if self.bounded:
            if not convex:
                raise ValueError("bounds are only supported with convex=True")
            lo, hi = float(bounds[0]), float(bounds[1])
            if not hi > lo:
                raise ValueError(f"bounds must satisfy lo < hi, got {bounds}")
            if num_sets < 2:
                raise ValueError("bounds require at least 2 fuzzy sets")
            # each gap gets at least `floor_frac` of the range; clipped so the
            # floors can never exceed the range itself
            self.floor_frac = min(min_gap / (hi - lo), 0.5 / (num_sets - 1))
            self.register_buffer("lo", torch.tensor(lo))
            self.register_buffer("hi", torch.tensor(hi))
        # Floor on the distance between neighbouring centres (convex mode).
        # Without it, gaps can collapse toward the old 1e-8 floor during long
        # training runs; membership gradients scale as 1/gap^2, which then
        # overflows float32 and poisons Adam's moments with inf -> NaN.
        # 1e-3 is far below any meaningful resolution for z-scored inputs.
        self.min_gap = min_gap

        if num_sets > 1:
            span = (init_range[1] - init_range[0]) / (num_sets - 1)
        else:
            span = init_range[1] - init_range[0] if init_range[1] != init_range[0] else 1.0

        if self.bounded:
            # Centres = lo + (hi-lo) * cumsum(gap fractions); the fractions are
            # a floored softmax, so they are positive and sum to exactly 1:
            # first centre = lo, last centre = hi, interior strictly ordered.
            # Zero logits -> evenly spaced centres (same start as unbounded).
            self.raw_gaps = nn.Parameter(torch.zeros(num_sets - 1))
        elif convex:
            # Parameterise as base + cumulative positive gaps so centres
            # are guaranteed to be strictly ascending.
            self.base = nn.Parameter(torch.tensor(float(init_range[0])))
            if num_sets > 1:
                raw_gap = math.log(math.exp(span) - 1.0 + 1e-8)
                self.raw_gaps = nn.Parameter(torch.full((num_sets - 1,), raw_gap))
            else:
                self.register_parameter("raw_gaps", None)
        else:
            # Free centres + independent left/right distances
            centres = torch.linspace(init_range[0], init_range[1], num_sets)
            self.centres = nn.Parameter(centres)

            raw_init = math.log(math.exp(span) - 1.0 + 1e-8)
            self.raw_left = nn.Parameter(torch.full((num_sets,), raw_init))
            self.raw_right = nn.Parameter(torch.full((num_sets,), raw_init))

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def _sorted_centres(self) -> torch.Tensor:
        """Return sorted centres (convex mode)."""
        if self.bounded:
            n_gaps = self.num_sets - 1
            frac = (self.floor_frac
                    + (1.0 - n_gaps * self.floor_frac)
                    * torch.softmax(self.raw_gaps, dim=0))       # (G-1,), sums to 1
            span = self.hi - self.lo
            return self.lo + span * torch.cat([
                torch.zeros(1, device=frac.device, dtype=frac.dtype),
                torch.cumsum(frac, dim=0),
            ])
        if self.num_sets == 1:
            return self.base.unsqueeze(0)
        gaps = F.softplus(self.raw_gaps) + self.min_gap  # (G-1,), strictly > 0
        return self.base + torch.cat([
            torch.zeros(1, device=self.base.device),
            torch.cumsum(gaps, dim=0),
        ])

    def _convex_left_right(self, centres: torch.Tensor):
        """Derive left/right distances from sorted centres."""
        if self.num_sets == 1:
            left = torch.ones(1, device=centres.device)
            right = torch.ones(1, device=centres.device)
            return left, right

        diffs = centres[1:] - centres[:-1]  # (G-1,), all > 0

        # left_i  = c_i - c_{i-1}  (first triangle mirrors its right gap)
        left = torch.cat([diffs[:1], diffs])
        # right_i = c_{i+1} - c_i  (last triangle mirrors its left gap)
        right = torch.cat([diffs, diffs[-1:]])
        return left, right

    @property
    def left(self) -> torch.Tensor:
        """Positive left distances (free mode only)."""
        return F.softplus(self.raw_left) + 1e-8

    @property
    def right(self) -> torch.Tensor:
        """Positive right distances (free mode only)."""
        return F.softplus(self.raw_right) + 1e-8

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch,)
            Scalar inputs for this dimension.

        Returns
        -------
        Tensor of shape (batch, G)
            Membership degrees in [0, 1].
        """
        x = x.unsqueeze(-1)  # (batch, 1)

        if self.convex:
            c = self._sorted_centres()                   # (G,)
            left, right = self._convex_left_right(c)     # (G,), (G,)
        else:
            c = self.centres                              # (G,)
            left, right = self.left, self.right           # (G,), (G,)

        c = c.unsqueeze(0)          # (1, G)
        diff = x - c                # (batch, G)

        mu_left = 1.0 + diff / left.unsqueeze(0)
        mu_right = 1.0 - diff / right.unsqueeze(0)
        mu = torch.where(diff < 0, mu_left, mu_right)
        return torch.clamp(mu, min=0.0)


# ---------------------------------------------------------------------------
# Complexity layer (per dimension)
# ---------------------------------------------------------------------------

class ComplexityLayer(nn.Module):
    """Map G membership degrees to C complexity activations via learned weights.

    For complexity cell *c*:

        a_c = activation( Σ_g  |w|_{c,g} · μ_g  +  bias_c )

    An optional **resolution weight norm** (cnorm) can be applied to the
    raw weights before the matrix multiply.  For each resolution cell *g*,
    the weights to all *C* complexity cells are normalised so that they
    are non-negative and sum to 1.  Three modes are available:

    ``"none"`` (default)
        No normalisation — raw weights are used as-is.

    ``"softmax"``
        Softmax across the complexity dimension::

            |w|_{c,g} = exp(w_{c,g}) / Σ_{c'} exp(w_{c',g})

    ``"thresholded_sum_to_one"``
        Negative weights are clamped to 0, then the remaining positive
        weights are divided by their sum::

            |w|_{c,g} = max(0, w_{c,g}) / Σ_{c'} max(0, w_{c',g})

        Unlike softmax, this produces exact zeros for negative raw
        weights, giving a sparser normalisation.

    A nonlinear activation (default: softplus) is applied after the affine
    transform.  **softplus** is recommended for tensor-product architectures
    because it is strictly positive — a product of softplus outputs is
    always > 0, preventing dead gradients in higher-dimensional meshes.

    The bias term gives the network a learnable offset so that even when
    membership values are small, the pre-activation does not collapse to
    zero.

    Parameters
    ----------
    num_sets : int
        Number of fuzzy sets (*G*) feeding into this layer.
    num_complexity : int
        Number of complexity cells (*C*) for this dimension.
    activation : str
        Activation function after the affine map.
        One of ``"softplus"``, ``"relu"``, ``"elu"``, ``"tanh"``,
        ``"none"``  (default ``"softplus"``).
    resolution_weight_norm : str
        Resolution-wise weight normalisation mode.
        One of ``"none"``, ``"softmax"``, ``"thresholded_sum_to_one"``
        (default ``"none"``).
    """

    _WEIGHT_NORMS = frozenset({"none", "softmax", "thresholded_sum_to_one"})

    def __init__(self, num_sets: int, num_complexity: int,
                 activation: str = "softplus",
                 resolution_weight_norm: str = "none"):
        super().__init__()
        if resolution_weight_norm not in self._WEIGHT_NORMS:
            raise ValueError(
                f"Unknown resolution_weight_norm {resolution_weight_norm!r}, "
                f"choose from {sorted(self._WEIGHT_NORMS)}"
            )
        self.resolution_weight_norm = resolution_weight_norm
        self.weights = nn.Parameter(torch.empty(num_complexity, num_sets))
        self.bias = nn.Parameter(torch.zeros(num_complexity))
        nn.init.xavier_uniform_(self.weights)

        self.activation = _build_activation(activation)

    @property
    def effective_weights(self) -> torch.Tensor:
        """Return the (C, G) weight matrix after optional normalisation.

        Normalisation is applied along dim=0 (the complexity dimension)
        independently for each resolution cell (column).
        """
        if self.resolution_weight_norm == "softmax":
            return F.softmax(self.weights, dim=0)
        if self.resolution_weight_norm == "thresholded_sum_to_one":
            clamped = torch.clamp(self.weights, min=0.0)
            # 1e-8 (not 1e-12): near-dead columns otherwise produce ~1/col_sum^2
            # gradient spikes that can overflow float32 in long runs.
            col_sums = clamped.sum(dim=0, keepdim=True) + 1e-8
            return clamped / col_sums
        return self.weights

    def pre_activation(self, memberships: torch.Tensor) -> torch.Tensor:
        """Affine transform before the activation function.

        Returns
        -------
        Tensor of shape (batch, C)
        """
        return memberships @ self.effective_weights.t() + self.bias

    def forward(self, memberships: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        memberships : Tensor of shape (batch, G)

        Returns
        -------
        Tensor of shape (batch, C)
        """
        return self.activation(self.pre_activation(memberships))


# ---------------------------------------------------------------------------
# Mesh (tensor-product) layer
# ---------------------------------------------------------------------------

class MeshLayer(nn.Module):
    """Tensor-product meshing of per-dimension complexity activations.

    Each *rule* is indexed by a tuple (c₁, c₂, …, cₙ) drawn from the
    Cartesian product of per-dimension complexity cells.

    Rule firing strength (raw):

        r_{c₁…cₙ} = Π_i  a_{i, cᵢ}

    When ``normalize=True`` (the default) the raw firing strengths are
    divided by their sum so they form a proper partition of unity — this
    mirrors the standard fuzzy-inference normalisation step and keeps the
    magnitudes bounded regardless of the number of input dimensions:

        r̂_{rule} = r_{rule} / Σ_{rules} r_{rule}

    Output (per output neuron *k*):

        y_k = Σ_{rules}  s_{k, rule} · b_{k, rule} · r̂_{rule}

    Parameters
    ----------
    complexity_sizes : list[int]
        C_i for each input dimension.
    num_outputs : int
        Dimensionality of the network output.
    normalize : bool
        If ``True`` (default), normalise firing strengths to sum to 1.
    on_mesh_activation : str
        Activation function applied to each output neuron after the
        weighted aggregation.  One of ``"softplus"``, ``"relu"``,
        ``"elu"``, ``"tanh"``, ``"none"`` (default ``"softplus"``).
    """

    def __init__(self, complexity_sizes: List[int], num_outputs: int,
                 normalize: bool = True,
                 on_mesh_activation: str = "softplus"):
        super().__init__()
        self.complexity_sizes = complexity_sizes
        self.num_outputs = num_outputs
        self.normalize = normalize
        self.on_mesh_activation_name = on_mesh_activation
        self.on_mesh_activation = _build_activation(on_mesh_activation)

        # Total number of rules = product of all C_i
        self.num_rules = 1
        for c in complexity_sizes:
            self.num_rules *= c

        # Pre-compute rule index tuples (each is a tuple of per-dim indices)
        self.register_buffer(
            "rule_indices",
            torch.tensor(
                list(itertools.product(*(range(c) for c in complexity_sizes))),
                dtype=torch.long,
            ),
        )  # shape: (num_rules, n_dims)

        # Learnable rule weights: s (rule weight) and b (consequent weight)
        # Both are multiplicative, so b is initialised to 1.0 (not 0.0).
        self.s = nn.Parameter(torch.empty(num_outputs, self.num_rules))
        self.b = nn.Parameter(torch.ones(num_outputs, self.num_rules))
        nn.init.xavier_uniform_(self.s)

    def forward(self, activations: List[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        activations : list of Tensors, each of shape (batch, C_i)

        Returns
        -------
        Tensor of shape (batch, num_outputs)
        """
        batch_size = activations[0].shape[0]
        n_dims = len(activations)

        # Gather the activation for each rule's index in each dimension
        # and multiply across dimensions to get rule firing strengths.
        firing = torch.ones(batch_size, self.num_rules, device=activations[0].device)
        for dim_idx in range(n_dims):
            idx = self.rule_indices[:, dim_idx]
            dim_act = activations[dim_idx][:, idx]
            firing = firing * dim_act

        # Normalise firing strengths to sum to 1 (partition of unity)
        if self.normalize:
            firing = firing / (firing.sum(dim=-1, keepdim=True) + 1e-12)

        # Output: y_k = Σ_rule  s_{k,rule} · b_{k,rule} · firing_rule
        # firing: (batch, num_rules) → (batch, 1, num_rules)
        # s, b: (num_outputs, num_rules)
        # result: (batch, num_outputs)
        rule_outputs = self.s.unsqueeze(0) * self.b.unsqueeze(0) * firing.unsqueeze(1)
        return self.on_mesh_activation(rule_outputs).sum(dim=-1)
