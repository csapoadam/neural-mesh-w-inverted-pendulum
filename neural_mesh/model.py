"""
NeuralMeshModel — the top-level model combining all layers.

Supports two task modes:
    * "regression"      — raw linear output, trained with MSE (RMSE reported)
    * "classification"  — softmax output, trained with cross-entropy loss
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_mesh.layers import (
    ComplexityLayer, MeshLayer, TriangleMembership, _build_activation,
)


class NeuralMeshModel(nn.Module):
    """Full Neural Mesh network.

    Parameters
    ----------
    input_dim : int
        Number of input dimensions (*n*).
    resolution : int | list[int]
        *G* — number of fuzzy sets per dimension.  A single int applies to
        every dimension; a list gives per-dimension values.
    complexity : int | list[int]
        *C* — number of complexity cells per dimension (same broadcasting
        rules as *resolution*).
    num_outputs : int
        Number of output neurons (1 for scalar regression, *K* for
        *K*-class classification).
    task : {"regression", "classification"}
        Determines the loss function and final activation.
    init_ranges : list[tuple[float, float]] | None
        Per-dimension (lo, hi) for initial triangle centre placement.
        Defaults to (0, 1) for every dimension.
    complexity_activation : str
        Activation function applied after the complexity layer.
        One of ``"softplus"``, ``"relu"``, ``"elu"``, ``"tanh"``,
        ``"none"`` (default ``"softplus"``).
    on_mesh_activation : str
        Activation function applied to each output neuron of the mesh
        layer (after the weighted rule aggregation).  Same options as
        *complexity_activation* (default ``"softplus"``).
    post_mesh_activation : str
        Activation function applied to the final model output, after
        the mesh layer and before the task-specific head (e.g.
        ``log_softmax`` for classification).  Same options as
        *complexity_activation* (default ``"softplus"``).
    normalize_firing : bool
        If ``True`` (default), normalise rule firing strengths to sum to 1.
        Recommended for higher-dimensional inputs to keep magnitudes bounded.
    resolution_convex_triangles : bool
        If ``True`` (default), the triangle membership functions form a
        Ruspini (convex) partition where memberships sum to 1 at every
        point.  Only the centre positions are learnable; supports are
        derived from neighbouring centres.  If ``False``, centres, left
        distances, and right distances are all independently learnable.
    resolution_weight_norm : str
        Resolution-wise weight normalisation mode for the complexity
        layer weights.  One of ``"none"`` (default), ``"softmax"``,
        or ``"thresholded_sum_to_one"``.
    resolution_bounds : list[tuple[float, float]] | "init" | None
        Confine the triangle centres of each dimension to a fixed interval
        during training (convex triangles only).  The first and last
        centres are pinned to the interval ends, so the Ruspini partition
        covers it completely and no fuzzy set can drift out of the data.
        Pass per-dimension (lo, hi) -- typically the min/max of the
        TRAINING inputs -- or ``"init"`` to reuse *init_ranges*.  ``None``
        (default) keeps the original, unbounded behaviour.
    """

    def __init__(
        self,
        input_dim: int,
        resolution: Union[int, List[int]],
        complexity: Union[int, List[int]],
        num_outputs: int = 1,
        task: str = "regression",
        init_ranges: Optional[List[Tuple[float, float]]] = None,
        complexity_activation: str = "softplus",
        on_mesh_activation: str = "softplus",
        post_mesh_activation: str = "softplus",
        normalize_firing: bool = True,
        resolution_convex_triangles: bool = True,
        resolution_weight_norm: str = "none",
        resolution_bounds: Optional[Union[str, List[Tuple[float, float]]]] = None,
    ):
        super().__init__()
        assert task in ("regression", "classification")
        self.input_dim = input_dim
        self.num_outputs = num_outputs
        self.task = task

        # Broadcast scalars to per-dimension lists
        if isinstance(resolution, int):
            resolution = [resolution] * input_dim
        if isinstance(complexity, int):
            complexity = [complexity] * input_dim
        assert len(resolution) == input_dim
        assert len(complexity) == input_dim
        self.resolution = resolution
        self.complexity = complexity

        if init_ranges is None:
            init_ranges = [(0.0, 1.0)] * input_dim
        assert len(init_ranges) == input_dim

        # Optional per-dimension bounds on the triangle centres
        if isinstance(resolution_bounds, str):
            if resolution_bounds != "init":
                raise ValueError('resolution_bounds must be None, "init" or a list of (lo, hi)')
            resolution_bounds = list(init_ranges)
        if resolution_bounds is None:
            bounds_list = [None] * input_dim
        else:
            assert len(resolution_bounds) == input_dim
            bounds_list = [tuple(map(float, b)) for b in resolution_bounds]
        self.resolution_bounds = resolution_bounds

        # Per-dimension membership + complexity layers
        self.resolution_convex_triangles = resolution_convex_triangles
        self.memberships = nn.ModuleList(
            [TriangleMembership(g, r, convex=resolution_convex_triangles, bounds=b)
             for g, r, b in zip(resolution, init_ranges, bounds_list)]
        )
        self.complexity_activation = complexity_activation
        self.resolution_weight_norm = resolution_weight_norm
        self.complexities = nn.ModuleList(
            [ComplexityLayer(g, c, activation=complexity_activation,
                             resolution_weight_norm=resolution_weight_norm)
             for g, c in zip(resolution, complexity)]
        )

        # Tensor-product mesh layer
        self.normalize_firing = normalize_firing
        self.on_mesh_activation = on_mesh_activation
        self.mesh = MeshLayer(complexity, num_outputs, normalize=normalize_firing,
                              on_mesh_activation=on_mesh_activation)

        # Post-mesh activation (applied after the mesh layer output)
        self.post_mesh_activation = post_mesh_activation
        self._post_mesh_act = _build_activation(post_mesh_activation)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, input_dim)

        Returns
        -------
        Tensor of shape (batch, num_outputs)
            Raw logits (regression) or log-probabilities (classification).
        """
        activations = []
        for dim_idx in range(self.input_dim):
            mu = self.memberships[dim_idx](x[:, dim_idx])       # (batch, G_i)
            act = self.complexities[dim_idx](mu)                 # (batch, C_i)
            activations.append(act)

        out = self.mesh(activations)           # (batch, num_outputs)
        out = self._post_mesh_act(out)         # post-mesh activation

        if self.task == "classification":
            out = F.log_softmax(out, dim=-1)

        return out

    def forward_with_intermediates(self, x: torch.Tensor) -> dict:
        """Run a forward pass and return all intermediate activations.

        Returns
        -------
        dict with keys:
            ``"resolution"``       — list of (batch, G_i) membership tensors
            ``"complexity_pre"``   — list of (batch, C_i) pre-activation tensors
            ``"complexity_post"``  — list of (batch, C_i) post-activation tensors
            ``"firing_raw"``       — (batch, num_rules) raw firing strengths
            ``"firing_norm"``      — (batch, num_rules) normalised firing (if applicable)
            ``"output"``           — (batch, num_outputs) final output
        """
        resolution = []
        complexity_pre = []
        complexity_post = []

        activations = []
        for dim_idx in range(self.input_dim):
            mu = self.memberships[dim_idx](x[:, dim_idx])
            resolution.append(mu)

            pre = self.complexities[dim_idx].pre_activation(mu)
            complexity_pre.append(pre)

            post = self.complexities[dim_idx].activation(pre)
            complexity_post.append(post)
            activations.append(post)

        # Reproduce the mesh layer's internals to capture firing strengths
        mesh = self.mesh
        batch_size = x.shape[0]
        firing = torch.ones(batch_size, mesh.num_rules, device=x.device)
        for dim_idx in range(self.input_dim):
            idx = mesh.rule_indices[:, dim_idx]
            firing = firing * activations[dim_idx][:, idx]

        firing_raw = firing
        if mesh.normalize:
            firing_norm = firing / (firing.sum(dim=-1, keepdim=True) + 1e-12)
        else:
            firing_norm = firing_raw

        rule_outputs = mesh.s.unsqueeze(0) * mesh.b.unsqueeze(0) * firing_norm.unsqueeze(1)
        out = mesh.on_mesh_activation(rule_outputs).sum(dim=-1)
        out = self._post_mesh_act(out)         # post-mesh activation
        if self.task == "classification":
            out = F.log_softmax(out, dim=-1)

        return {
            "resolution": resolution,
            "complexity_pre": complexity_pre,
            "complexity_post": complexity_post,
            "firing_raw": firing_raw,
            "firing_norm": firing_norm,
            "output": out,
        }

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions (class indices for classification)."""
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
            if self.task == "classification":
                return out.argmax(dim=-1)
            return out

    def summary(self) -> str:
        """Human-readable summary of the architecture."""
        total = sum(p.numel() for p in self.parameters())
        lines = [
            f"NeuralMeshModel  task={self.task}  input_dim={self.input_dim}  "
            f"outputs={self.num_outputs}",
            f"  Resolution (G): {self.resolution}"
            f"  convex_triangles={self.resolution_convex_triangles}"
            f"  weight_norm={self.resolution_weight_norm}"
            f"  bounded={'no' if self.resolution_bounds is None else 'yes'}",
            f"  Complexity (C): {self.complexity}"
            f"  activation={self.complexity_activation}",
            f"  On-mesh activation:  {self.on_mesh_activation}",
            f"  Post-mesh activation: {self.post_mesh_activation}",
            f"  Total rules:    {self.mesh.num_rules}",
            f"  Learnable params: {total}",
        ]
        return "\n".join(lines)
