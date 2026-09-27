"""
MeshVisualizer — matplotlib-based plots of Neural Mesh activations.

Requires a populated :class:`ActivationRecorder`.

Usage::

    from neural_mesh.visualization import MeshVisualizer

    viz = MeshVisualizer(recorder)
    viz.plot_resolution(dim=0)
    viz.plot_complexity(dim=1, show_pre=True)
    viz.plot_output()
    viz.plot_membership_functions(dim=0)
    viz.plot_all()
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError:
    raise ImportError("matplotlib is required for visualization.  "
                      "Install it with:  pip install matplotlib")

from neural_mesh.recorder import ActivationRecorder


class MeshVisualizer:
    """Visualize recorded activations from a Neural Mesh model.

    Parameters
    ----------
    recorder : ActivationRecorder
        A recorder that has been populated during training.
    figsize : tuple[float, float]
        Default figure size for plots.
    cmap : str
        Default colormap name for heatmaps.
    """

    def __init__(
        self,
        recorder: ActivationRecorder,
        figsize: tuple = (10, 4),
        cmap: str = "viridis",
    ):
        self.rec = recorder
        self.model = recorder.model
        self.figsize = figsize
        self.cmap = cmap

    # ------------------------------------------------------------------
    # Membership functions (model parameters, not activations)
    # ------------------------------------------------------------------

    def plot_membership_functions(
        self,
        dim: int,
        n_points: int = 300,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> plt.Axes:
        """Plot the learned triangle membership functions for a dimension.

        Parameters
        ----------
        dim : int
            Input dimension index.
        n_points : int
            Number of evaluation points.
        ax : matplotlib Axes, optional
        show : bool
            Call ``plt.show()`` if True.
        """
        layer = self.model.memberships[dim]

        with torch.no_grad():
            if layer.convex:
                centres = layer._sorted_centres().cpu().numpy()
                left, right = layer._convex_left_right(layer._sorted_centres())
                left = left.cpu().numpy()
                right = right.cpu().numpy()
            else:
                centres = layer.centres.cpu().numpy()
                left = layer.left.cpu().numpy()
                right = layer.right.cpu().numpy()

        lo = centres.min() - left[0] * 1.2
        hi = centres.max() + right[-1] * 1.2
        xs = np.linspace(lo, hi, n_points)
        x_t = torch.tensor(xs, dtype=torch.float32)

        with torch.no_grad():
            mu = layer(x_t).cpu().numpy()  # (n_points, G)

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        colors = cm.get_cmap(self.cmap, layer.num_sets)
        for g in range(layer.num_sets):
            ax.plot(xs, mu[:, g], color=colors(g), label=f"g={g}")
            ax.axvline(centres[g], color=colors(g), ls=":", alpha=0.4)

        # Show sum of memberships
        ax.plot(xs, mu.sum(axis=1), "k--", alpha=0.5, label="Σμ")

        ax.set_title(f"Membership functions — Dimension {dim}")
        ax.set_xlabel("Input value")
        ax.set_ylabel("Membership degree")
        ax.legend(fontsize="small", ncol=min(layer.num_sets + 1, 6))
        ax.set_ylim(-0.05, 1.15)

        if show:
            plt.tight_layout()
            plt.show()
        return ax

    # ------------------------------------------------------------------
    # Resolution (membership) activations
    # ------------------------------------------------------------------

    def plot_resolution(
        self,
        dim: int,
        epoch: int = -1,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> plt.Axes:
        """Heatmap of resolution (membership) activations for a dimension.

        Rows = samples, columns = fuzzy sets.
        """
        data = self.rec.get("resolution", epoch)[dim].numpy()
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(data, aspect="auto", cmap=self.cmap, vmin=0, vmax=1)
        ax.set_title(f"Resolution activations — Dim {dim}  (epoch {self._resolve_epoch(epoch)})")
        ax.set_xlabel("Fuzzy set index (g)")
        ax.set_ylabel("Sample")
        plt.colorbar(im, ax=ax)
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    # ------------------------------------------------------------------
    # Complexity activations
    # ------------------------------------------------------------------

    def plot_complexity(
        self,
        dim: int,
        epoch: int = -1,
        show_pre: bool = True,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ):
        """Heatmap(s) of complexity pre- and/or post-activations.

        Parameters
        ----------
        dim : int
        epoch : int
        show_pre : bool
            If True and both ``complexity_pre`` and ``complexity_post``
            are recorded, show them side by side.
        ax : Axes, optional
            If given, everything is drawn on this single axis (post is
            preferred; pre is shown only if post is not recorded).
        """
        ep = self._resolve_epoch(epoch)
        has_pre = "complexity_pre" in self.rec.layers
        has_post = "complexity_post" in self.rec.layers

        # Decide which panels to draw
        panels = []  # list of (label, data_array)
        if has_pre and (show_pre or not has_post):
            panels.append(("PRE", self.rec.get("complexity_pre", epoch)[dim].numpy()))
        if has_post:
            panels.append(("POST", self.rec.get("complexity_post", epoch)[dim].numpy()))

        if not panels:
            raise RuntimeError("Neither complexity_pre nor complexity_post "
                               "was recorded.")

        # If the caller supplied a single ax, use it for one panel only
        if ax is not None:
            label, data = panels[-1]  # prefer post
            im = ax.imshow(data, aspect="auto", cmap=self.cmap)
            ax.set_title(f"Complexity {label} — Dim {dim}  (epoch {ep})")
            ax.set_xlabel("Complexity cell (c)")
            ax.set_ylabel("Sample")
            plt.colorbar(im, ax=ax)
            if show:
                plt.tight_layout()
                plt.show()
            return [ax]

        # Otherwise create our own figure with one subplot per panel
        fig, axes = plt.subplots(
            1, len(panels),
            figsize=(self.figsize[0] * len(panels) / 1.5, self.figsize[1]),
        )
        if len(panels) == 1:
            axes = [axes]
        else:
            axes = list(axes)

        for i, (label, data) in enumerate(panels):
            im = axes[i].imshow(data, aspect="auto", cmap=self.cmap)
            axes[i].set_title(f"Complexity {label} — Dim {dim}  (epoch {ep})")
            axes[i].set_xlabel("Complexity cell (c)")
            axes[i].set_ylabel("Sample")
            plt.colorbar(im, ax=axes[i])

        if show:
            plt.tight_layout()
            plt.show()
        return axes

    # ------------------------------------------------------------------
    # Output layer
    # ------------------------------------------------------------------

    def plot_output(
        self,
        epoch: int = -1,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> plt.Axes:
        """Bar/heatmap of output-layer activations."""
        data = self.rec.get("output", epoch).numpy()
        ep = self._resolve_epoch(epoch)

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        if data.shape[1] <= 5:
            # Few outputs → grouped bar chart
            n_samples, n_out = data.shape
            x = np.arange(n_samples)
            width = 0.8 / n_out
            for k in range(n_out):
                ax.bar(x + k * width, data[:, k], width, label=f"out {k}")
            ax.set_xlabel("Sample")
            ax.set_ylabel("Activation")
            ax.legend(fontsize="small")
        else:
            im = ax.imshow(data, aspect="auto", cmap=self.cmap)
            ax.set_xlabel("Output neuron")
            ax.set_ylabel("Sample")
            plt.colorbar(im, ax=ax)

        ax.set_title(f"Output activations  (epoch {ep})")
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    # ------------------------------------------------------------------
    # Firing strengths
    # ------------------------------------------------------------------

    def plot_firing(
        self,
        epoch: int = -1,
        normalized: bool = True,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> plt.Axes:
        """Heatmap of rule firing strengths.

        Parameters
        ----------
        normalized : bool
            Use normalised (True) or raw (False) firing strengths.
        """
        key = "firing_norm" if normalized else "firing_raw"
        data = self.rec.get(key, epoch).numpy()
        ep = self._resolve_epoch(epoch)

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(data, aspect="auto", cmap=self.cmap)
        ax.set_title(f"{'Normalised' if normalized else 'Raw'} firing strengths  (epoch {ep})")
        ax.set_xlabel("Rule index")
        ax.set_ylabel("Sample")
        plt.colorbar(im, ax=ax)
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    # ------------------------------------------------------------------
    # Weight visualization
    # ------------------------------------------------------------------

    def plot_complexity_weights(
        self,
        dim: int,
        resolution_cell: Optional[int] = None,
        show_effective: bool = True,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ):
        """Visualize resolution-to-complexity weights for a dimension.

        Parameters
        ----------
        dim : int
            Input dimension index.
        resolution_cell : int or None
            If given, show a bar chart of the weights from this single
            resolution cell to all complexity cells.  If ``None``, show
            the full (C, G) weight matrix as a heatmap.
        show_effective : bool
            If ``True`` (default), show the effective weights (after
            resolution_weight_norm).  If ``False``, show the raw
            learnable parameters.
        """
        layer = self.model.complexities[dim]
        with torch.no_grad():
            if show_effective:
                W = layer.effective_weights.cpu().numpy()  # (C, G)
                label = "Effective weights"
            else:
                W = layer.weights.cpu().numpy()            # (C, G)
                label = "Raw weights"

        C, G = W.shape
        norm_mode = layer.resolution_weight_norm

        if resolution_cell is not None:
            # Bar chart: weights from one resolution cell to all complexity cells
            g = resolution_cell
            if g < 0 or g >= G:
                raise IndexError(f"resolution_cell={g} out of range [0, {G})")
            vals = W[:, g]

            if ax is None:
                fig, ax = plt.subplots(figsize=self.figsize)

            colors_map = cm.get_cmap(self.cmap, C)
            bars = ax.bar(range(C), vals, color=[colors_map(c) for c in range(C)])
            ax.set_xticks(range(C))
            ax.set_xticklabels([f"c={c}" for c in range(C)])
            ax.set_xlabel("Complexity cell")
            ax.set_ylabel("Weight value")
            ax.set_title(f"{label} — Dim {dim}, resolution cell g={g}"
                         f"  (norm={norm_mode})")

            # Annotate bar values
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

            if show_effective and norm_mode != "none":
                ax.set_ylim(bottom=min(0, vals.min() - 0.05),
                            top=max(vals.max() * 1.15, 1.05))
                ax.axhline(1.0 / C, color="gray", ls="--", alpha=0.4,
                           label=f"uniform = {1.0/C:.3f}")
                ax.legend(fontsize="small")

            if show:
                plt.tight_layout()
                plt.show()
            return ax

        else:
            # Heatmap of full weight matrix
            if ax is None:
                fig, ax = plt.subplots(figsize=self.figsize)

            im = ax.imshow(W, aspect="auto", cmap=self.cmap)
            ax.set_title(f"{label} — Dim {dim}  (norm={norm_mode})")
            ax.set_xlabel("Resolution cell (g)")
            ax.set_ylabel("Complexity cell (c)")
            ax.set_xticks(range(G))
            ax.set_yticks(range(C))
            plt.colorbar(im, ax=ax)

            # Annotate each cell with its value
            for c_idx in range(C):
                for g_idx in range(G):
                    val = W[c_idx, g_idx]
                    color = "white" if val < (W.max() + W.min()) / 2 else "black"
                    ax.text(g_idx, c_idx, f"{val:.2f}", ha="center",
                            va="center", fontsize=7, color=color)

            if show:
                plt.tight_layout()
                plt.show()
            return ax

    def plot_mesh_weights(
        self,
        weight: str = "s",
        output_idx: int = 0,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> plt.Axes:
        """Visualize the mesh layer's s or b weights for one output neuron.

        Parameters
        ----------
        weight : {"s", "b"}
            Which weight tensor to plot.
        output_idx : int
            Which output neuron (row) to show.
        """
        mesh = self.model.mesh
        with torch.no_grad():
            if weight == "s":
                data = mesh.s[output_idx].cpu().numpy()
                title = f"Rule weights (s) — output {output_idx}"
            elif weight == "b":
                data = mesh.b[output_idx].cpu().numpy()
                title = f"Consequent weights (b) — output {output_idx}"
            else:
                raise ValueError(f"weight must be 's' or 'b', got {weight!r}")

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        ax.bar(range(len(data)), data, color=cm.get_cmap(self.cmap)(
            np.linspace(0.2, 0.8, len(data))))
        ax.set_xlabel("Rule index")
        ax.set_ylabel("Weight value")
        ax.set_title(title)

        if show:
            plt.tight_layout()
            plt.show()
        return ax

    # ------------------------------------------------------------------
    # Training history
    # ------------------------------------------------------------------

    def plot_activation_over_time(
        self,
        layer: str,
        dim: int = 0,
        sample: int = 0,
        cell: Optional[int] = None,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> plt.Axes:
        """Line plot of a specific activation value across recorded epochs.

        Parameters
        ----------
        layer : str
            One of the recorded layer names.
        dim : int
            Dimension index (for per-dimension layers).
        sample : int
            Sample index within the recorded samples.
        cell : int or None
            Specific cell/set index to plot.  If None, plot all cells.
        """
        epochs = self.rec.epochs
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        # Collect data across epochs
        vals = []
        for ep in epochs:
            v = self.rec.history[ep][layer]
            if isinstance(v, list):
                v = v[dim]
            vals.append(v[sample].numpy())

        vals = np.stack(vals)  # (num_epochs, num_cells)

        if cell is not None:
            ax.plot(epochs, vals[:, cell], label=f"cell {cell}")
        else:
            for c in range(vals.shape[1]):
                ax.plot(epochs, vals[:, c], label=f"cell {c}", alpha=0.7)

        ax.set_title(f"{layer} over training — dim {dim}, sample {sample}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Activation")
        ax.legend(fontsize="small", ncol=min(vals.shape[1], 6))
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    # ------------------------------------------------------------------
    # Composite plot
    # ------------------------------------------------------------------

    def plot_all(
        self,
        epoch: int = -1,
        dims: Optional[List[int]] = None,
        show: bool = True,
    ):
        """Generate a multi-panel summary of all recorded layers.

        Parameters
        ----------
        epoch : int
        dims : list[int] or None
            Which dimensions to plot.  Default: all.
        """
        if dims is None:
            dims = list(range(self.model.input_dim))

        recorded = self.rec.layers
        n_dim_rows = len(dims)

        # Count how many row-types we need
        row_types = []
        if "resolution" in recorded:
            row_types.append("resolution")
        if "complexity_pre" in recorded or "complexity_post" in recorded:
            row_types.append("complexity")
        if "firing_norm" in recorded or "firing_raw" in recorded:
            row_types.append("firing")
        if "output" in recorded:
            row_types.append("output")

        # Dimensions get their own rows; firing + output are single rows
        dim_rows = [r for r in row_types if r in ("resolution", "complexity")]
        global_rows = [r for r in row_types if r in ("firing", "output")]

        total_rows = len(dim_rows) * n_dim_rows + len(global_rows)
        fig, axes = plt.subplots(
            total_rows, 1,
            figsize=(self.figsize[0], 3.2 * total_rows),
        )
        if total_rows == 1:
            axes = [axes]

        row = 0
        for d in dims:
            if "resolution" in dim_rows:
                self.plot_resolution(d, epoch, ax=axes[row], show=False)
                row += 1
            if "complexity" in dim_rows:
                # Plot post-activation only in the composite view
                key = "complexity_post" if "complexity_post" in recorded else "complexity_pre"
                data = self.rec.get(key, epoch)[d].numpy()
                im = axes[row].imshow(data, aspect="auto", cmap=self.cmap)
                label = "POST" if key == "complexity_post" else "PRE"
                axes[row].set_title(
                    f"Complexity {label} — Dim {d}  "
                    f"(epoch {self._resolve_epoch(epoch)})"
                )
                axes[row].set_xlabel("Complexity cell (c)")
                axes[row].set_ylabel("Sample")
                plt.colorbar(im, ax=axes[row])
                row += 1

        if "firing" in global_rows:
            key = "firing_norm" if "firing_norm" in recorded else "firing_raw"
            self.plot_firing(epoch, normalized=(key == "firing_norm"),
                             ax=axes[row], show=False)
            row += 1
        if "output" in global_rows:
            self.plot_output(epoch, ax=axes[row], show=False)
            row += 1

        plt.tight_layout()
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_epoch(self, epoch: int) -> int:
        return self.rec._resolve_epoch(epoch)
