"""
ActivationRecorder — captures intermediate activations during training.

Usage::

    recorder = ActivationRecorder(
        model,
        layers=("resolution", "complexity_post", "output"),
        record_every=10,      # snapshot every 10th epoch
        max_history=50,       # keep the last 50 snapshots
        sample_indices=[0, 1, 2, 3, 4],  # only store these samples
    )

    # Inside your training loop (or pass recorder to ``train``):
    for epoch in range(epochs):
        ...
        recorder.maybe_record(epoch, X_train)

    # After training:
    viz = MeshVisualizer(recorder)
    viz.plot_resolution(dim=0, epoch=-1)
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import torch

from neural_mesh.model import NeuralMeshModel

# All layer keys that forward_with_intermediates can return
ALL_LAYERS = frozenset({
    "resolution", "complexity_pre", "complexity_post",
    "firing_raw", "firing_norm", "output",
})


class ActivationRecorder:
    """Record activation snapshots from a :class:`NeuralMeshModel`.

    Parameters
    ----------
    model : NeuralMeshModel
        The model to observe.
    layers : sequence of str
        Which intermediate tensors to store.  Valid names:
        ``"resolution"``, ``"complexity_pre"``, ``"complexity_post"``,
        ``"firing_raw"``, ``"firing_norm"``, ``"output"``.
        Default: all of them.
    record_every : int
        Record a snapshot every *N* calls to :meth:`maybe_record`.
        Default 1 (every call).
    max_history : int or None
        Maximum number of snapshots to keep (FIFO).  ``None`` = unlimited.
    sample_indices : list[int] or None
        If given, only store activations for these sample indices
        (rows of *X*).  ``None`` = store all samples.
    """

    def __init__(
        self,
        model: NeuralMeshModel,
        layers: Sequence[str] = tuple(ALL_LAYERS),
        record_every: int = 1,
        max_history: Optional[int] = None,
        sample_indices: Optional[List[int]] = None,
    ):
        unknown = set(layers) - ALL_LAYERS
        if unknown:
            raise ValueError(f"Unknown layer names: {unknown}.  "
                             f"Choose from {sorted(ALL_LAYERS)}")
        self.model = model
        self.layers: Set[str] = set(layers)
        self.record_every = max(record_every, 1)
        self.max_history = max_history
        self.sample_indices = sample_indices

        # epoch → {layer_name: tensor or list[tensor]}
        self.history: OrderedDict[int, Dict] = OrderedDict()

        self._call_count = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    @torch.no_grad()
    def maybe_record(self, epoch: int, X: torch.Tensor) -> bool:
        """Record a snapshot if the current call count is divisible by
        ``record_every``.  Returns ``True`` if a snapshot was stored.
        """
        self._call_count += 1
        if self._call_count % self.record_every != 0:
            return False

        self.model.eval()
        intermediates = self.model.forward_with_intermediates(X)
        self.model.train()

        snapshot: Dict = {}
        for key in self.layers:
            val = intermediates[key]
            if isinstance(val, list):
                # Per-dimension lists (resolution, complexity_*)
                val = [self._maybe_slice(t).detach().cpu() for t in val]
            else:
                val = self._maybe_slice(val).detach().cpu()
            snapshot[key] = val

        self.history[epoch] = snapshot

        # FIFO eviction
        if self.max_history is not None:
            while len(self.history) > self.max_history:
                self.history.popitem(last=False)

        return True

    def _maybe_slice(self, t: torch.Tensor) -> torch.Tensor:
        """Optionally select only certain sample rows."""
        if self.sample_indices is not None:
            return t[self.sample_indices]
        return t

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------

    @property
    def epochs(self) -> List[int]:
        """List of recorded epoch numbers."""
        return list(self.history.keys())

    def _resolve_epoch(self, epoch: int) -> int:
        """Map negative indices to actual epoch numbers.

        -1 → last recorded epoch, -2 → second to last, etc.
        Positive values are returned as-is.
        """
        if epoch < 0:
            ep_list = self.epochs
            if -epoch > len(ep_list):
                raise IndexError(
                    f"Requested epoch index {epoch} but only "
                    f"{len(ep_list)} snapshots are stored."
                )
            return ep_list[epoch]
        return epoch

    def get(self, layer: str, epoch: int = -1):
        """Retrieve a recorded snapshot.

        Parameters
        ----------
        layer : str
            Layer name (e.g. ``"resolution"``).
        epoch : int
            Epoch number, or a negative index (``-1`` = last,
            ``-2`` = second to last, etc.).
        """
        epoch = self._resolve_epoch(epoch)
        return self.history[epoch][layer]

    def clear(self):
        """Discard all stored snapshots."""
        self.history.clear()
        self._call_count = 0
