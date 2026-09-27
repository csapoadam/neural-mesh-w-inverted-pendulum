"""
Rule-set construction, optimisation and visualisation for the Neural Mesh.

This module migrates the rule-set machinery of the *tpmpy* toolbox into the
Neural Mesh architecture.  The pipeline has three stages:

1. **Extraction** (``extract_rule_set``) — rules arise inside the mesh as the
   Cartesian product of per-dimension *complexity cells*.  Each complexity
   cell aggregates the triangle fuzzy sets of its dimension through the
   complexity layer's (effective) weights, so it defines an aggregated fuzzy
   antecedent over that dimension.  For every combination of antecedent
   cells (= one mesh rule) the consequent is read out from the output
   expression by evaluating the trained model at the rule's *prototype
   point* (the tensor product of the antecedent centres).  Rule weights are
   set in direct proportion to how strongly the training data activates each
   rule (mean normalised firing strength).

2. **Optimisation** (``gradient_rulify``) — the extracted ``RuleSet`` is
   fine-tuned as a differentiable TSK-0 fuzzy system with Gaussian
   membership functions.  Sparsity/overlap/cardinality penalties plus
   periodic pruning and ``merge_rules`` produce a compact rule set with
   fewer rules.

3. **Visualisation** (``GradientRulifyResult.plot_rules``) — the optimised
   Gaussian antecedents are plotted per dimension, exactly as in tpmpy.

Ported (and trimmed) from tpmpy's ``tensorlib/ruleset.py``,
``tensorlib/inference.py`` (``infer_from_ruleset``) and
``tensorlib/gradient_search.py`` (part 2).
"""

from __future__ import annotations

import sys
from itertools import product

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "RuleSet",
    "extract_rule_set",
    "infer_from_ruleset",
    "gradient_rulify",
    "GradientRulifyResult",
]


# ---------------------------------------------------------------------------
# Small progress-bar helper (replaces tpmpy's fp_types.progress_bar)
# ---------------------------------------------------------------------------

def _progress_bar(current: int, total: int, message: str = "", width: int = 30):
    frac = current / max(total, 1)
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r[{bar}] {current}/{total}  {message}")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


# ---------------------------------------------------------------------------
# Consequent helpers: consequents may be SCALARS (classic TSK-0, one rule set
# per output) or VECTORS of length K (MIMO / "co-dependent" rule set: ONE
# shared antecedent structure whose rules each carry K consequent values,
# one per output dimension).
# ---------------------------------------------------------------------------

def _cons_scalar(c):
    """Scalar key for sorting/summarising a (possibly vector) consequent.

    For vector consequents the FIRST output component is used."""
    return float(np.atleast_1d(np.asarray(c, dtype=float))[0])


def _format_consequent(c):
    a = np.atleast_1d(np.asarray(c, dtype=float))
    if a.size == 1:
        return f"{a[0]:.2f}"
    return "[" + ", ".join(f"{v:.2f}" for v in a) + "]"


# ---------------------------------------------------------------------------
# RuleSet (ported from tpmpy tensorlib/ruleset.py)
# ---------------------------------------------------------------------------

class RuleSet:
    """
    A set of weighted fuzzy rules.

    Each rule has a weight in direct proportion to the amount of data that
    represents it.  Each antecedent variable in each rule is an ordinal
    index 0, 1, …, and each ordinal index has a (min, max) range of values
    associated with it along the corresponding input dimension.

    Parameters
    ----------
    rules : list of (ant, cons) tuples
        *ant* is a list of per-dimension antecedent indices, *cons* the
        (scalar) consequent value.
    antecedent_ranges : list of D lists
        One sub-list per dimension; each sub-list contains (min, max)
        tuples defining the value range of each antecedent index.
    weights : list of floats
        One representativeness weight per rule.
    counts : ndarray or None
        Optional D-dimensional array of data-point counts per antecedent
        cell (hard assignment).  Carried through for reference only.
    """

    def __init__(self, rules, antecedent_ranges, weights, counts=None):
        self.rules = rules
        self.antecedent_ranges = antecedent_ranges
        self.weights = weights
        self.counts = counts

    def __str__(self):
        outs = []
        for inx, (inputs, output) in enumerate(self.rules):
            current_out = " AND ".join(str(x) for x in inputs)
            current_out += " => " + _format_consequent(output)
            outs.append(
                f"{current_out} (representativeness weight: "
                f"{self.weights[inx]:.4f})"
            )
        outs.append(f"\nTotal number of rules: {len(self.rules)}")
        outs.append(f"\nTotal representation: {sum(self.weights)}")
        return "\n".join(outs)

    def canonicalize(self):
        """Renumber antecedent indices so that, within every dimension,
        indices follow spatial order: antecedent *k+1* lies to the right
        of antecedent *k* on the number line (ranges sorted by midpoint).

        Rules, weights and counts are remapped consistently, so inference
        is unchanged — only the labelling becomes spatially meaningful.
        Returns ``self`` (mutated in place) for chaining.
        """
        orders = []
        for d, ranges_d in enumerate(self.antecedent_ranges):
            mids = [(lo + hi) / 2.0 for (lo, hi) in ranges_d]
            order = sorted(range(len(mids)), key=lambda k: mids[k])
            orders.append(order)                       # new_idx -> old_idx
            self.antecedent_ranges[d] = [ranges_d[o] for o in order]
        old_to_new = [
            {o: n for n, o in enumerate(order)} for order in orders
        ]
        self.rules = [
            ([old_to_new[d][a] for d, a in enumerate(ant)], cons)
            for (ant, cons) in self.rules
        ]
        if self.counts is not None:
            c = np.asarray(self.counts)
            if (c.ndim == len(self.antecedent_ranges)
                    and all(c.shape[d] == len(self.antecedent_ranges[d])
                            for d in range(c.ndim))):
                for d, order in enumerate(orders):
                    c = np.take(c, order, axis=d)
                self.counts = c
        return self

    def sorted(self, min_weight=-0.01, method="consequents"):
        """Return a copy sorted by 'consequents' or 'weights' (descending),
        keeping only rules with weight above *min_weight*."""
        common = [
            [self.rules[inx], self.weights[inx]]
            for inx in range(len(self.rules))
            if self.weights[inx] > min_weight
        ]
        if method == "consequents":
            common_sorted = sorted(common, key=lambda e: _cons_scalar(e[0][1]),
                                   reverse=True)
        else:
            common_sorted = sorted(common, key=lambda e: e[1], reverse=True)

        return RuleSet(
            [e[0] for e in common_sorted],
            self.antecedent_ranges,
            [e[1] for e in common_sorted],
            self.counts,
        )

    def to_csv(self, min_weight, filename):
        """Write rules with weight >= *min_weight* as CSV rows of numeric
        antecedent indices followed by the consequent."""
        rows = []
        for inx, (inputs, output) in enumerate(self.rules):
            if self.weights[inx] >= min_weight:
                nums = ([str(i) for i in inputs]
                        + [f"{v:.2f}" for v in np.atleast_1d(output)])
                rows.append(",".join(nums))
        with open(filename, "w", newline="") as f:
            f.write("\n".join(rows))


# ---------------------------------------------------------------------------
# Rule extraction from a trained NeuralMeshModel
# ---------------------------------------------------------------------------

def _membership_geometry(membership):
    """Return (centres, left, right) numpy arrays for a TriangleMembership,
    handling both convex (Ruspini) and free modes."""
    with torch.no_grad():
        if membership.convex:
            centres = membership._sorted_centres()
            left, right = membership._convex_left_right(centres)
        else:
            centres = membership.centres
            left, right = membership.left, membership.right
        return (
            centres.detach().cpu().numpy().astype(np.float64),
            left.detach().cpu().numpy().astype(np.float64),
            right.detach().cpu().numpy().astype(np.float64),
        )


def extract_rule_set(
    model,
    X=None,
    output_index=0,
    min_weight=0.0,
    min_halfwidth=1e-3,
):
    """
    Construct a ``RuleSet`` from the weights of a trained
    ``NeuralMeshModel``.

    Where the rules arise
    ---------------------
    * **Antecedents** — for input dimension *d*, complexity cell *c* is an
      aggregated fuzzy set: its (effective) weight row over the *G_d*
      triangle memberships defines *where* along the dimension the cell is
      active.  The cell's centre is the weight-averaged triangle centre and
      its half-width combines the weighted spread of the contributing
      centres with their triangle supports.  Each cell becomes one
      antecedent range ``(centre - hw, centre + hw)``.

    * **Consequents** — mesh rule *r* = (c₁,…,cₙ) is one combination of
      antecedent cells.  Its consequent is read out from the output
      expression by evaluating the model at the rule's prototype point
      (the vector of antecedent centres).  For classification models the
      consequent is the predicted class index (as float).

    * **Weights** — if training inputs *X* are given, the weight of rule
      *r* is the mean normalised firing strength of *r* over the data
      (weights sum to 1).  Without *X*, weights are uniform.

    Parameters
    ----------
    model : NeuralMeshModel
        A trained model.
    X : Tensor of shape (P, D) or None
        Training inputs used to compute representativeness weights and
        per-cell counts.
    output_index : int or None
        Which output neuron to read consequents from (regression with
        multiple outputs).  Pass ``None`` to extract VECTOR consequents
        over ALL outputs at once — this produces a single MIMO
        ("co-dependent") rule set with one shared antecedent structure,
        suitable for joint optimisation with ``gradient_rulify`` on data
        of shape (P, D + K).  Ignored for classification (argmax is used).
    min_weight : float
        Rules whose weight falls below this value are dropped (antecedent
        ranges are kept intact so indices stay valid).
    min_halfwidth : float
        Lower bound on antecedent half-widths.

    Returns
    -------
    RuleSet
        Rules sorted by weight (descending).
    """
    model.eval()
    device = next(model.parameters()).device
    D = model.input_dim

    # --- antecedent centres and ranges per dimension -----------------------
    centres_per_dim = []       # list of (C_d,) arrays
    ranges_per_dim = []        # list of lists of (lo, hi)
    for d in range(D):
        tri_c, tri_l, tri_r = _membership_geometry(model.memberships[d])
        W = (
            model.complexities[d]
            .effective_weights.detach().cpu().numpy().astype(np.float64)
        )                                                   # (C_d, G_d)
        Wabs = np.abs(W)
        row_sums = Wabs.sum(axis=1, keepdims=True) + 1e-12
        P_w = Wabs / row_sums                               # (C_d, G_d), rows sum to 1

        cell_centres = P_w @ tri_c                          # (C_d,)
        # spread of contributing triangle centres around the cell centre
        spread = np.sqrt(
            (P_w * (tri_c[None, :] - cell_centres[:, None]) ** 2).sum(axis=1)
        )
        # plus the weighted average triangle half-support
        half_support = P_w @ ((tri_l + tri_r) / 2.0)
        halfwidth = np.maximum(spread + half_support / 2.0, min_halfwidth)

        centres_per_dim.append(cell_centres)
        ranges_per_dim.append(
            [
                (float(c - h), float(c + h))
                for c, h in zip(cell_centres, halfwidth)
            ]
        )

    # --- rule index tuples come straight from the mesh layer ---------------
    rule_indices = model.mesh.rule_indices.detach().cpu().numpy()  # (R, D)
    R = rule_indices.shape[0]

    # --- consequents: evaluate model at each rule's prototype point --------
    prototypes = np.stack(
        [centres_per_dim[d][rule_indices[:, d]] for d in range(D)], axis=1
    )                                                        # (R, D)
    with torch.no_grad():
        proto_t = torch.tensor(prototypes, dtype=torch.float32, device=device)
        out = model(proto_t)                                 # (R, num_outputs)
        if model.task == "classification":
            consequents = out.argmax(dim=-1).double().cpu().numpy()
        elif output_index is None:
            # Vector consequents over all outputs (MIMO / co-dependent)
            consequents = out.double().cpu().numpy()          # (R, K)
        else:
            consequents = out[:, output_index].double().cpu().numpy()

    # --- weights: mean normalised firing over the data ---------------------
    counts = None
    if X is not None:
        with torch.no_grad():
            X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
            inter = model.forward_with_intermediates(X_t)
            firing = inter["firing_norm"].double().cpu().numpy()  # (P, R)
        weights = firing.mean(axis=0)
        weights = weights / (weights.sum() + 1e-12)

        # hard-assignment counts per antecedent cell combination
        hard = firing.argmax(axis=1)                          # (P,)
        counts_flat = np.bincount(hard, minlength=R).astype(float)
        counts = counts_flat.reshape(tuple(model.mesh.complexity_sizes))
    else:
        weights = np.full(R, 1.0 / R)

    # --- assemble, filter, sort ---------------------------------------------
    def _cons_of(r):
        if consequents.ndim == 2:
            return consequents[r].astype(float).copy()
        return float(consequents[r])

    triples = [
        (rule_indices[r].tolist(), _cons_of(r), float(weights[r]))
        for r in range(R)
        if weights[r] >= min_weight
    ]
    triples.sort(key=lambda t: t[2], reverse=True)

    total_w = sum(t[2] for t in triples) or 1.0

    return RuleSet(
        rules=[(ant, cons) for (ant, cons, _) in triples],
        antecedent_ranges=ranges_per_dim,
        weights=[w / total_w for (_, _, w) in triples],
        counts=counts,
    ).canonicalize()


# ---------------------------------------------------------------------------
# Crisp inference from a RuleSet (ported from tpmpy tensorlib/inference.py)
# ---------------------------------------------------------------------------

def _precompute_rule_dict(rule_set, cutoff_weight=None):
    mids_per_dim = [
        np.array([(lo + hi) / 2.0 for (lo, hi) in dim], dtype=float)
        for dim in rule_set.antecedent_ranges
    ]
    sizes = np.array([len(m) for m in mids_per_dim], dtype=int)

    rules = {}
    for rinx, (antecedent, consequent) in enumerate(rule_set.rules):
        w = rule_set.weights[rinx]
        if cutoff_weight is not None and w < cutoff_weight:
            continue
        key = tuple(np.asarray(antecedent, dtype=int).tolist())
        rules.setdefault(key, []).append(
            (w, np.atleast_1d(np.asarray(consequent, dtype=float)))
        )

    return mids_per_dim, sizes, rules


def _nearest_indices(inputvec, mids_per_dim):
    return np.array(
        [int(np.argmin(np.abs(mids - x)))
         for x, mids in zip(inputvec, mids_per_dim)],
        dtype=int,
    )


def _distance_normalizer(distances):
    denom = sum(distances)
    if denom == 0:
        return [0 for _ in distances]
    ws = [1 - item / denom for item in distances]
    ws = [1 if abs(sum(ws)) < 0.001 else item / sum(ws) for item in ws]
    return ws


def infer_from_ruleset(
    rule_set,
    inputs,
    cutoff_weight=None,
    k_matches=3,
    max_radius=2,
    metric="euclidean",
    postprocessor_fn=lambda x: x,
):
    """
    Infer outputs from a fuzzy rule set over a set of input vectors.

    For each input vector, the nearest antecedent combination(s) are located
    by expanding a neighbourhood shell outward from the closest antecedent
    indices; up to *k_matches* rules are blended via a weighted average of
    their consequents (rule weight × distance factor).

    Parameters
    ----------
    rule_set : RuleSet
    inputs : array-like of shape (P, D)
    cutoff_weight : float or None
        Rules below this weight are ignored.
    k_matches : int
        Maximum number of nearby rules to blend.
    max_radius : int
        Maximum neighbourhood expansion radius.
    metric : {"euclidean", "manhattan", "chebyshev"}
    postprocessor_fn : callable
        Applied to the (P, 1) result array (e.g. ``np.round``).

    Returns
    -------
    ndarray of shape (P, 1) — or (P, K) for MIMO rule sets whose
    consequents are K-vectors.
    """
    inputs = np.asarray(inputs, dtype=float)
    P, D = inputs.shape

    mids_per_dim, sizes, rule_dict = _precompute_rule_dict(
        rule_set, cutoff_weight=cutoff_weight
    )

    if not rule_dict:
        raise ValueError("no rules left after cutoff_weight filtering")
    K_out = next(iter(rule_dict.values()))[0][1].size
    results = np.zeros((P, K_out), dtype=float)

    for p in range(P):
        x = inputs[p]
        base = _nearest_indices(x, mids_per_dim)

        collected = []  # (dist, weight, consequent)

        for r in range(max_radius + 1):
            for offs in product(range(-r, r + 1), repeat=D):
                idx = base + np.array(offs, dtype=int)
                if np.any(idx < 0) or np.any(idx >= sizes):
                    continue

                key = tuple(idx.tolist())
                hits = rule_dict.get(key)
                if not hits:
                    continue

                mids = np.array(
                    [mids_per_dim[d][idx[d]] for d in range(D)], dtype=float
                )
                diff = mids - x
                if metric == "euclidean":
                    dist = float(np.sqrt(np.dot(diff, diff)))
                elif metric == "manhattan":
                    dist = float(np.abs(diff).sum())
                elif metric == "chebyshev":
                    dist = float(np.abs(diff).max())
                else:
                    raise ValueError(
                        "metric must be 'euclidean', 'manhattan', "
                        "or 'chebyshev'"
                    )

                for w, c in hits:
                    collected.append((dist, w, c))

            if len(collected) >= k_matches:
                break

        if not collected:
            raise ValueError(f"could not find consequent for {x} in row {p}")

        collected.sort(key=lambda t: t[0])
        chosen = collected[:k_matches]

        total_w = sum(w for _, w, _ in chosen)
        if total_w == 0:
            raise ValueError(f"total weight is zero after cutoff for row {p}")

        dist_factors = _distance_normalizer([d for d, _, _ in chosen])

        results[p] = sum(
            dist_factors[inx] * (w / total_w) * c
            for inx, (d, w, c) in enumerate(chosen)
        )

    return postprocessor_fn(results)


# ---------------------------------------------------------------------------
# Differentiable concordance index
# ---------------------------------------------------------------------------

def _differentiable_cindex(y_true, y_pred, sigma=0.1):
    """Smooth approximation of the concordance index (sigmoid step)."""
    n = y_true.shape[0]
    idx_i, idx_j = torch.triu_indices(n, n, offset=1)
    true_diff = y_true[idx_i] - y_true[idx_j]
    pred_diff = y_pred[idx_i] - y_pred[idx_j]

    valid = (true_diff.abs() > 1e-8).to(y_true.dtype)
    concordance = torch.sigmoid(true_diff * pred_diff / sigma)

    total_valid = valid.sum() + 1e-10
    return (concordance * valid).sum() / total_valid


def _r2_torch(targets, preds):
    """Differentiable R² — the mean over output columns when 2-D.

    Accepts (P,) or (P, K) tensors; per-column R² is averaged uniformly."""
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)
    preds = preds.reshape(targets.shape)
    ss_res = ((targets - preds) ** 2).sum(dim=0)
    ss_tot = ((targets - targets.mean(dim=0)) ** 2).sum(dim=0)
    return (1.0 - ss_res / (ss_tot + 1e-10)).mean()


# ---------------------------------------------------------------------------
# Result container for gradient_rulify
# ---------------------------------------------------------------------------

class GradientRulifyResult:
    """Snapshot of the best rule-set found so far during gradient_rulify."""

    def __init__(
        self,
        iteration,
        fitness,
        ant_centers_per_dim,   # list of D arrays
        ant_sigmas_per_dim,    # list of D arrays
        consequents,           # (R,) array
        weights,               # (R,) array  (already positive, not log)
        ant_idx,               # (R, D) int array
        initial_rule_set,      # reference to the original RuleSet
        fitness_details=None,
    ):
        self.iteration = iteration
        self.fitness = fitness
        self._ant_centers = ant_centers_per_dim
        self._ant_sigmas = ant_sigmas_per_dim
        self._consequents = consequents
        self._weights = weights
        self._ant_idx = ant_idx
        self._initial_rule_set = initial_rule_set
        self.fitness_details = fitness_details or {}

    def get_fitness(self):
        return self.fitness

    # ---- export to standard RuleSet ----------------------------------------

    def to_rule_set(self, sigma_multiplier=2.0, min_weight_pct=0.0):
        """
        Convert the optimised Gaussian parameters back to a standard
        ``RuleSet`` compatible with ``infer_from_ruleset``.

        Parameters
        ----------
        sigma_multiplier : float
            Each antecedent range is exported as
            ``(center - k*sigma, center + k*sigma)``.  Default 2.0 covers
            ~95 % of the Gaussian.
        min_weight_pct : float
            Rules whose normalised weight is below this fraction are
            pruned from the exported set.
        """
        total_w = float(self._weights.sum())
        rules, rule_weights = [], []
        for r in range(len(self._consequents)):
            w = float(self._weights[r])
            if total_w > 0 and w / total_w < min_weight_pct:
                continue
            ant = self._ant_idx[r].tolist()
            c = self._consequents[r]
            cons = (np.asarray(c, dtype=float).copy()
                    if np.ndim(c) > 0 else float(c))
            rules.append((ant, cons))
            rule_weights.append(w)

        # Compact: keep only antecedents referenced by a surviving rule,
        # so filtered exports do not list orphan ranges.
        new_ant_ranges = []
        remap = []
        for d, (centers, sigmas) in enumerate(
            zip(self._ant_centers, self._ant_sigmas)
        ):
            used = sorted({ant[d] for ant, _ in rules})
            remap.append({o: n for n, o in enumerate(used)})
            new_ant_ranges.append([
                (float(centers[o] - sigma_multiplier * sigmas[o]),
                 float(centers[o] + sigma_multiplier * sigmas[o]))
                for o in used
            ])
        rules = [
            ([remap[d][a] for d, a in enumerate(ant)], cons)
            for (ant, cons) in rules
        ]

        w_sum = sum(rule_weights) or 1.0
        rule_weights = [w / w_sum for w in rule_weights]

        return RuleSet(
            rules=rules,
            antecedent_ranges=new_ant_ranges,
            weights=rule_weights,
            counts=getattr(self._initial_rule_set, "counts", None),
        ).canonicalize()

    # ---- direct inference using the Gaussian model -------------------------

    def infer(self, inputs, postprocessor_fn=None):
        """
        Run inference on *inputs* using the optimised Gaussian membership
        functions — continuous weighted blending, no discretisation.

        Parameters
        ----------
        inputs : array-like, shape (P, D)
        postprocessor_fn : callable or None
            Applied to the (P, 1) predictions.  ``None`` → identity.

        Returns
        -------
        ndarray of shape (P, 1) — or (P, K) for MIMO rule sets whose
        consequents are K-vectors (column order = output order).
        """
        inputs = np.asarray(inputs, dtype=np.float64)
        P, D = inputs.shape

        centers = np.stack([
            self._ant_centers[d][self._ant_idx[:, d]]
            for d in range(D)
        ], axis=1)
        sigmas = np.stack([
            self._ant_sigmas[d][self._ant_idx[:, d]]
            for d in range(D)
        ], axis=1)

        x = inputs[:, np.newaxis, :]                       # (P, 1, D)
        membership = np.exp(-0.5 * ((x - centers) / sigmas) ** 2)
        firing = membership.prod(axis=2)                   # (P, R)

        wf = firing * self._weights[np.newaxis, :]
        denom = wf.sum(axis=1, keepdims=True) + 1e-10
        cons = np.asarray(self._consequents)
        if cons.ndim == 2:                       # (R, K): vector consequents
            predictions = (wf @ cons) / denom    # (P, K)
        else:
            predictions = (wf * cons[np.newaxis, :]).sum(
                axis=1, keepdims=True
            ) / denom

        if postprocessor_fn is not None:
            predictions = postprocessor_fn(predictions)
        return predictions

    # ---- rule merging via clustering ----------------------------------------

    def merge_rules(self, n_target=None, max_consequent_gap=None,
                    sigma_multiplier=2.0, verbose=False):
        """
        Merge similar rules to produce a compact ``RuleSet`` that can be
        re-optimised with ``gradient_rulify``.

        Rules are clustered by their effective centre in input space and
        their consequent value; each cluster is fused into a single rule
        whose Gaussian covers the union of the originals.

        Parameters
        ----------
        n_target : int or None
            Desired number of merged rules (agglomerative clustering).
        max_consequent_gap : float or None
            Maximum consequent difference allowed within one cluster.
            At least one of *n_target* / *max_consequent_gap* required.
        sigma_multiplier : float
            Exported ranges are ``(center ± sigma_multiplier * sigma)``.
        verbose : bool
            Print merge statistics.

        Returns
        -------
        RuleSet
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        if n_target is None and max_consequent_gap is None:
            raise ValueError(
                "Provide at least one of n_target or max_consequent_gap"
            )

        D = len(self._ant_centers)
        R = len(self._consequents)

        eff_centers = np.stack([
            self._ant_centers[d][self._ant_idx[:, d]]
            for d in range(D)
        ], axis=1)                                 # (R, D)
        eff_sigmas = np.stack([
            self._ant_sigmas[d][self._ant_idx[:, d]]
            for d in range(D)
        ], axis=1)                                 # (R, D)

        cons = np.asarray(self._consequents).copy()
        cons2d = cons if cons.ndim == 2 else cons[:, None]     # (R, K)
        ws = self._weights.copy()

        center_ranges = (
            eff_centers.max(axis=0) - eff_centers.min(axis=0) + 1e-10
        )
        cons_range = cons2d.max(axis=0) - cons2d.min(axis=0) + 1e-10  # (K,)

        ## Scalar consequent summary used by the max_consequent_gap logic:
        ## the per-output-normalised mean (equals the consequent itself,
        ## rescaled, in the scalar case).
        cons_scalar = (cons2d / cons_range).mean(axis=1)

        features = np.column_stack([
            eff_centers / center_ranges,
            cons2d / cons_range,
        ])                                         # (R, D+K)

        if R <= 1:
            labels = np.array([0])
        else:
            dist_vec = pdist(features, metric="euclidean")
            Z = linkage(dist_vec, method="ward")

            if n_target is not None:
                n_clust = max(1, min(n_target, R))
                labels = fcluster(Z, t=n_clust, criterion="maxclust") - 1
            else:
                labels = fcluster(Z, t=R, criterion="maxclust") - 1

            if max_consequent_gap is not None:
                ## NOTE: for MIMO rule sets the gap check applies to the
                ## normalised mean consequent (cons_scalar), so express
                ## max_consequent_gap in those units for K > 1.
                new_labels = labels.copy()
                next_label = labels.max() + 1
                for c in range(labels.max() + 1):
                    members = np.where(labels == c)[0]
                    if len(members) <= 1:
                        continue
                    c_cons = cons_scalar[members]
                    if c_cons.max() - c_cons.min() > max_consequent_gap:
                        order = np.argsort(c_cons)
                        sorted_cons = c_cons[order]
                        current_label = new_labels[members[order[0]]]
                        for i in range(1, len(order)):
                            if (sorted_cons[i] - sorted_cons[i - 1]
                                    > max_consequent_gap / 2.0):
                                next_label += 1
                                current_label = next_label
                            new_labels[members[order[i]]] = current_label
                labels = new_labels
                unique_labels = sorted(set(labels))
                remap = {old: new for new, old in enumerate(unique_labels)}
                labels = np.array([remap[l] for l in labels])

        n_clusters = labels.max() + 1

        merged_ant_ranges = [[] for _ in range(D)]
        merged_rules = []
        merged_weights = []

        for c in range(n_clusters):
            members = np.where(labels == c)[0]
            member_ws = ws[members]
            w_total = member_ws.sum()
            if w_total < 1e-15:
                w_total = 1.0

            mc = (member_ws[:, None] * cons2d[members]).sum(axis=0) / w_total
            merged_cons = mc.copy() if cons.ndim == 2 else float(mc[0])

            ant_indices = []
            for d in range(D):
                m_centers = eff_centers[members, d]
                m_sigmas = eff_sigmas[members, d]
                m_ws = member_ws

                new_c = float((m_ws * m_centers).sum() / w_total)
                lo = (m_centers - sigma_multiplier * m_sigmas).min()
                hi = (m_centers + sigma_multiplier * m_sigmas).max()
                new_s = max(float((hi - lo) / (2 * sigma_multiplier)), 0.01)

                ant_range = (
                    float(new_c - sigma_multiplier * new_s),
                    float(new_c + sigma_multiplier * new_s),
                )
                found = False
                for idx, existing in enumerate(merged_ant_ranges[d]):
                    mid_e = (existing[0] + existing[1]) / 2
                    mid_n = (ant_range[0] + ant_range[1]) / 2
                    if abs(mid_e - mid_n) < 0.01 * (abs(mid_e) + 1e-10):
                        ant_indices.append(idx)
                        found = True
                        break
                if not found:
                    ant_indices.append(len(merged_ant_ranges[d]))
                    merged_ant_ranges[d].append(ant_range)

            merged_rules.append((ant_indices, merged_cons))
            merged_weights.append(float(w_total))

        w_sum = sum(merged_weights) or 1.0
        merged_weights = [w / w_sum for w in merged_weights]

        if verbose:
            print(
                f"Merged {R} rules → {n_clusters} rules.  "
                f"Antecedent ranges per dim: "
                f"{[len(r) for r in merged_ant_ranges]}"
            )

        return RuleSet(
            rules=merged_rules,
            antecedent_ranges=merged_ant_ranges,
            weights=merged_weights,
            counts=getattr(self._initial_rule_set, "counts", None),
        ).canonicalize()

    # ---- visualisation -------------------------------------------------------

    def plot_rules(
        self,
        dim_names=None,
        z_means=None,
        z_stds=None,
        min_weight=0.01,
        n_points=200,
        figsize_per_dim=(10, 3.5),
        skip_dims=None,
        save=False,
        filename="rulesplot",
        show=True,
    ):
        """
        Plot the Gaussian antecedent membership functions for every
        dimension, one subplot per dimension — same visual style as tpmpy.

        Within each dimension, duplicate Gaussians (rules sharing the same
        antecedent index) are drawn only once, with all rule labels at the
        peak.  Rules are drawn low-weight first so high-weight rules stay
        visible on top.

        Parameters
        ----------
        dim_names : list of str or None
            Human-readable names per input dimension.
        z_means, z_stds : array-like of length D, or None
            If both given, x-axes are transformed back from z-scores to
            the original scale.
        min_weight : float
            Only show rules with normalised weight >= this value.
        n_points : int
            Points per Gaussian curve.
        figsize_per_dim : (float, float)
            (width, height) of each per-dimension subplot.
        skip_dims : list of int or None
            Dimension indices to skip.
        save : bool
            Save the figure as ``{filename}.png``.
        show : bool
            Call ``plt.show()`` at the end.
        """
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        from collections import defaultdict

        D = len(self._ant_centers)

        skip_dims = [] if skip_dims is None else skip_dims

        if dim_names is None:
            dim_names = [f"Dim {d}" for d in range(D)]

        denorm = (z_means is not None and z_stds is not None)
        if denorm:
            z_means = np.asarray(z_means, dtype=np.float64)
            z_stds = np.asarray(z_stds, dtype=np.float64)

        # --- determine which rules to show ---------------------------------
        w = self._weights.copy()
        w_sum = w.sum() if w.sum() > 0 else 1.0
        w_normed = w / w_sum
        active_mask = w_normed >= min_weight
        active_internal = np.where(active_mask)[0]

        if len(active_internal) == 0:
            print("No rules above min_weight threshold.")
            return

        # --- renumber rules by consequent (descending) ----------------------
        sorted_by_cons = sorted(
            active_internal,
            key=lambda r: -_cons_scalar(self._consequents[r]),
        )
        display_idx = {}
        for display_i, internal_r in enumerate(sorted_by_cons):
            display_idx[internal_r] = display_i

        # low-weight drawn first, high-weight on top
        active_rules = sorted(active_internal, key=lambda r: w_normed[r])

        _DISTINCT_COLORS = [
            "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
            "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
            "#dcbeff", "#9A6324", "#ffe119", "#000075", "#aaffc3",
            "#808000", "#ffd8b1", "#e6beff", "#800000", "#000000",
        ]
        rule_color = {}
        for internal_r in active_internal:
            di = display_idx[internal_r]
            rule_color[internal_r] = _DISTINCT_COLORS[
                di % len(_DISTINCT_COLORS)
            ]

        fig, axes = plt.subplots(
            D - len(skip_dims), 1,
            figsize=(figsize_per_dim[0], figsize_per_dim[1] * D),
            squeeze=False,
        )

        num_skipped_so_far = 0

        for d in range(D):
            if d in skip_dims:
                num_skipped_so_far += 1
                continue

            ax = axes[d - num_skipped_so_far, 0]
            centers_d = np.asarray(self._ant_centers[d], dtype=np.float64)
            sigmas_d = np.asarray(self._ant_sigmas[d], dtype=np.float64)

            active_ks = set(int(self._ant_idx[r, d]) for r in active_rules)
            active_centers = centers_d[list(active_ks)]
            active_sigmas = sigmas_d[list(active_ks)]
            all_lo = (active_centers - 4 * active_sigmas).min()
            all_hi = (active_centers + 4 * active_sigmas).max()
            x_z = np.linspace(all_lo, all_hi, n_points)

            if denorm:
                x_display = x_z * z_stds[d] + z_means[d]
            else:
                x_display = x_z

            ant_groups = defaultdict(list)
            for r in active_rules:
                ant_k = int(self._ant_idx[r, d])
                ant_groups[ant_k].append(r)

            for ant_k, members in ant_groups.items():
                c = centers_d[ant_k]
                s = sigmas_d[ant_k]
                gauss = np.exp(-0.5 * ((x_z - c) / s) ** 2)

                top_r = max(members, key=lambda r: w_normed[r])
                color = rule_color[top_r]
                alpha = max(0.4, min(1.0, w_normed[top_r] * 5))

                ax.plot(x_display, gauss, color=color, alpha=alpha,
                        linewidth=2.0)

                if denorm:
                    c_display = c * z_stds[d] + z_means[d]
                else:
                    c_display = c

                labels = [
                    f"R{display_idx[r]}"
                    for r in sorted(members, key=lambda r: display_idx[r])
                ]
                ax.annotate(
                    ", ".join(labels),
                    xy=(c_display, 1.0),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=7,
                    fontweight="bold",
                    color=color,
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white",
                        edgecolor=color,
                        alpha=0.8,
                        linewidth=0.5,
                    ),
                )

            ax.set_title(dim_names[d], fontsize=12, fontweight="bold")
            ax.set_ylabel("Membership")
            ax.set_ylim(-0.05, 1.25)
            ax.grid(True, alpha=0.3)

            if d == D - 1:
                ax.set_xlabel(
                    "Original scale" if denorm else "Model input scale"
                )

        legend_rules = sorted(active_internal, key=lambda r: display_idx[r])
        handles = []
        for r in legend_rules:
            di = display_idx[r]
            handles.append(mlines.Line2D(
                [], [],
                color=rule_color[r],
                linewidth=2,
                label=(
                    f"R{di}: cons={_format_consequent(self._consequents[r])}, "
                    f"w={w_normed[r]:.3f}"
                ),
            ))

        if len(handles) <= 20:
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=min(4, len(handles)),
                fontsize=7,
                bbox_to_anchor=(0.5, -0.02),
            )

        fig.suptitle(
            f"Rule Antecedent Gaussians ({len(active_rules)} rules shown)",
            fontsize=14, fontweight="bold", y=1.01,
        )
        plt.tight_layout()

        if save:
            plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        return fig

    def __str__(self):
        details = "; ".join(
            f"{k} = {v:.4f}" for k, v in self.fitness_details.items()
        )
        active = int((self._weights > 1e-6).sum())
        return (
            f"Iteration {self.iteration}: fitness = {self.fitness:.6f} "
            f"({details}); {active} active rules"
        )

    def __repr__(self):
        return self.__str__()


# ---------------------------------------------------------------------------
# Differentiable TSK-0 fuzzy inference model
# ---------------------------------------------------------------------------

class _DifferentiableRuleModel(nn.Module):
    """
    Differentiable Takagi-Sugeno-Kang (order-0) fuzzy inference system.

    Each rule *r* has:

    - **Antecedent membership** in dimension *d*: a Gaussian
      ``μ_rd(x) = exp( -(x - c_rd)² / (2 σ_rd²) )`` where *c* and *σ* are
      **shared** across rules referencing the same antecedent index in
      that dimension (matching the ``antecedent_ranges`` structure).

    - **Firing strength**: ``f_r(x) = Π_d μ_rd(x_d)``

    - **Prediction**: ``ŷ = Σ_r (f_r · w_r · q_r) / Σ_r (f_r · w_r)``.
    """

    def __init__(self, data_np, initial_rule_set, consequent_bounds=None):
        super().__init__()

        self.D = len(initial_rule_set.antecedent_ranges)
        self.R = len(initial_rule_set.rules)
        self.consequent_bounds = consequent_bounds  # None or (lo, hi)

        # Number of outputs: scalar consequents -> 1 (classic TSK-0),
        # vector consequents -> K (MIMO / "co-dependent" rule set: ONE
        # shared antecedent structure, per-output consequent values).
        self.K = int(np.atleast_1d(
            np.asarray(initial_rule_set.rules[0][1], dtype=float)).size)
        if data_np.shape[1] != self.D + self.K:
            raise ValueError(
                f"data has {data_np.shape[1]} columns but the rule set "
                f"implies {self.D} input(s) + {self.K} output(s); for a "
                f"MIMO rule set pass ALL K target columns after the inputs"
            )

        self.register_buffer(
            "inputs", torch.tensor(data_np[:, :-self.K], dtype=torch.float64)
        )
        self.register_buffer(
            "targets", torch.tensor(data_np[:, -self.K:], dtype=torch.float64)
        )

        # --- Shared antecedent parameters (per dimension, per range) --------
        self.ant_centers = nn.ParameterList()
        self.ant_log_sigmas = nn.ParameterList()
        for d in range(self.D):
            ranges_d = initial_rule_set.antecedent_ranges[d]
            centers = torch.tensor(
                [(float(lo) + float(hi)) / 2.0 for lo, hi in ranges_d],
                dtype=torch.float64,
            )
            widths = torch.tensor(
                [max((float(hi) - float(lo)) / 3.0, 0.01)
                 for lo, hi in ranges_d],
                dtype=torch.float64,
            )
            self.ant_centers.append(nn.Parameter(centers))
            self.ant_log_sigmas.append(nn.Parameter(torch.log(widths)))

        # --- Per-rule parameters --------------------------------------------
        if self.K == 1:
            raw_consequents = torch.tensor(
                [float(np.atleast_1d(rule[1])[0])
                 for rule in initial_rule_set.rules],
                dtype=torch.float64,
            )                                                  # (R,)
        else:
            raw_consequents = torch.tensor(
                np.stack([np.asarray(rule[1], dtype=float)
                          for rule in initial_rule_set.rules]),
                dtype=torch.float64,
            )                                                  # (R, K)

        if self.consequent_bounds is not None:
            lo, hi = self.consequent_bounds
            eps = 1e-6
            clamped = raw_consequents.clamp(lo + eps, hi - eps)
            normalised = (clamped - lo) / (hi - lo)
            logits = torch.log(normalised / (1.0 - normalised))
            self.consequents = nn.Parameter(logits)
        else:
            self.consequents = nn.Parameter(raw_consequents)

        weights = torch.tensor(
            [float(w) for w in initial_rule_set.weights],
            dtype=torch.float64,
        ).clamp(min=1e-8)
        self.log_weights = nn.Parameter(torch.log(weights))

        ant_idx = torch.tensor(
            [
                [int(initial_rule_set.rules[r][0][d]) for d in range(self.D)]
                for r in range(self.R)
            ],
            dtype=torch.long,
        )
        self.register_buffer("ant_idx", ant_idx)

    # ---- helpers -----------------------------------------------------------

    def actual_consequents(self):
        """Effective consequent values (post-sigmoid if bounded)."""
        if self.consequent_bounds is not None:
            lo, hi = self.consequent_bounds
            return lo + (hi - lo) * torch.sigmoid(self.consequents)
        return self.consequents

    # ---- forward -----------------------------------------------------------

    def forward(self, return_firing=False):
        centers = torch.stack(
            [self.ant_centers[d][self.ant_idx[:, d]] for d in range(self.D)],
            dim=1,
        )
        sigmas = torch.stack(
            [torch.exp(self.ant_log_sigmas[d])[self.ant_idx[:, d]]
             for d in range(self.D)],
            dim=1,
        )

        x = self.inputs.unsqueeze(1)                       # (P, 1, D)
        membership = torch.exp(-0.5 * ((x - centers) / sigmas) ** 2)
        firing = membership.prod(dim=2)                    # (P, R)

        actual_cons = self.actual_consequents()            # (R,) or (R, K)
        weights = torch.exp(self.log_weights)              # (R,)
        wf = firing * weights                              # (P, R)
        denom = wf.sum(dim=1, keepdim=True) + 1e-10        # (P, 1)
        if self.K == 1:
            predictions = (wf * actual_cons).sum(dim=1, keepdim=True) / denom
        else:
            predictions = (wf @ actual_cons) / denom       # (P, K)
        if return_firing:
            return predictions, wf
        return predictions

    # ---- snapshot for result -----------------------------------------------

    def snapshot(self):
        """Numpy copies of current parameters (consequents post-sigmoid)."""
        with torch.no_grad():
            ant_c = [self.ant_centers[d].cpu().numpy().copy()
                     for d in range(self.D)]
            ant_s = [torch.exp(self.ant_log_sigmas[d]).cpu().numpy().copy()
                     for d in range(self.D)]
            cons = self.actual_consequents().cpu().numpy().copy()
            w = torch.exp(self.log_weights).cpu().numpy().copy()
            idx = self.ant_idx.cpu().numpy().copy()
        return ant_c, ant_s, cons, w, idx

    def active_rule_count(self, threshold=1e-4):
        with torch.no_grad():
            return int((torch.exp(self.log_weights) > threshold).sum())

    def to_pruned_rule_set(self, initial_rule_set, threshold=1e-4,
                           sigma_multiplier=2.0):
        """
        Export a pruned RuleSet, physically removing rules whose weight is
        below *threshold* and compacting antecedent indices.
        """
        with torch.no_grad():
            weights = torch.exp(self.log_weights).cpu().numpy()
            consequents = self.actual_consequents().cpu().numpy()
            ant_idx = self.ant_idx.cpu().numpy()
            ant_centers = [self.ant_centers[d].cpu().numpy()
                           for d in range(self.D)]
            ant_sigmas = [torch.exp(self.ant_log_sigmas[d]).cpu().numpy()
                          for d in range(self.D)]

        keep = weights > threshold
        if not keep.any():
            keep[np.argmax(weights)] = True

        kept_idx = np.where(keep)[0]
        kept_weights = weights[kept_idx]
        kept_cons = consequents[kept_idx]
        kept_ant_idx = ant_idx[kept_idx]        # (R', D)

        new_ant_ranges = []
        remapped_ant_idx = kept_ant_idx.copy()
        for d in range(self.D):
            unique_old = sorted(set(kept_ant_idx[:, d].tolist()))
            old_to_new = {old: new for new, old in enumerate(unique_old)}
            remapped_ant_idx[:, d] = [old_to_new[v]
                                      for v in kept_ant_idx[:, d]]
            ranges_d = []
            for old_i in unique_old:
                c = float(ant_centers[d][old_i])
                s = float(ant_sigmas[d][old_i])
                ranges_d.append((c - sigma_multiplier * s,
                                 c + sigma_multiplier * s))
            new_ant_ranges.append(ranges_d)

        w_sum = kept_weights.sum() or 1.0
        norm_weights = (kept_weights / w_sum).tolist()

        rules = [
            (remapped_ant_idx[r].tolist(),
             kept_cons[r].copy() if kept_cons.ndim == 2 else float(kept_cons[r]))
            for r in range(len(kept_cons))
        ]

        return RuleSet(
            rules=rules,
            antecedent_ranges=new_ant_ranges,
            weights=norm_weights,
            counts=getattr(initial_rule_set, "counts", None),
        ).canonicalize()


# ---------------------------------------------------------------------------
# Main entry point: gradient_rulify
# ---------------------------------------------------------------------------

def _build_model_and_optimizer(data_np, rule_set, learning_rate,
                               optimize_antecedents, optimize_consequents,
                               optimize_weights, remaining_iters,
                               consequent_bounds=None):
    """Helper: create model, optimizer, scheduler from a RuleSet."""
    model = _DifferentiableRuleModel(data_np, rule_set,
                                     consequent_bounds=consequent_bounds)
    model.double()

    for d in range(model.D):
        model.ant_centers[d].requires_grad_(optimize_antecedents)
        model.ant_log_sigmas[d].requires_grad_(optimize_antecedents)
    model.consequents.requires_grad_(optimize_consequents)
    model.log_weights.requires_grad_(optimize_weights)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(remaining_iters, 1),
        eta_min=learning_rate * 0.01,
    )
    return model, optimizer, scheduler, trainable


def gradient_rulify(
    data,
    initial_rule_set,
    num_iterations=3000,
    learning_rate=0.01,
    optimize_antecedents=True,
    optimize_consequents=True,
    optimize_weights=True,
    weight_sparsity_lambda=0.01,
    fitness_fn="R2",
    rulescard_lambda=0.0,
    firing_entropy_lambda=0.0,
    ant_overlap_lambda=0.0,
    sigma_reg_lambda=0.0,
    monotonicity_lambda=0.0,
    consequent_bounds=None,
    prune_every=0,
    prune_threshold=1e-4,
    postprocessor_fn=None,
    yield_every=100,
    verbose=False,
    show_progress=True,
):
    """
    Gradient-based rule-set optimisation.

    Directly optimises rule parameters (antecedent centres/widths,
    consequents, rule weights) via gradient descent on a differentiable
    TSK-0 fuzzy system with Gaussian membership functions.

    Typical workflow
    ----------------
    1.  Extract an initial ``RuleSet`` from a trained mesh with
        ``extract_rule_set(model, X)``.
    2.  Call ``gradient_rulify(data, initial_rule_set, ...)`` (with
        sparsity/cardinality penalties and pruning) to fine-tune and
        shrink the rule set.
    3.  Optionally ``result.merge_rules(n_target=...)`` and re-run
        ``gradient_rulify`` on the merged set.
    4.  Export with ``result.to_rule_set()`` (for ``infer_from_ruleset``)
        or infer directly with ``result.infer(inputs)``; visualise with
        ``result.plot_rules()``.

    Parameters
    ----------
    data : ndarray, shape (P, N+1) or (P, N+K)
        Each row has N inputs followed by the output column(s).  The
        number of output columns must match the rule set's consequents:
        1 for scalar consequents (classic), K when *initial_rule_set*
        was extracted with ``output_index=None`` (MIMO / co-dependent
        rule set — ONE shared antecedent structure, K-vector consequents
        optimised jointly; the fitness is the mean per-output R²).  For
        classification meshes, use the class label as the output column
        and pass ``consequent_bounds=(0, K-1)`` plus a rounding
        ``postprocessor_fn``.
    initial_rule_set : RuleSet
        Starting rule set whose structure (antecedent indices) is
        preserved; boundaries, consequents and weights are optimised.
    num_iterations, learning_rate
        Adam steps and initial learning rate (cosine-annealed).
    optimize_antecedents, optimize_consequents, optimize_weights : bool
        Which parameter groups to optimise.
    weight_sparsity_lambda : float
        L1 penalty on rule weights (encourages pruning).
    fitness_fn : {"R2", "concordance_inx", "R2conc"}
        Base metric; append ``"Rulescard"`` (e.g. ``"R2Rulescard"``) to
        also penalise the number of active rules by
        ``rulescard_lambda * active_fraction``.
    rulescard_lambda : float
        Weight of the Rulescard penalty (0.1–0.3 is reasonable).
    firing_entropy_lambda : float
        Penalises high entropy of the per-point normalised firing
        distribution → fewer dominant rules per point (0.01–0.1).
    ant_overlap_lambda : float
        Dimension-wise, consequent-aware antecedent overlap penalty:
        overlap between Gaussians whose rules disagree on the output is
        penalised; agreeing Gaussians may overlap freely (0.05–0.5).
    sigma_reg_lambda : float
        L2 penalty on Gaussian widths, preventing "don't care"
        antecedents with pathologically large sigmas (0.001–0.05 for
        z-scored data).
    monotonicity_lambda : float
        Penalises ``mean_d(1 - ρ_d²)`` where ρ_d is the Pearson
        correlation between Gaussian centres and their effective
        consequents in dimension d — encourages spatially coherent rule
        ordering (0.01–0.1).
    consequent_bounds : (lo, hi) or None
        If given, consequents are constrained to [lo, hi] via a sigmoid
        parameterisation; exported values are always post-sigmoid.
    prune_every : int
        If > 0, physically remove rules with weight < *prune_threshold*
        every this many iterations (model is rebuilt).  500–2000 works
        well; 0 disables.
    prune_threshold : float
        Weight threshold for pruning.
    postprocessor_fn : callable or None
        Applied to predictions before the loss (PyTorch-compatible,
        e.g. ``torch.round`` or ``lambda x: torch.clamp(x, 0, 2)``).
    yield_every : int
        Yield a ``GradientRulifyResult`` every *n* iterations.
    verbose, show_progress : bool
        Output verbosity.

    Yields
    ------
    GradientRulifyResult
        Snapshot of the best result found so far.
    """
    data_np = np.asarray(data, dtype=np.float64)

    # ---- Parse Rulescard suffix ---------------------------------------------
    use_rulescard = fitness_fn.endswith("Rulescard")
    base_fitness_fn = (
        fitness_fn[: -len("Rulescard")] if use_rulescard else fitness_fn
    )
    if base_fitness_fn not in ("R2", "concordance_inx", "R2conc"):
        raise ValueError(
            f"Unknown fitness_fn: {fitness_fn!r}. Expected one of 'R2', "
            f"'concordance_inx', 'R2conc' (optionally + 'Rulescard')."
        )

    # ---- model ---------------------------------------------------------------
    model, optimizer, scheduler, trainable = _build_model_and_optimizer(
        data_np, initial_rule_set, learning_rate,
        optimize_antecedents, optimize_consequents, optimize_weights,
        num_iterations, consequent_bounds=consequent_bounds,
    )

    if model.K > 1 and base_fitness_fn in ("concordance_inx", "R2conc"):
        raise ValueError(
            "concordance-based fitness functions support scalar-consequent "
            "rule sets only; use fitness_fn='R2' (optionally + 'Rulescard') "
            "for MIMO rule sets"
        )

    targets_flat = model.targets.squeeze()
    best_fitness = -float("inf")
    best_snapshot = None
    best_iteration = 0
    total_pruned = 0

    for it in range(num_iterations):
        optimizer.zero_grad()

        if firing_entropy_lambda > 0:
            predictions, wf = model(return_firing=True)
        else:
            predictions = model()
            wf = None
        if postprocessor_fn is not None:
            predictions = postprocessor_fn(predictions)
        preds_flat = predictions.squeeze()

        # ---- loss ------------------------------------------------------------
        ## _r2_torch averages per-column R2 for MIMO (vector-consequent)
        ## rule sets; for scalar rule sets it is the classic R2.
        if base_fitness_fn == "R2":
            r2 = _r2_torch(targets_flat, preds_flat)
            fitness_val = r2
            loss = -r2

        elif base_fitness_fn == "concordance_inx":
            cindex = _differentiable_cindex(targets_flat, preds_flat)
            fitness_val = cindex
            loss = -cindex

        elif base_fitness_fn == "R2conc":
            r2 = _r2_torch(targets_flat, preds_flat)
            cindex = _differentiable_cindex(targets_flat, preds_flat)
            fitness_val = (r2 + cindex) / 2.0
            loss = -fitness_val

        # Rulescard penalty: differentiable approximation of active rules.
        if use_rulescard and rulescard_lambda > 0:
            threshold_logit = -6.9
            steepness = 5.0
            soft_active = torch.sigmoid(
                steepness * (model.log_weights - threshold_logit)
            )
            active_fraction = soft_active.sum() / model.R
            loss = loss + rulescard_lambda * active_fraction
            fitness_val = (
                fitness_val - rulescard_lambda * active_fraction.detach()
            )

        # Weight sparsity (L1 on positive weights)
        if weight_sparsity_lambda > 0 and optimize_weights:
            loss = loss + weight_sparsity_lambda * torch.exp(
                model.log_weights
            ).sum()

        # Sigma regulariser (L2)
        if sigma_reg_lambda > 0 and optimize_antecedents:
            sigma_sq_sum = torch.tensor(0.0, dtype=torch.float64)
            total_sigmas = 0
            for d in range(model.D):
                sigmas_d = torch.exp(model.ant_log_sigmas[d])
                sigma_sq_sum = sigma_sq_sum + (sigmas_d ** 2).sum()
                total_sigmas += len(sigmas_d)
            if total_sigmas > 0:
                loss = loss + sigma_reg_lambda * sigma_sq_sum / total_sigmas

        # Monotonicity penalty (Gaussian-level, weights detached).
        # For MIMO rule sets the per-dimension correlation is computed per
        # output column and averaged.
        if monotonicity_lambda > 0 and optimize_antecedents:
            actual_cons_mono = model.actual_consequents()
            cons2d_mono = (actual_cons_mono if actual_cons_mono.dim() == 2
                           else actual_cons_mono.unsqueeze(1))    # (R, K)
            w_detached = torch.exp(model.log_weights).detach()    # (R,)

            mono_penalty = torch.tensor(0.0, dtype=torch.float64)
            n_dims = 0
            for d in range(model.D):
                K_d = len(model.ant_centers[d])
                if K_d < 2:
                    continue
                centers_d = model.ant_centers[d]                  # (K_d,)
                indices_d = model.ant_idx[:, d]                   # (R,)

                c_mean = centers_d.mean()
                c_centered = centers_d - c_mean
                c_std = c_centered.pow(2).mean().sqrt() + 1e-10

                pen_d = torch.tensor(0.0, dtype=torch.float64)
                for kk in range(cons2d_mono.shape[1]):
                    wq = w_detached * cons2d_mono[:, kk]          # (R,)
                    eff_num = torch.zeros(K_d, dtype=torch.float64)
                    eff_den = torch.zeros(K_d, dtype=torch.float64)
                    eff_num.scatter_add_(0, indices_d, wq)
                    eff_den.scatter_add_(0, indices_d, w_detached)
                    eff_cons_d = eff_num / (eff_den + 1e-10)      # (K_d,)

                    q_mean = eff_cons_d.mean()
                    q_centered = eff_cons_d - q_mean
                    q_std = q_centered.pow(2).mean().sqrt() + 1e-10

                    rho = (c_centered * q_centered).mean() / (c_std * q_std)
                    pen_d = pen_d + (1.0 - rho ** 2)
                mono_penalty = mono_penalty + pen_d / cons2d_mono.shape[1]
                n_dims += 1
            if n_dims > 0:
                loss = loss + monotonicity_lambda * mono_penalty / n_dims

        # Antecedent-overlap penalty (dimension-wise, consequent-aware).
        # For MIMO rule sets, consequent dissimilarity is the per-output
        # normalised difference averaged over the K output columns.
        if ant_overlap_lambda > 0:
            actual_cons = model.actual_consequents()
            cons2d_ov = (actual_cons if actual_cons.dim() == 2
                         else actual_cons.unsqueeze(1))       # (R, K)
            rule_weights = torch.exp(model.log_weights)       # (R,)
            cons_range = (cons2d_ov.max(dim=0).values
                          - cons2d_ov.min(dim=0).values
                          ).detach().clamp(min=1e-10)         # (K,)

            overlap_sum = torch.tensor(0.0, dtype=torch.float64)
            total_pairs = 0
            for d in range(model.D):
                centers_d = model.ant_centers[d]              # (K_d,)
                sigmas_d = torch.exp(model.ant_log_sigmas[d]) # (K_d,)
                K_d = len(centers_d)
                if K_d < 2:
                    continue

                indices_d = model.ant_idx[:, d]               # (R,)
                eff_den = torch.zeros(K_d, dtype=torch.float64)
                eff_den.scatter_add_(0, indices_d, rule_weights)
                eff_cons_cols = []
                for kk in range(cons2d_ov.shape[1]):
                    wq = rule_weights * cons2d_ov[:, kk]      # (R,)
                    eff_num = torch.zeros(K_d, dtype=torch.float64)
                    eff_num.scatter_add_(0, indices_d, wq)
                    eff_cons_cols.append(eff_num / (eff_den + 1e-10))
                eff_cons = torch.stack(eff_cons_cols, dim=1)  # (K_d, K)

                c_i = centers_d.unsqueeze(1)                  # (K, 1)
                c_j = centers_d.unsqueeze(0)                  # (1, K)
                s_i = sigmas_d.unsqueeze(1)
                s_j = sigmas_d.unsqueeze(0)
                dist_sq = (c_i - c_j) ** 2
                sum_var = s_i ** 2 + s_j ** 2
                overlap = torch.exp(-dist_sq / (2.0 * sum_var + 1e-10))

                ec_i = eff_cons.unsqueeze(1)                  # (K_d, 1, K)
                ec_j = eff_cons.unsqueeze(0)                  # (1, K_d, K)
                dissimilarity = ((ec_i - ec_j).abs()
                                 / cons_range).clamp(max=1.0).mean(dim=-1)

                weighted_overlap = overlap * dissimilarity

                mask = torch.triu(
                    torch.ones(K_d, K_d, dtype=torch.bool,
                               device=centers_d.device),
                    diagonal=1,
                )
                overlap_sum = overlap_sum + weighted_overlap[mask].sum()
                total_pairs += K_d * (K_d - 1) // 2
            if total_pairs > 0:
                mean_overlap = overlap_sum / total_pairs
                loss = loss + ant_overlap_lambda * mean_overlap

        # Firing-entropy penalty
        if firing_entropy_lambda > 0 and wf is not None:
            firing_dist = wf / (wf.sum(dim=1, keepdim=True) + 1e-10)
            log_dist = torch.log(firing_dist + 1e-10)
            entropy_per_point = -(firing_dist * log_dist).sum(dim=1)
            max_entropy = np.log(model.R) if model.R > 1 else 1.0
            mean_normalised_entropy = entropy_per_point.mean() / max_entropy
            loss = loss + firing_entropy_lambda * mean_normalised_entropy

        loss.backward()

        # ---- bookkeeping (BEFORE the step: fitness belongs to current params) --
        current_fitness = fitness_val.item()

        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_snapshot = model.snapshot()
            best_iteration = it

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
        optimizer.step()
        scheduler.step()

        if show_progress:
            active = model.active_rule_count(threshold=prune_threshold)
            _progress_bar(
                it + 1, num_iterations,
                f"Rule opt (best: {best_fitness:.4f}, "
                f"rules: {model.R}, active: {active})",
            )

        if verbose:
            print(
                f"  Iter {it+1}/{num_iterations}: "
                f"fitness={current_fitness:.6f}, "
                f"rules={model.R}, "
                f"active="
                f"{model.active_rule_count(threshold=prune_threshold)}, "
                f"lr={scheduler.get_last_lr()[0]:.6f}"
            )

        # ---- periodic pruning ----------------------------------------------
        if (prune_every > 0
                and (it + 1) % prune_every == 0
                and it < num_iterations - 1):
            old_R = model.R
            active = model.active_rule_count(threshold=prune_threshold)

            if active < old_R:
                pruned_rs = model.to_pruned_rule_set(
                    initial_rule_set, threshold=prune_threshold,
                )
                new_R = len(pruned_rs.rules)
                total_pruned += (old_R - new_R)

                if verbose or show_progress:
                    print(
                        f"\n  [Prune @ iter {it+1}] "
                        f"{old_R} → {new_R} rules "
                        f"(removed {old_R - new_R}, "
                        f"total removed: {total_pruned})"
                    )

                remaining = num_iterations - (it + 1)
                model, optimizer, scheduler, trainable = (
                    _build_model_and_optimizer(
                        data_np, pruned_rs, learning_rate,
                        optimize_antecedents, optimize_consequents,
                        optimize_weights, remaining,
                        consequent_bounds=consequent_bounds,
                    )
                )
                targets_flat = model.targets.squeeze()
                best_snapshot = model.snapshot()
                best_fitness = -float("inf")   # old best no longer exists

        # ---- yield -----------------------------------------------------------
        if (it + 1) % yield_every == 0 or it == num_iterations - 1:
            fitness_details = {}
            with torch.no_grad():
                pf = predictions.squeeze()
                fitness_details["R2"] = _r2_torch(targets_flat, pf).item()
                if base_fitness_fn in ("concordance_inx", "R2conc"):
                    fitness_details["cindex"] = _differentiable_cindex(
                        targets_flat, pf
                    ).item()
                fitness_details["active_rules"] = model.active_rule_count(
                    threshold=prune_threshold,
                )
                fitness_details["total_rules"] = model.R

            ant_c, ant_s, cons, w, idx = best_snapshot
            yield GradientRulifyResult(
                iteration=it + 1,
                fitness=best_fitness,
                ant_centers_per_dim=ant_c,
                ant_sigmas_per_dim=ant_s,
                consequents=cons,
                weights=w,
                ant_idx=idx,
                initial_rule_set=initial_rule_set,
                fitness_details=fitness_details,
            )

    if verbose or show_progress:
        print(
            f"\nRule optimisation complete.  "
            f"Best fitness: {best_fitness:.6f} "
            f"(iteration {best_iteration + 1}), "
            f"final rules: {model.R}"
        )
