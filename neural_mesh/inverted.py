"""
Inverted Neural Mesh — a closed-form, *non-neural* multilinear model.

This module is deliberately kept apart from the rest of the package: it uses
no torch, no autograd and no membership/complexity layers.  It is a direct
implementation of the mode-matrix estimation in ``inverted-neuralmesh/code.m``.

The idea
--------
In the standard Neural Mesh the mesh grid holds the *rule firing strengths*
(an outer product of per-dimension complexity activations) and the learned
object is a free weight tensor over that grid.  In the **inverted** mesh the
mesh grid holds the *input tensor itself*, and the learned objects are one
matrix per mode::

    T_k  =  S_k  x_1 U_1  x_2 U_2  ...  x_N U_N

for sample pairs ``(S_k, T_k)``, ``k = 1..K``, where ``S_k`` has shape
``(I_1, ..., I_N)``, ``T_k`` has shape ``(J_1, ..., J_N)`` and
``U_n`` is ``J_n x I_n``.

Vectorised, this says the full input->output map is a Kronecker product::

    vec(T) = (U_1 (x) U_2 (x) ... (x) U_N) vec(S)          [C / row-major order]

so the map has ``prod(J_n) * prod(I_n)`` entries but only ``sum(J_n * I_n)``
free parameters.

Three estimators are provided
-----------------------------
``method="lstsq"``  (the default; a faithful port of ``code.m``)
    1. Vectorise every sample and fit the *unconstrained* map
       ``M = Y @ pinv(X)``.
    2. Re-index ``M`` into ``(j_1,i_1, j_2,i_2, ..., j_N,i_N)`` order.
    3. Take a rank-1 CP approximation of the result (dominant left singular
       vector of each mode unfolding, then one common scale factor).
    4. Reshape each mode vector back into ``U_n : J_n x I_n``.

    Step 1 forms a ``prod(J) x prod(I)`` matrix, so this route is only
    practical for small mode sizes, and it needs ``K >= prod(I_n)`` samples
    for ``pinv(X)`` to be well determined.  Step 3 is exact for ``N = 2``
    (the SVD gives the best rank-1 matrix approximation, cf. Van Loan &
    Pitsianis 1993) but is *not* optimal for ``N >= 3``.

``method="als"``
    Alternating least squares directly on
    ``min_{U_1..U_N} sum_k || T_k - S_k x_1 U_1 ... x_N U_N ||^2``.
    Never forms ``M``, so it is stable when ``K`` is close to ``prod(I_n)``
    and has no order-3 optimality problem.  Optional ridge penalty.

    Each sweep exactly minimises the objective over one mode matrix with the
    others held fixed, so the loss is monotone non-increasing.  The objective
    is NOT convex, though, so a poor starting point can converge to a local
    minimum that is worse than what ``"lstsq"`` returns.  Hence:

``method="lstsq+als"``
    Run ``"lstsq"`` first and use its mode matrices to start ``"als"``.
    The closed-form solution supplies the initialisation; ALS then refines
    it in the metric that actually matters (fit error rather than distance
    to the unconstrained map).  This is normally the best of the three:
    it cannot end up worse than ``"lstsq"``, and it avoids the bad basins
    that plain ``"als"`` can fall into.  ``errT_init_`` records the fit
    error of the initialisation, so the ALS gain is visible.

    It does form ``M`` (via the ``"lstsq"`` stage), so it inherits that
    route's ``K >= prod(I_n)`` requirement and its memory cost.

References
----------
Van Loan & Pitsianis (1993), *Approximation with Kronecker products*.
De Lathauwer, De Moor & Vandewalle (2000), *On the best rank-1 and
rank-(R1,...,RN) approximation of higher-order tensors*, SIMAX 21(4):1324.
Hoff (2015), *Multilinear tensor regression for longitudinal relational data*.
Lock (2018), *Tensor-on-tensor regression*, JCGS 27(3).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "estimate_mode_matrices",
    "InvertedNeuralMesh",
    "mode_product",
    "kron_list",
]


# ---------------------------------------------------------------------------
# Small tensor helpers (sample-first convention: axis 0 indexes samples)
# ---------------------------------------------------------------------------

def mode_product(A: np.ndarray, U: np.ndarray, mode: int) -> np.ndarray:
    """``A x_mode U`` for a batch of tensors.

    Parameters
    ----------
    A : ndarray, shape (K, d_1, ..., d_N)
        Batch of tensors; axis 0 indexes samples.
    U : ndarray, shape (J, d_mode)
    mode : int
        1-based mode index into the *tensor* part (axis ``mode`` of ``A``).

    Returns
    -------
    ndarray, shape (K, d_1, ..., J, ..., d_N)
    """
    out = np.tensordot(A, U, axes=([mode], [1]))
    return np.moveaxis(out, -1, mode)


def multi_mode_product(A: np.ndarray, Us: Sequence[np.ndarray],
                       skip: Optional[int] = None) -> np.ndarray:
    """Apply ``U_n`` along every tensor mode of ``A`` (optionally skipping one).

    ``skip`` is a 1-based mode index, matching :func:`mode_product`.
    """
    out = A
    for n, U in enumerate(Us, start=1):
        if skip is not None and n == skip:
            continue
        out = mode_product(out, U, n)
    return out


def kron_list(Us: Sequence[np.ndarray]) -> np.ndarray:
    """``U_1 (x) U_2 (x) ... (x) U_N`` — the vectorised map, in C order.

    Only useful for inspection/diagnostics: the result has
    ``prod(J_n) x prod(I_n)`` entries.
    """
    out = np.asarray(Us[0])
    for U in Us[1:]:
        out = np.kron(out, U)
    return out


def _unfold(A: np.ndarray, mode: int) -> np.ndarray:
    """Mode-``mode`` unfolding (0-based) of a single tensor."""
    return np.reshape(np.moveaxis(A, mode, 0), (A.shape[mode], -1))


def _outer(vs: Sequence[np.ndarray]) -> np.ndarray:
    out = vs[0]
    for v in vs[1:]:
        out = np.multiply.outer(out, v)
    return out


# ---------------------------------------------------------------------------
# The code.m estimator
# ---------------------------------------------------------------------------

def estimate_mode_matrices(
    S: np.ndarray,
    T: np.ndarray,
    rcond: Optional[float] = None,
) -> Tuple[list, np.ndarray, float, float]:
    """Port of ``estimate_mode_matrices`` from ``inverted-neuralmesh/code.m``.

    Parameters
    ----------
    S : ndarray, shape (K, I_1, ..., I_N)
    T : ndarray, shape (K, J_1, ..., J_N)
        Sample-FIRST (the MATLAB original is sample-last).
    rcond : float, optional
        Cutoff passed to ``numpy.linalg.pinv``.

    Returns
    -------
    Us : list of ndarray
        ``U_n`` of shape ``(J_n, I_n)``.  The common scale is folded into
        ``U_1``, as in the MATLAB original.
    M : ndarray, shape (prod(J), prod(I))
        The unconstrained least-squares map, before the rank-1 restriction.
    errM : float
        ``||M - Mhat||_F / ||M||_F`` — how much of the unconstrained map the
        Kronecker structure throws away.  This is the diagnostic to look at
        before trusting the model.
    errT : float
        Relative training-fit error of the structured map.
    """
    S = np.asarray(S, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    K = S.shape[0]
    if T.shape[0] != K:
        raise ValueError("S and T must have the same number of samples.")
    N = S.ndim - 1
    if T.ndim - 1 != N:
        raise ValueError(
            f"S and T must have the same number of modes "
            f"(got {N} and {T.ndim - 1}). Use InvertedNeuralMesh(output_shape=...) "
            f"to reshape a flat target into the right number of modes."
        )

    I = list(S.shape[1:])
    J = list(T.shape[1:])
    IS, JT = int(np.prod(I)), int(np.prod(J))

    # 2. vectorise every sample:  Y ~ M X
    X = S.reshape(K, IS).T                     # (IS, K)
    Y = T.reshape(K, JT).T                     # (JT, K)

    # 3. unconstrained map
    M = Y @ np.linalg.pinv(X, rcond=rcond) if rcond is not None else Y @ np.linalg.pinv(X)

    # 4. re-index M as (j_1,i_1, j_2,i_2, ..., j_N,i_N) and collapse each pair
    MM = M.reshape(J + I)
    order = []
    for n in range(N):
        order += [n, N + n]
    MM = MM.transpose(order)
    D = [J[n] * I[n] for n in range(N)]
    R = MM.reshape(D)

    # 5. rank-1 CP approximation (dominant left singular vector per mode)
    u = []
    for n in range(N):
        Rn = _unfold(R, n)
        P, _, _ = np.linalg.svd(Rn, full_matrices=False)
        u.append(P[:, 0])

    # 6. common scale factor
    rhat = _outer(u)
    lam = float((rhat * R).sum())

    # 7. back to matrices; the whole scale goes into U_1 (as in code.m)
    Us = [u[n].reshape(J[n], I[n]) for n in range(N)]
    Us[0] = lam * Us[0]

    # 8/9. diagnostics
    Mhat = kron_list(Us)
    errM = float(np.linalg.norm(M - Mhat) / (np.linalg.norm(M) + 1e-300))
    Yhat = Mhat @ X
    errT = float(np.linalg.norm(Y - Yhat) / (np.linalg.norm(Y) + 1e-300))

    return Us, M, errM, errT


# ---------------------------------------------------------------------------
# Model object
# ---------------------------------------------------------------------------

class InvertedNeuralMesh:
    """Inverted Neural Mesh: ``T_k = S_k x_1 U_1 x_2 U_2 ... x_N U_N``.

    The mesh holds the input tensor; the only learned objects are the
    per-mode matrices ``U_n``.  There is no membership layer, no complexity
    layer and no rule tensor — this is the closed-form multilinear model
    described in ``code.m``, not a neural network.

    Parameters
    ----------
    output_shape : tuple of int, optional
        Shape of one output sample as a tensor with the SAME number of modes
        as one input sample.  Required whenever the targets are supplied
        flat.  Example: inputs ``(K, 20, 3)`` with targets ``(K, 3)`` need
        ``output_shape=(1, 3)`` — mode 1 collapses the 20 positions to a
        single combination, mode 2 maps xyz -> xyz.
    method : {"lstsq", "als", "lstsq+als"}
        ``"lstsq"`` reproduces ``code.m`` (fit the unconstrained map, then
        project onto the Kronecker manifold).  ``"als"`` fits the structured
        model directly and never forms the unconstrained map; it is the more
        stable choice when ``K`` is not comfortably larger than
        ``prod(I_n)``, and the better one for ``N >= 3``, but from a poor
        start it can settle in a local minimum worse than ``"lstsq"``.
        ``"lstsq+als"`` starts ALS from the ``"lstsq"`` solution and is
        normally the best choice.
    fit_intercept : bool
        Centre ``S`` and ``T`` on the training means and restore the offset
        at predict time.  The multilinear map has no constant term of its
        own, so this is usually wanted.
    ridge : float
        L2 penalty on the mode matrices (``method="als"`` only).
    als_iters : int
        Maximum alternating sweeps.
    tol : float
        Relative-improvement stopping tolerance for ALS.
    random_state : int, optional
        Seed for the ALS initialisation.

    Attributes
    ----------
    mode_matrices_ : list of ndarray
    errM_ : float or None
        Kronecker-approximation error of the unconstrained map (``"lstsq"``
        and ``"lstsq+als"``).  Large values mean the structure is discarding
        a lot of the fitted relationship.  NOTE: this is a parameter-space
        distance, not predictive damage -- an errM of 0.24 can correspond to
        anything from a negligible to a catastrophic loss of accuracy,
        depending on how large the affected inputs are.  Use held-out R2 to
        judge what the structure actually costs.
    errT_ : float
        Relative training-fit error of the fitted model.
    errT_init_ : float or None
        (``"lstsq+als"`` only) the same quantity for the initialisation, so
        the improvement contributed by ALS is visible.
    n_params_ : int
    """

    def __init__(
        self,
        output_shape: Optional[Sequence[int]] = None,
        method: str = "lstsq",
        fit_intercept: bool = True,
        ridge: float = 0.0,
        als_iters: int = 200,
        tol: float = 1e-9,
        random_state: Optional[int] = None,
    ):
        if method not in ("lstsq", "als", "lstsq+als"):
            raise ValueError(
                f"method must be 'lstsq', 'als' or 'lstsq+als', got {method!r}")
        self.output_shape = tuple(output_shape) if output_shape is not None else None
        self.method = method
        self.fit_intercept = fit_intercept
        self.ridge = float(ridge)
        self.als_iters = int(als_iters)
        self.tol = float(tol)
        self.random_state = random_state

    # -- shape bookkeeping --------------------------------------------------

    def _as_tensor_targets(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=np.float64)
        if self.output_shape is None:
            return T
        return T.reshape((T.shape[0],) + self.output_shape)

    def _restore_target_shape(self, T: np.ndarray) -> np.ndarray:
        if self._target_ndim_ == T.ndim:
            return T
        return T.reshape((T.shape[0],) + self._target_flat_shape_)

    # -- fitting ------------------------------------------------------------

    def fit(self, S: np.ndarray, T: np.ndarray) -> "InvertedNeuralMesh":
        S = np.asarray(S, dtype=np.float64)
        T_in = np.asarray(T, dtype=np.float64)
        self._target_ndim_ = T_in.ndim
        self._target_flat_shape_ = T_in.shape[1:]
        Tt = self._as_tensor_targets(T_in)

        if S.ndim - 1 != Tt.ndim - 1:
            raise ValueError(
                f"After reshaping, S has {S.ndim - 1} modes and T has "
                f"{Tt.ndim - 1}. Pass output_shape=... with the same number "
                f"of modes as one input sample."
            )

        if self.fit_intercept:
            self.S_mean_ = S.mean(axis=0, keepdims=True)
            self.T_mean_ = Tt.mean(axis=0, keepdims=True)
        else:
            self.S_mean_ = np.zeros((1,) + S.shape[1:])
            self.T_mean_ = np.zeros((1,) + Tt.shape[1:])
        Sc, Tc = S - self.S_mean_, Tt - self.T_mean_

        self.input_shape_ = S.shape[1:]
        self.target_shape_ = Tt.shape[1:]

        def _rel_fit(Us):
            pred = multi_mode_product(Sc, Us)
            return float(np.linalg.norm(Tc - pred) / (np.linalg.norm(Tc) + 1e-300))

        if self.method == "lstsq":
            Us, M, errM, errT = estimate_mode_matrices(Sc, Tc)
            self.mode_matrices_ = Us
            self.errM_ = errM
            self.errT_ = errT
            self.errT_init_ = None
            self.unconstrained_map_ = M

        elif self.method == "lstsq+als":
            ## closed form first, then refine it in the metric that matters
            Us0, M, errM, errT0 = estimate_mode_matrices(Sc, Tc)
            self.mode_matrices_ = self._fit_als(Sc, Tc, init=Us0)
            self.errM_ = errM                 # from the initialisation stage
            self.errT_init_ = errT0
            self.errT_ = _rel_fit(self.mode_matrices_)
            self.unconstrained_map_ = M
            ## ALS is monotone from its start, so this should never regress;
            ## guard against a pathological solve rather than fail silently.
            if self.errT_ > errT0 + 1e-9:
                self.mode_matrices_ = Us0
                self.errT_ = errT0

        else:  # "als"
            self.mode_matrices_ = self._fit_als(Sc, Tc)
            self.errM_ = None
            self.errT_init_ = None
            self.unconstrained_map_ = None
            self.errT_ = _rel_fit(self.mode_matrices_)

        self.n_params_ = int(sum(U.size for U in self.mode_matrices_))
        return self

    def _fit_als(self, S: np.ndarray, T: np.ndarray,
                 init: Optional[Sequence[np.ndarray]] = None) -> list:
        """Alternating least squares.

        ``init`` supplies the starting mode matrices; ``method="lstsq+als"``
        passes the closed-form solution here.  The objective is non-convex,
        so the starting point genuinely matters.
        """
        rng = np.random.default_rng(self.random_state)
        I = list(S.shape[1:])
        J = list(T.shape[1:])
        N = len(I)

        if init is not None:
            Us = [np.array(U, dtype=np.float64, copy=True) for U in init]
        else:
            # Fallback start: identity-ish where shapes allow, else small random.
            Us = []
            for n in range(N):
                U = np.zeros((J[n], I[n]))
                m = min(J[n], I[n])
                U[:m, :m] = np.eye(m)
                if J[n] == 1:
                    U[:] = 1.0 / I[n]                 # collapse -> mean
                U = U + 0.01 * rng.normal(size=U.shape)
                Us.append(U)

        prev = np.inf
        sweeps = 0
        for sweeps in range(1, self.als_iters + 1):
            for n in range(1, N + 1):
                Z = multi_mode_product(S, Us, skip=n)   # (K, J.., I_n, ..J)
                # unfold both along mode n (axis n), samples folded into cols
                Zu = np.reshape(np.moveaxis(Z, n, 0), (Z.shape[n], -1))
                Tu = np.reshape(np.moveaxis(T, n, 0), (T.shape[n], -1))
                G = Zu @ Zu.T
                if self.ridge:
                    G = G + self.ridge * np.eye(G.shape[0])
                Us[n - 1] = (Tu @ Zu.T) @ np.linalg.pinv(G)
            resid = float(np.linalg.norm(T - multi_mode_product(S, Us)))
            if np.isfinite(prev) and prev - resid <= self.tol * max(prev, 1e-300):
                break
            prev = resid
        self.als_sweeps_ = sweeps
        return Us

    # -- use ----------------------------------------------------------------

    def predict(self, S: np.ndarray) -> np.ndarray:
        S = np.asarray(S, dtype=np.float64)
        out = multi_mode_product(S - self.S_mean_, self.mode_matrices_) + self.T_mean_
        return self._restore_target_shape(out)

    def score(self, S: np.ndarray, T: np.ndarray,
              multioutput: str = "uniform_average") -> float:
        """R^2, matching ``sklearn.metrics.r2_score`` defaults."""
        T = np.asarray(T, dtype=np.float64)
        P = np.asarray(self.predict(S), dtype=np.float64)
        Tf = T.reshape(T.shape[0], -1)
        Pf = P.reshape(P.shape[0], -1)
        ss_res = ((Tf - Pf) ** 2).sum(axis=0)
        ss_tot = ((Tf - Tf.mean(axis=0)) ** 2).sum(axis=0)
        r2 = 1.0 - ss_res / np.where(ss_tot == 0, 1.0, ss_tot)
        if multioutput == "raw_values":
            return r2
        return float(np.mean(r2))

    def coefficient_map(self) -> np.ndarray:
        """The full vectorised map ``U_1 (x) ... (x) U_N`` (diagnostics only)."""
        return kron_list(self.mode_matrices_)

    def summary(self) -> str:
        full = int(np.prod(self.target_shape_)) * int(np.prod(self.input_shape_))
        lines = [
            f"InvertedNeuralMesh  method={self.method}",
            f"  input  shape per sample: {tuple(self.input_shape_)}",
            f"  output shape per sample: {tuple(self.target_shape_)}"
            f"  (returned as {self._target_flat_shape_})",
            f"  mode matrices: " + ", ".join(
                f"U{n+1} {U.shape}" for n, U in enumerate(self.mode_matrices_)),
            f"  learned parameters: {self.n_params_:,}"
            f"   (unconstrained linear map would need {full:,})",
            f"  training fit error (relative): {self.errT_:.4f}",
        ]
        if getattr(self, "errT_init_", None) is not None:
            lines.append(
                f"    from lstsq initialisation:  {self.errT_init_:.4f}"
                f"   -> ALS improved it by {self.errT_init_ - self.errT_:.4g}")
        if self.errM_ is not None:
            lines.append(
                f"  Kronecker approximation error errM: {self.errM_:.4f}"
                f"   (parameter-space; not a measure of predictive damage)")
        return "\n".join(lines)
