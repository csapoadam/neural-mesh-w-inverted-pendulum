"""
Neural Mesh Architecture
========================

A fuzzy-rule-based tensor product neural network that uses triangle
membership functions as antecedents and learned weighted consequents,
combined via a tensor-product meshing across input dimensions.

Structural (hyperparameters, set via grid search):
    G = (G1, G2, ..., Gn)  — resolution per dimension (number of fuzzy sets)
    C = (C1, C2, ..., Cn)  — complexity per dimension (rule-base factors)

Learnable parameters (optimised via gradient descent):
    Triangle supports   — centres and widths of each fuzzy set
    Complexity weights  — w_{i,c,g} that map membership degrees to activations
    Rule bias           — b per rule (or per rule × output)
    Rule weight         — s per rule (or per rule × output)
"""

from neural_mesh.model import NeuralMeshModel
from neural_mesh.training import train, evaluate
from neural_mesh.recorder import ActivationRecorder
from neural_mesh.visualization import MeshVisualizer
from neural_mesh.rules import (
    RuleSet, extract_rule_set, infer_from_ruleset,
    gradient_rulify, GradientRulifyResult,
)

## ── Inverted Neural Mesh ─────────────────────────────────────────────────
## Closed-form, NON-neural multilinear model: the mesh holds the input
## tensor and the only learned objects are one matrix per mode
## (T_k = S_k x_1 U_1 ... x_N U_N).  The module itself needs only numpy --
## no torch, no autograd -- though importing it through this package still
## runs the torch-backed imports above.
from neural_mesh.inverted import (
    InvertedNeuralMesh, estimate_mode_matrices,
    mode_product, multi_mode_product, kron_list,
)

__all__ = [
    "NeuralMeshModel", "train", "evaluate",
    "ActivationRecorder", "MeshVisualizer",
    "RuleSet", "extract_rule_set", "infer_from_ruleset",
    "gradient_rulify", "GradientRulifyResult",
    "InvertedNeuralMesh", "estimate_mode_matrices",
    "mode_product", "multi_mode_product", "kron_list",
]
