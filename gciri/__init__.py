"""Model code accompanying the GCIRI city-resilience paper."""

from .downscale import ALPHA_E, ALPHA_H, ALPHA_V
from .fusion import ABS_6, REC_6, RESP_6, assign_modes, fuse_scores
from .trapezoid import compute_three_capabilities, compute_trapezoid_giri
from .workflow import run_gciri_workflow

__all__ = [
    "ABS_6",
    "REC_6",
    "RESP_6",
    "ALPHA_H",
    "ALPHA_E",
    "ALPHA_V",
    "assign_modes",
    "compute_three_capabilities",
    "compute_trapezoid_giri",
    "fuse_scores",
    "run_gciri_workflow",
]
