"""
TDLM: Systematic comparison of trip distribution laws and models in Python

A Python port of the TDLM R package, with 
numpy-based implementations and parallel processing support for multiple 
exponent values.

Author: Rémi Perrier (2025)
"""

from . import tdlm as tdlm
from .tdlm import extract_opportunities, run_optimization, run_law_model_gof, run_law_model, run_law, run_model, gof, _TDLMError

__version__ = "0.2.2"
__author__ = "Rémi Perrier"
__email__ = "remi.perrier@cnrs.fr"
__license__ = "GPL-3.0"

__all__ = [
    "tdlm",
    "extract_opportunities",
    "run_optimization",
    "run_law_model_gof",
    "run_law_model",
    "run_law",
    "run_model",
    "gof",
    "_TDLMError"
]
