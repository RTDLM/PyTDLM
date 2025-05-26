"""
TDLM: Trip Distribution Law Models Library

A Python library for simulating trip distribution using various gravity models
and spatial interaction laws.

Author: Maxime Lenormand (2015)
Converted to library format with enhanced parallel processing support
"""

from . import tdlm as tdlm
from .tdlm import run_law_model, gof, TDLMError

__version__ = "0.1.0"
__author__ = "Maxime Lenormand"
__email__ = "maxime.lenormand@inrae.fr"
__license__ = "GPL-3.0"

__all__ = [
    "tdlm",
    "run_law_model", 
    "gof",
    "TDLMError"
]
