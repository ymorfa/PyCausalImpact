"""PyCausalImpact: Library for estimating causal impact on time series.

This package exposes :class:`~pycausalimpact.core.CausalImpactPy`, which now
provides :meth:`~pycausalimpact.core.CausalImpactPy.get_posterior_samples` to
retrieve posterior predictive and effect samples.
"""

from .core import CausalImpactPy

__all__ = ["CausalImpactPy"]
__version__ = "0.1.0"
