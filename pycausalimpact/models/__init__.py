from .statsmodels import StatsmodelsAdapter

# Prophet (optional)
try:
    from .prophet import ProphetAdapter
except ImportError:
    ProphetAdapter = None

# Sktime (optional)
try:
    from .sktime import SktimeAdapter
except ImportError:
    SktimeAdapter = None

# TFP (optional)
try:
    from .tfp import TFPStructuralTimeSeries
except ImportError:
    TFPStructuralTimeSeries = None

__all__ = ["StatsmodelsAdapter"]

if ProphetAdapter is not None:
    __all__.append("ProphetAdapter")

if SktimeAdapter is not None:
    __all__.append("SktimeAdapter")

# Expose TensorFlow Probability adapter if available
if TFPStructuralTimeSeries is not None:
    __all__.append("TFPStructuralTimeSeries")
