from .statsmodels import StatsmodelsAdapter

# Prophet (opcional)
try:
    from .prophet import ProphetAdapter
except ImportError:
    ProphetAdapter = None

# Sktime (opcional)
try:
    from .sktime import SktimeAdapter
except ImportError:
    SktimeAdapter = None

# TFP (opcional)
try:
    from .tfp import TFPStructuralTimeSeries
except ImportError:
    TFPStructuralTimeSeries = None

__all__ = ["StatsmodelsAdapter"]

if ProphetAdapter is not None:
    __all__.append("ProphetAdapter")

if SktimeAdapter is not None:
    __all__.append("SktimeAdapter")

if TFPStructuralTimeSeries is not None:
    __all__.append("TFPStructuralTimeSeries")
