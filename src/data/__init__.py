# src/data/__init__.py

# Empty is fine

# Or with exports (if you have multiple functions):
from .load_data import load_data
from .preprocess import preprocess_data
from .feature_engineering import create_sequences

__all__ = [
    "load_data",
    "preprocess_data",
    "create_sequences",
]
