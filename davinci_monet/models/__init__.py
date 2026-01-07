"""Model data handling module.

This module provides classes for loading and processing atmospheric
model output from various sources (CMAQ, WRF-Chem, UFS, etc.).
"""

from davinci_monet.models.base import (
    ModelData,
    create_model_data,
)

__all__ = [
    "ModelData",
    "create_model_data",
]
