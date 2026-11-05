"""Federated Learning Platform (simulation-first).

Train a shared model across distributed clients without moving their data.
Pure-stdlib core; FastAPI is an optional service layer.
"""

from flp.config import FLConfig
from flp.model import LogisticModel

__all__ = ["FLConfig", "LogisticModel"]
__version__ = "0.1.0"
