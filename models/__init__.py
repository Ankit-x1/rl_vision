"""Model package initialization."""

from .backbone import ResNet18EarlyExit, create_model
from .early_exit import EarlyExitHead, MultiExitHead

__all__ = [
    'ResNet18EarlyExit',
    'create_model',
    'EarlyExitHead',
    'MultiExitHead'
]
