"""
Trainers Package

This package contains training logic for sequential recommendation models:
- trainer: Base trainer class with common training infrastructure
- sequence_trainer: Sequence-specific trainer implementation
"""

from .trainer import Trainer
from .sequence_trainer import SeqTrainer

__all__ = [
    'Trainer',
    'SeqTrainer',
]
