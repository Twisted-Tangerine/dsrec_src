"""
DSRec Models Package

This package contains model definitions for sequential recommendation:
- backbone: Base sequential recommendation models (SASRec, BERT4Rec, GRU4Rec)
- components: Reusable neural network components (attention, loss functions, etc.)
- dsrec: DSRec models with semantic adapters and alignment loss
"""

from .backbone import SASRec, Bert4Rec, GRU4Rec
from .dsrec import (
    DSRecSASRec,
    DSRecBert4Rec,
    DSRecGRU4Rec,
    SASRecPLUS,
    Bert4RecPLUS,
    GRU4RecPLUS
)

__all__ = [
    # Base models
    'SASRec',
    'Bert4Rec',
    'GRU4Rec',
    # PLUS variants (with semantic adapters)
    'SASRecPLUS',
    'Bert4RecPLUS',
    'GRU4RecPLUS',
    # DSRec models (with alignment loss)
    'DSRecSASRec',
    'DSRecBert4Rec',
    'DSRecGRU4Rec',
]
