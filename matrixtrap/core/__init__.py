"""
Core Module
===========

Fundamental mathematical primitives for the MatrixTrap cryptosystem.
"""

from .field import FiniteField
from .matrix import Matrix

__all__ = ['FiniteField', 'Matrix']
