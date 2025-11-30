"""
Signing Module
==============

Create and verify digital signatures using the MatrixTrap cryptosystem.
"""

from .signature import sign, verify

__all__ = ['sign', 'verify']
