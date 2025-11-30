"""
Exchange Module
===============

Key exchange functionality for the MatrixTrap cryptosystem.
Derive shared secrets similar to Diffie-Hellman key exchange.
"""

from .key_exchange import derive_shared_secret, derive_key_material

__all__ = ['derive_shared_secret', 'derive_key_material']
