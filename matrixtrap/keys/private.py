"""
Private Key
===========

MatrixTrap private key representation and serialization.
"""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.matrix import Matrix


@dataclass
class PrivateKey:
    """
    MatrixTrap private key.

    The private key contains the trapdoor matrices that allow
    decomposition of the public matrix P = L × D × R:
    - L, L_inv: Left trapdoor matrix and its inverse
    - R, R_inv: Right trapdoor matrix and its inverse
    - D, D_inv: Diagonal secret matrix and its inverse
    - Q_inv: Inverse of the secondary transformation matrix
    - n: Matrix dimension
    - prime: Field prime modulus
    """
    L: 'Matrix'         # Left trapdoor matrix
    L_inv: 'Matrix'     # Inverse of L
    R: 'Matrix'         # Right trapdoor matrix
    R_inv: 'Matrix'     # Inverse of R
    D: 'Matrix'         # Diagonal secret matrix
    D_inv: 'Matrix'     # Inverse of D
    Q_inv: 'Matrix'     # Inverse of Q
    n: int              # Matrix dimension
    prime: int          # Field prime

    def to_dict(self) -> dict:
        """Serialize private key to a dictionary."""
        return {
            'L': self.L.to_list(),
            'L_inv': self.L_inv.to_list(),
            'R': self.R.to_list(),
            'R_inv': self.R_inv.to_list(),
            'D': self.D.to_list(),
            'D_inv': self.D_inv.to_list(),
            'Q_inv': self.Q_inv.to_list(),
            'n': self.n,
            'prime': self.prime
        }

    def to_json(self) -> str:
        """Serialize private key to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> 'PrivateKey':
        """
        Deserialize private key from a dictionary.

        Args:
            d: Dictionary containing serialized private key

        Returns:
            Reconstructed PrivateKey instance
        """
        from ..core.field import FiniteField
        from ..core.matrix import Matrix

        field = FiniteField(d['prime'])
        return cls(
            L=Matrix(d['L'], field),
            L_inv=Matrix(d['L_inv'], field),
            R=Matrix(d['R'], field),
            R_inv=Matrix(d['R_inv'], field),
            D=Matrix(d['D'], field),
            D_inv=Matrix(d['D_inv'], field),
            Q_inv=Matrix(d['Q_inv'], field),
            n=d['n'],
            prime=d['prime']
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'PrivateKey':
        """
        Deserialize private key from JSON string.

        Args:
            json_str: JSON string containing serialized private key

        Returns:
            Reconstructed PrivateKey instance
        """
        return cls.from_dict(json.loads(json_str))
