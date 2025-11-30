"""
Public Key
==========

MatrixTrap public key representation and serialization.
"""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.matrix import Matrix


@dataclass
class PublicKey:
    """
    MatrixTrap public key.

    The public key consists of:
    - P: Composite public matrix P = L × D × R (trapdoor hidden)
    - Q: Secondary transformation matrix for encryption
    - n: Matrix dimension
    - prime: Field prime modulus
    - noise_bound: Bound for encryption noise
    """
    P: 'Matrix'         # Composite public matrix P = L × D × R
    Q: 'Matrix'         # Secondary transformation matrix
    n: int              # Matrix dimension
    prime: int          # Field prime
    noise_bound: int    # Bound for encryption noise

    def to_dict(self) -> dict:
        """Serialize public key to a dictionary."""
        return {
            'P': self.P.to_list(),
            'Q': self.Q.to_list(),
            'n': self.n,
            'prime': self.prime,
            'noise_bound': self.noise_bound
        }

    def to_json(self) -> str:
        """Serialize public key to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> 'PublicKey':
        """
        Deserialize public key from a dictionary.

        Args:
            d: Dictionary containing serialized public key

        Returns:
            Reconstructed PublicKey instance
        """
        from ..core.field import FiniteField
        from ..core.matrix import Matrix

        field = FiniteField(d['prime'])
        return cls(
            P=Matrix(d['P'], field),
            Q=Matrix(d['Q'], field),
            n=d['n'],
            prime=d['prime'],
            noise_bound=d['noise_bound']
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'PublicKey':
        """
        Deserialize public key from JSON string.

        Args:
            json_str: JSON string containing serialized public key

        Returns:
            Reconstructed PublicKey instance
        """
        return cls.from_dict(json.loads(json_str))
