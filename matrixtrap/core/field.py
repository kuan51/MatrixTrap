"""
Finite Field Operations
=======================

Operations in a prime finite field GF(p).
"""

import secrets


class FiniteField:
    """Operations in a prime finite field GF(p)."""

    def __init__(self, prime: int):
        """
        Initialize a finite field with the given prime modulus.

        Args:
            prime: A prime number defining the field GF(p)
        """
        self.p = prime

    def add(self, a: int, b: int) -> int:
        """Add two field elements."""
        return (a + b) % self.p

    def sub(self, a: int, b: int) -> int:
        """Subtract two field elements."""
        return (a - b) % self.p

    def mul(self, a: int, b: int) -> int:
        """Multiply two field elements."""
        return (a * b) % self.p

    def pow(self, base: int, exp: int) -> int:
        """Compute base^exp in the field."""
        return pow(base, exp, self.p)

    def inv(self, a: int) -> int:
        """
        Compute modular multiplicative inverse using Fermat's little theorem.

        Args:
            a: Field element to invert

        Returns:
            The multiplicative inverse a^(-1) mod p

        Raises:
            ValueError: If a is zero
        """
        if a == 0:
            raise ValueError("Cannot invert zero")
        return pow(a, self.p - 2, self.p)

    def neg(self, a: int) -> int:
        """Compute additive inverse (negation)."""
        return (-a) % self.p

    def random(self) -> int:
        """Generate a random field element."""
        return secrets.randbelow(self.p)

    def random_nonzero(self) -> int:
        """Generate a random non-zero field element."""
        while True:
            r = self.random()
            if r != 0:
                return r

    def __repr__(self) -> str:
        return f"FiniteField(p={self.p})"
