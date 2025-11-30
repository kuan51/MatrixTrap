"""
Key Generation
==============

Generate MatrixTrap key pairs.
"""

from typing import Tuple

from ..core.field import FiniteField
from ..core.matrix import Matrix
from .public import PublicKey
from .private import PrivateKey


# Default parameters
DEFAULT_PRIME = 2**127 - 1  # Mersenne prime for efficiency
DEFAULT_DIMENSION = 8
DEFAULT_NOISE_BOUND = 1000


def generate_keypair(
    prime: int = None,
    dimension: int = None,
    noise_bound: int = None
) -> Tuple[PublicKey, PrivateKey]:
    """
    Generate a new MatrixTrap public/private key pair.

    The key generation process:
    1. Generate random invertible matrices L, R (trapdoor matrices)
    2. Generate diagonal matrix D with non-zero entries (secret)
    3. Compute public matrix P = L × D × R
    4. Generate secondary matrix Q (invertible)
    5. Compute all necessary inverses

    Args:
        prime: Field prime modulus (default: 2^127 - 1)
        dimension: Matrix dimension n (default: 8)
        noise_bound: Bound for encryption noise (default: 1000)

    Returns:
        Tuple of (PublicKey, PrivateKey)

    Example:
        >>> from matrixtrap.keys import generate_keypair
        >>> public_key, private_key = generate_keypair()
    """
    prime = prime or DEFAULT_PRIME
    n = dimension or DEFAULT_DIMENSION
    noise_bound = noise_bound or DEFAULT_NOISE_BOUND

    field = FiniteField(prime)

    # Generate trapdoor matrices
    L = Matrix.random_invertible(n, field)
    R = Matrix.random_invertible(n, field)

    # Generate diagonal secret matrix with non-zero entries
    diag = [field.random_nonzero() for _ in range(n)]
    D = Matrix.diagonal(diag, field)

    # Compute public matrix P = L × D × R
    P = L.multiply(D).multiply(R)

    # Generate secondary transformation matrix
    Q = Matrix.random_invertible(n, field)

    # Compute inverses for private key
    L_inv = L.inverse()
    R_inv = R.inverse()
    diag_inv = [field.inv(d) for d in diag]
    D_inv = Matrix.diagonal(diag_inv, field)
    Q_inv = Q.inverse()

    public_key = PublicKey(
        P=P, Q=Q, n=n,
        prime=prime, noise_bound=noise_bound
    )

    private_key = PrivateKey(
        L=L, L_inv=L_inv,
        R=R, R_inv=R_inv,
        D=D, D_inv=D_inv,
        Q_inv=Q_inv,
        n=n, prime=prime
    )

    return public_key, private_key
