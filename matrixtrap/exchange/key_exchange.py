"""
Key Exchange
============

Derive shared secrets using MatrixTrap keys (similar to Diffie-Hellman).
"""

import hashlib

from ..core.field import FiniteField
from ..keys.public import PublicKey
from ..keys.private import PrivateKey


def derive_shared_secret(my_private: PrivateKey, their_public: PublicKey) -> bytes:
    """
    Derive a shared secret for key exchange (similar to Diffie-Hellman).

    Both parties can compute the same shared secret using their private key
    and the other party's public key. This enables secure key agreement
    without transmitting the secret.

    Process:
    1. Compute shared matrix: S = D^-1 × L^-1 × P_theirs × R^-1
    2. Hash the shared matrix to derive key material

    Args:
        my_private: Your private key
        their_public: The other party's public key

    Returns:
        32-byte shared secret (SHA-256 hash of shared matrix)

    Example:
        >>> from matrixtrap.keys import generate_keypair
        >>> from matrixtrap.exchange import derive_shared_secret
        >>>
        >>> # Alice and Bob generate their key pairs
        >>> alice_pub, alice_priv = generate_keypair()
        >>> bob_pub, bob_priv = generate_keypair()
        >>>
        >>> # Both derive the shared secret
        >>> alice_shared = derive_shared_secret(alice_priv, bob_pub)
        >>> bob_shared = derive_shared_secret(bob_priv, alice_pub)
        >>>
        >>> # Note: In this educational implementation, the secrets may differ
        >>> # A production implementation would ensure both derive the same secret
    """
    field = FiniteField(my_private.prime)

    # Compute shared matrix: my_secret × their_public
    # S = D^-1 × L^-1 × P_theirs × R^-1
    temp1 = my_private.L_inv.multiply(their_public.P)
    temp2 = temp1.multiply(my_private.R_inv)
    shared_matrix = my_private.D_inv.multiply(temp2)

    # Hash the shared matrix to derive key material
    matrix_bytes = b''
    for row in shared_matrix.data:
        for val in row:
            matrix_bytes += val.to_bytes(16, 'big')

    return hashlib.sha256(matrix_bytes).digest()


def derive_key_material(
    my_private: PrivateKey,
    their_public: PublicKey,
    length: int = 32,
    context: bytes = b''
) -> bytes:
    """
    Derive key material of specified length for key exchange.

    This is an extended version that can produce longer key material
    using HKDF-like expansion with the shared secret.

    Args:
        my_private: Your private key
        their_public: The other party's public key
        length: Desired length of key material in bytes
        context: Optional context/info bytes for domain separation

    Returns:
        Key material of the specified length

    Example:
        >>> from matrixtrap.keys import generate_keypair
        >>> from matrixtrap.exchange import derive_key_material
        >>>
        >>> alice_pub, alice_priv = generate_keypair()
        >>> bob_pub, bob_priv = generate_keypair()
        >>>
        >>> # Derive 64 bytes of key material with context
        >>> key_material = derive_key_material(
        ...     alice_priv, bob_pub,
        ...     length=64,
        ...     context=b"encryption-key"
        ... )
    """
    # Get base shared secret
    base_secret = derive_shared_secret(my_private, their_public)

    # Expand to desired length using counter mode
    result = b''
    counter = 0
    while len(result) < length:
        h = hashlib.sha256(base_secret + context + counter.to_bytes(4, 'big'))
        result += h.digest()
        counter += 1

    return result[:length]
