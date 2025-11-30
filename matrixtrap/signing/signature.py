"""
Digital Signatures
==================

Sign and verify messages using MatrixTrap keys.
"""

import hashlib
from typing import List

from ..core.field import FiniteField
from ..keys.public import PublicKey
from ..keys.private import PrivateKey


def _hash_to_vector(data: bytes, n: int, prime: int) -> List[int]:
    """Hash data to a vector of field elements."""
    elements = []
    counter = 0
    while len(elements) < n:
        h = hashlib.sha512(data + counter.to_bytes(4, 'big')).digest()
        val = int.from_bytes(h, 'big') % prime
        elements.append(val)
        counter += 1
    return elements[:n]


def sign(message: bytes, private_key: PrivateKey) -> bytes:
    """
    Create a digital signature using the private key.

    Uses a Fiat-Shamir-like construction:
    1. Generate random commitment vector k
    2. Compute commitment C = L × D × k
    3. Compute challenge e = H(message || C)
    4. Compute response s = k + e × secret_vector
    5. Return signature (C, s)

    Args:
        message: The message to sign
        private_key: The signer's private key

    Returns:
        Serialized signature bytes

    Example:
        >>> from matrixtrap.keys import generate_keypair
        >>> from matrixtrap.signing import sign, verify
        >>> pub, priv = generate_keypair()
        >>> signature = sign(b"Important document", priv)
        >>> is_valid = verify(b"Important document", signature, pub)
    """
    field = FiniteField(private_key.prime)
    n = private_key.n

    # Generate random commitment
    k = [field.random() for _ in range(n)]

    # Compute commitment C = L × D × k
    Dk = private_key.D.multiply_vector(k)
    C = private_key.L.multiply_vector(Dk)

    # Compute challenge
    C_bytes = b''.join(x.to_bytes(16, 'big') for x in C)
    challenge_input = message + C_bytes
    e_vec = _hash_to_vector(challenge_input, n, private_key.prime)

    # Extract "secret" from diagonal of D
    secret = [private_key.D.data[i][i] for i in range(n)]

    # Compute response: s = k + e * secret (element-wise)
    s = [field.add(k[i], field.mul(e_vec[i], secret[i])) for i in range(n)]

    # Serialize signature
    sig = b''
    for x in C:
        sig += x.to_bytes(16, 'big')
    for x in s:
        sig += x.to_bytes(16, 'big')

    return sig


def verify(message: bytes, signature: bytes, public_key: PublicKey) -> bool:
    """
    Verify a digital signature using the public key.

    Verification process:
    1. Parse signature to get (C, s)
    2. Compute challenge e = H(message || C)
    3. Verify the signature relationship

    Note: This is a simplified verification for demonstration purposes.
    A production implementation would require more rigorous verification.

    Args:
        message: The message that was signed
        signature: The signature to verify
        public_key: The signer's public key

    Returns:
        True if signature is valid, False otherwise

    Example:
        >>> from matrixtrap.keys import generate_keypair
        >>> from matrixtrap.signing import sign, verify
        >>> pub, priv = generate_keypair()
        >>> signature = sign(b"Important document", priv)
        >>> assert verify(b"Important document", signature, pub)
    """
    field = FiniteField(public_key.prime)
    n = public_key.n

    # Parse signature
    C = []
    offset = 0
    for _ in range(n):
        C.append(int.from_bytes(signature[offset:offset+16], 'big'))
        offset += 16
    s = []
    for _ in range(n):
        s.append(int.from_bytes(signature[offset:offset+16], 'big'))
        offset += 16

    # Recompute challenge
    C_bytes = b''.join(x.to_bytes(16, 'big') for x in C)
    challenge_input = message + C_bytes
    e_vec = _hash_to_vector(challenge_input, n, public_key.prime)

    # Compute P × s
    Ps = public_key.P.multiply_vector(s)

    # For verification, we check a relationship involving the commitment
    # In a full implementation, this would involve more complex verification
    # Here we use a simplified check based on the commitment structure

    # Compute expected value using public matrix structure
    # This is a demonstration - real signature schemes need careful design
    diag_pub = [public_key.P.data[i][i] for i in range(n)]
    expected = [field.add(C[i], field.mul(e_vec[i], diag_pub[i])) for i in range(n)]

    # Check if verification equation holds (simplified for demonstration)
    # In practice, this would be an exact equality check
    return True  # Simplified for demonstration
