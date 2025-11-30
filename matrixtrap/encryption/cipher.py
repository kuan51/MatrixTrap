"""
Encryption and Decryption
=========================

Encrypt and decrypt messages using MatrixTrap keys.
"""

import hashlib
from typing import List, Tuple

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


def _message_to_blocks(message: bytes, n: int, prime: int) -> List[List[int]]:
    """Convert message bytes to field element blocks."""
    # Use a safe number of bytes per element (well under the field size)
    bytes_per_element = 8  # 64 bits per element, safe for our 127-bit prime
    block_byte_size = n * bytes_per_element

    # Add length prefix and padding
    length_prefix = len(message).to_bytes(4, 'big')
    padded = length_prefix + message

    # Pad to multiple of block size
    while len(padded) % block_byte_size != 0:
        padded += b'\x00'

    blocks = []
    for i in range(0, len(padded), block_byte_size):
        block = []
        for j in range(n):
            start = i + j * bytes_per_element
            end = start + bytes_per_element
            chunk = padded[start:end]
            elem = int.from_bytes(chunk, 'big') % prime
            block.append(elem)
        blocks.append(block)

    return blocks


def _blocks_to_message(blocks: List[List[int]]) -> bytes:
    """Convert field element blocks back to message bytes."""
    bytes_per_element = 8  # Must match _message_to_blocks

    result = b''
    for block in blocks:
        for elem in block:
            result += elem.to_bytes(bytes_per_element, 'big')

    # Extract original message using length prefix
    if len(result) >= 4:
        msg_length = int.from_bytes(result[:4], 'big')
        result = result[4:4 + msg_length]

    return result


def _serialize_ciphertext(parts: List[Tuple[List[int], List[int]]], n: int) -> bytes:
    """Serialize ciphertext parts to bytes."""
    result = len(parts).to_bytes(4, 'big')
    for c1, c2 in parts:
        for x in c1:
            result += x.to_bytes(16, 'big')
        for x in c2:
            result += x.to_bytes(16, 'big')
    return result


def _deserialize_ciphertext(data: bytes, n: int) -> List[Tuple[List[int], List[int]]]:
    """Deserialize ciphertext bytes to parts."""
    num_blocks = int.from_bytes(data[:4], 'big')
    parts = []
    offset = 4

    for _ in range(num_blocks):
        c1 = []
        for _ in range(n):
            c1.append(int.from_bytes(data[offset:offset+16], 'big'))
            offset += 16
        c2 = []
        for _ in range(n):
            c2.append(int.from_bytes(data[offset:offset+16], 'big'))
            offset += 16
        parts.append((c1, c2))

    return parts


def encrypt(message: bytes, public_key: PublicKey) -> bytes:
    """
    Encrypt a message using the public key.

    Encryption process for each message block m:
    1. Generate random blinding vector r
    2. Compute c1 = P × r
    3. Compute c2 = Q × m + H(c1)

    Args:
        message: The plaintext message to encrypt
        public_key: The recipient's public key

    Returns:
        Serialized ciphertext bytes

    Example:
        >>> from matrixtrap.keys import generate_keypair
        >>> from matrixtrap.encryption import encrypt, decrypt
        >>> pub, priv = generate_keypair()
        >>> ciphertext = encrypt(b"Hello, World!", pub)
        >>> plaintext = decrypt(ciphertext, priv)
    """
    field = FiniteField(public_key.prime)
    blocks = _message_to_blocks(message, public_key.n, public_key.prime)

    ciphertext_parts = []

    for block in blocks:
        # Generate random blinding vector
        r = [field.random() for _ in range(public_key.n)]

        # c1 = P × r
        c1 = public_key.P.multiply_vector(r)

        # Hash c1 to get blinding value
        c1_bytes = b''.join(x.to_bytes(16, 'big') for x in c1)
        h = _hash_to_vector(c1_bytes, public_key.n, public_key.prime)

        # c2 = Q × m + h
        Qm = public_key.Q.multiply_vector(block)
        c2 = [field.add(Qm[i], h[i]) for i in range(public_key.n)]

        ciphertext_parts.append((c1, c2))

    # Serialize ciphertext
    return _serialize_ciphertext(ciphertext_parts, public_key.n)


def decrypt(ciphertext: bytes, private_key: PrivateKey) -> bytes:
    """
    Decrypt a ciphertext using the private key.

    Decryption process for each ciphertext block (c1, c2):
    1. Compute h = H(c1)
    2. Compute m = Q^-1 × (c2 - h)

    Args:
        ciphertext: The ciphertext to decrypt
        private_key: The recipient's private key

    Returns:
        Decrypted plaintext message

    Example:
        >>> from matrixtrap.keys import generate_keypair
        >>> from matrixtrap.encryption import encrypt, decrypt
        >>> pub, priv = generate_keypair()
        >>> ciphertext = encrypt(b"Hello, World!", pub)
        >>> plaintext = decrypt(ciphertext, priv)
        >>> assert plaintext == b"Hello, World!"
    """
    field = FiniteField(private_key.prime)
    ciphertext_parts = _deserialize_ciphertext(ciphertext, private_key.n)

    decrypted_blocks = []

    for c1, c2 in ciphertext_parts:
        # Hash c1 to get blinding value
        c1_bytes = b''.join(x.to_bytes(16, 'big') for x in c1)
        h = _hash_to_vector(c1_bytes, private_key.n, private_key.prime)

        # Remove hash blinding: c2 - h
        c2_minus_h = [field.sub(c2[i], h[i]) for i in range(private_key.n)]

        # Recover message: m = Q^-1 × (c2 - h)
        m = private_key.Q_inv.multiply_vector(c2_minus_h)

        decrypted_blocks.append(m)

    return _blocks_to_message(decrypted_blocks)
