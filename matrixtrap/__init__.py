"""
MatrixTrap Asymmetric Cryptographic Library
===========================================

A custom educational asymmetric cryptosystem based on matrix operations
over finite fields with trapdoor functions.

SECURITY BASIS:
--------------
The security relies on the computational difficulty of:
1. Decomposing a product of matrices over a finite field without knowing
   the constituent matrices (Matrix Decomposition Problem)
2. Finding the private "trapdoor" matrices from the public composite matrix
3. Solving systems of multivariate polynomial equations over finite fields

MODULES:
--------
- matrixtrap.keys: Key generation and management
- matrixtrap.encryption: Encrypt and decrypt messages
- matrixtrap.signing: Create and verify digital signatures
- matrixtrap.exchange: Derive shared secrets (key exchange)
- matrixtrap.core: Low-level mathematical primitives

QUICK START:
-----------
    # Generate keys
    from matrixtrap.keys import generate_keypair
    public_key, private_key = generate_keypair()

    # Encrypt/Decrypt
    from matrixtrap.encryption import encrypt, decrypt
    ciphertext = encrypt(b"Hello, World!", public_key)
    plaintext = decrypt(ciphertext, private_key)

    # Sign/Verify
    from matrixtrap.signing import sign, verify
    signature = sign(b"Important message", private_key)
    is_valid = verify(b"Important message", signature, public_key)

    # Key Exchange
    from matrixtrap.exchange import derive_shared_secret
    shared_secret = derive_shared_secret(my_private_key, their_public_key)

DISCLAIMER:
----------
This is for EDUCATIONAL PURPOSES ONLY.
Never use custom cryptography in production systems.
Use peer-reviewed algorithms (RSA, ECDSA, etc.) for real applications.
"""

__version__ = '0.1.0'
__author__ = 'Educational Implementation'

# Convenient imports for common use cases
from .keys import generate_keypair, PublicKey, PrivateKey
from .encryption import encrypt, decrypt
from .signing import sign, verify
from .exchange import derive_shared_secret, derive_key_material

__all__ = [
    # Version info
    '__version__',
    '__author__',
    # Key management
    'generate_keypair',
    'PublicKey',
    'PrivateKey',
    # Encryption
    'encrypt',
    'decrypt',
    # Signing
    'sign',
    'verify',
    # Key exchange
    'derive_shared_secret',
    'derive_key_material',
]
