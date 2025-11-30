#!/usr/bin/env python3
"""
MatrixTrap Demonstration
========================

Demonstrates the modular MatrixTrap cryptographic library.
"""

import json

# Import from the modular library
from matrixtrap import (
    generate_keypair,
    encrypt, decrypt,
    sign, verify,
    derive_shared_secret,
    PublicKey,
    __version__
)


def demo():
    """Demonstrate the MatrixTrap cryptosystem."""

    print("=" * 70)
    print("MatrixTrap Asymmetric Cryptographic Library - Demonstration")
    print(f"Version: {__version__}")
    print("=" * 70)
    print()

    # Key Generation
    print("[1] Key Generation")
    print("    from matrixtrap import generate_keypair")
    print("    public_key, private_key = generate_keypair()")
    print()
    public_key, private_key = generate_keypair()
    print(f"    Public key generated (P matrix: {public_key.n}x{public_key.n})")
    print(f"    Private key generated (L, R, D trapdoor matrices)")
    print()

    # Display key structure
    print("[2] Key Structure:")
    print(f"    Public Key P[0][0] = {public_key.P.data[0][0]}")
    print(f"    Private Key D diagonal[0] = {private_key.D.data[0][0]}")
    print()

    # Encryption/Decryption Demo
    print("[3] Encryption/Decryption")
    print("    from matrixtrap import encrypt, decrypt")
    print()
    original_message = b"Hello, MatrixTrap! This is a secret message for secure communication."
    print(f"    Original: {original_message.decode()}")

    ciphertext = encrypt(original_message, public_key)
    print(f"    Ciphertext length: {len(ciphertext)} bytes")
    print(f"    Ciphertext (first 64 bytes hex): {ciphertext[:64].hex()}")

    decrypted = decrypt(ciphertext, private_key)
    print(f"    Decrypted: {decrypted.decode()}")
    print(f"    Match: {original_message == decrypted}")
    print()

    # Digital Signature Demo
    print("[4] Digital Signatures")
    print("    from matrixtrap import sign, verify")
    print()
    message_to_sign = b"This message needs authentication"
    signature = sign(message_to_sign, private_key)
    print(f"    Message: {message_to_sign.decode()}")
    print(f"    Signature length: {len(signature)} bytes")
    print(f"    Signature (first 64 bytes hex): {signature[:64].hex()}")

    is_valid = verify(message_to_sign, signature, public_key)
    print(f"    Signature valid: {is_valid}")
    print()

    # Key Exchange Demo
    print("[5] Key Exchange (Alice & Bob)")
    print("    from matrixtrap import derive_shared_secret")
    print()
    print("    Generating Alice's key pair...")
    alice_public, alice_private = generate_keypair()
    print("    Generating Bob's key pair...")
    bob_public, bob_private = generate_keypair()

    alice_shared = derive_shared_secret(alice_private, bob_public)
    bob_shared = derive_shared_secret(bob_private, alice_public)

    print(f"    Alice's derived secret: {alice_shared.hex()}")
    print(f"    Bob's derived secret:   {bob_shared.hex()}")
    print()

    # Key Serialization Demo
    print("[6] Key Serialization")
    print("    public_json = public_key.to_json()")
    print("    restored = PublicKey.from_json(public_json)")
    print()
    public_json = public_key.to_json()
    print(f"    Public key JSON length: {len(public_json)} characters")

    # Reconstruct from JSON
    reconstructed_public = PublicKey.from_json(public_json)
    test_ciphertext = encrypt(b"Test", reconstructed_public)
    test_decrypted = decrypt(test_ciphertext, private_key)
    print(f"    Reconstructed key works: {test_decrypted == b'Test'}")
    print()

    # Module Structure
    print("[7] Library Structure:")
    print("    matrixtrap/")
    print("    ├── __init__.py      # Main exports")
    print("    ├── core/            # Mathematical primitives")
    print("    │   ├── field.py     # Finite field operations")
    print("    │   └── matrix.py    # Matrix operations")
    print("    ├── keys/            # Key management")
    print("    │   ├── public.py    # PublicKey class")
    print("    │   ├── private.py   # PrivateKey class")
    print("    │   └── generation.py# Key generation")
    print("    ├── encryption/      # Encrypt/decrypt")
    print("    │   └── cipher.py    # encrypt(), decrypt()")
    print("    ├── signing/         # Digital signatures")
    print("    │   └── signature.py # sign(), verify()")
    print("    └── exchange/        # Key exchange")
    print("        └── key_exchange.py # derive_shared_secret()")
    print()

    # Security Analysis
    print("[8] Security Properties:")
    print("    - Asymmetric: Different keys for encryption/decryption")
    print("    - Trapdoor: P = L x D x R is easy to compute, hard to decompose")
    print("    - One-way: Cannot recover private matrices from public matrix")
    print("    - Semantic security: Random blinding in encryption")
    print("    - Digital signatures: Authentication capability")
    print("    - Key exchange: Diffie-Hellman-like shared secret derivation")
    print()

    print("=" * 70)
    print("REMINDER: This is for EDUCATIONAL PURPOSES ONLY!")
    print("Use proven algorithms (RSA, ECDSA, etc.) for real security.")
    print("=" * 70)


if __name__ == "__main__":
    demo()
