# MatrixTrap

A custom educational asymmetric cryptosystem based on matrix operations over finite fields with trapdoor functions.

## ⚠️ Educational Purpose Only

**DISCLAIMER:** This is for educational purposes only. Never use custom cryptography in production systems. Use peer-reviewed algorithms (RSA, ECDSA, etc.) for real applications.

## Overview

MatrixTrap implements an asymmetric cryptographic scheme based on:
- Matrix operations over finite fields
- Trapdoor functions using matrix decomposition
- Security from the Matrix Decomposition Problem (hard to factor P = L × D × R without knowing the trapdoor)

## Features

- **Asymmetric Encryption/Decryption**: Encrypt with public key, decrypt with private key
- **Digital Signatures**: Sign messages with private key, verify with public key
- **Key Exchange**: Derive shared secrets between two parties (Diffie-Hellman-like)
- **Key Serialization**: Export/import keys in JSON format

## Installation

### From Source

Clone the repository and install:

```bash
git clone <repository-url>
cd MatrixTrap
python -m pip install -e .
```

Or import directly:

```python
import sys
sys.path.insert(0, '/path/to/MatrixTrap')
```

## Quick Start

```python
from matrixtrap import generate_keypair, encrypt, decrypt, sign, verify

# Generate a keypair
public_key, private_key = generate_keypair()

# Encrypt a message
message = b"Hello, MatrixTrap!"
ciphertext = encrypt(message, public_key)

# Decrypt the message
plaintext = decrypt(ciphertext, private_key)
print(plaintext)  # b"Hello, MatrixTrap!"

# Sign a message
signature = sign(b"Important message", private_key)

# Verify the signature
is_valid = verify(b"Important message", signature, public_key)
print(is_valid)  # True
```

## Usage Guide

### Key Generation

Generate a new keypair:

```python
from matrixtrap import generate_keypair

public_key, private_key = generate_keypair()
```

The public key can be safely shared, while the private key must be kept secret.

### Encryption and Decryption

Encrypt a message using the public key:

```python
from matrixtrap import encrypt, decrypt

message = b"Secret message"
ciphertext = encrypt(message, public_key)

# Only the holder of private_key can decrypt
plaintext = decrypt(ciphertext, private_key)
assert plaintext == message
```

Encryption uses semantic security with random blinding to prevent pattern analysis.

### Digital Signatures

Create and verify digital signatures:

```python
from matrixtrap import sign, verify

message = b"This is important"

# Sign with private key
signature = sign(message, private_key)

# Verify with public key
is_valid = verify(message, signature, public_key)
print(is_valid)  # True

# Verification fails if message is tampered
is_valid = verify(b"Tampered message", signature, public_key)
print(is_valid)  # False
```

### Key Exchange (Shared Secret Derivation)

Derive a shared secret between two parties:

```python
from matrixtrap import generate_keypair, derive_shared_secret

# Alice generates her keypair
alice_public, alice_private = generate_keypair()

# Bob generates his keypair
bob_public, bob_private = generate_keypair()

# Alice computes shared secret using Bob's public key
alice_shared = derive_shared_secret(alice_private, bob_public)

# Bob computes shared secret using Alice's public key
bob_shared = derive_shared_secret(bob_private, alice_public)

# Both derive the same shared secret
assert alice_shared == bob_shared
```

The shared secret can be used as a key for symmetric encryption.

### Key Serialization

Save and restore keys in JSON format:

```python
# Export to JSON
public_json = public_key.to_json()
private_json = private_key.to_json()

# Save to file
with open('public_key.json', 'w') as f:
    f.write(public_json)

# Restore from JSON
from matrixtrap import PublicKey, PrivateKey

with open('public_key.json', 'r') as f:
    restored_public = PublicKey.from_json(f.read())

# The restored key works identically
ciphertext = encrypt(b"Test", restored_public)
plaintext = decrypt(ciphertext, private_key)
assert plaintext == b"Test"
```

## API Reference

### Main Functions

#### `generate_keypair() -> (PublicKey, PrivateKey)`
Generate a new asymmetric keypair.

```python
public_key, private_key = generate_keypair()
```

#### `encrypt(message: bytes, public_key: PublicKey) -> bytes`
Encrypt a message using the public key.

**Parameters:**
- `message`: Bytes to encrypt
- `public_key`: PublicKey instance

**Returns:** Encrypted bytes

#### `decrypt(ciphertext: bytes, private_key: PrivateKey) -> bytes`
Decrypt a ciphertext using the private key.

**Parameters:**
- `ciphertext`: Bytes to decrypt
- `private_key`: PrivateKey instance

**Returns:** Decrypted message bytes

#### `sign(message: bytes, private_key: PrivateKey) -> bytes`
Create a digital signature for a message.

**Parameters:**
- `message`: Bytes to sign
- `private_key`: PrivateKey instance

**Returns:** Signature bytes

#### `verify(message: bytes, signature: bytes, public_key: PublicKey) -> bool`
Verify a digital signature.

**Parameters:**
- `message`: Original message bytes
- `signature`: Signature bytes
- `public_key`: PublicKey instance

**Returns:** True if signature is valid, False otherwise

#### `derive_shared_secret(private_key: PrivateKey, other_public_key: PublicKey) -> bytes`
Derive a shared secret for key exchange.

**Parameters:**
- `private_key`: Your PrivateKey instance
- `other_public_key`: The other party's PublicKey instance

**Returns:** Shared secret bytes (256 bytes)

### Key Classes

#### `PublicKey`
Represents a public key.

**Methods:**
- `to_json() -> str`: Export to JSON format
- `from_json(json_str: str) -> PublicKey`: Import from JSON format

#### `PrivateKey`
Represents a private key.

**Methods:**
- `to_json() -> str`: Export to JSON format
- `from_json(json_str: str) -> PrivateKey`: Import from JSON format

## Library Structure

```
matrixtrap/
├── __init__.py              # Main module exports
├── core/                    # Mathematical primitives
│   ├── field.py            # Finite field operations
│   └── matrix.py           # Matrix operations over finite fields
├── keys/                    # Key management
│   ├── generation.py       # Key pair generation
│   ├── public.py           # PublicKey class
│   └── private.py          # PrivateKey class
├── encryption/             # Encryption operations
│   └── cipher.py           # encrypt() and decrypt() functions
├── signing/                # Digital signature operations
│   └── signature.py        # sign() and verify() functions
└── exchange/               # Key exchange operations
    └── key_exchange.py     # derive_shared_secret() function
```

## Examples

### Example 1: Secure Communication

```python
from matrixtrap import generate_keypair, encrypt, decrypt

# Alice generates her keypair
alice_public, alice_private = generate_keypair()

# Bob gets Alice's public key and encrypts a message
bob_message = b"Hello Alice, this is Bob!"
ciphertext = encrypt(bob_message, alice_public)

# Alice receives and decrypts
plaintext = decrypt(ciphertext, alice_private)
print(plaintext)  # b"Hello Alice, this is Bob!"
```

### Example 2: Message Authentication

```python
from matrixtrap import generate_keypair, sign, verify

public_key, private_key = generate_keypair()

# Sign a message
document = b"This is the official contract"
signature = sign(document, private_key)

# Anyone with the public key can verify the signature
is_authentic = verify(document, signature, public_key)
print(is_authentic)  # True
```

### Example 3: Secure Channel Setup

```python
from matrixtrap import generate_keypair, derive_shared_secret

# Alice and Bob each generate keypairs
alice_pub, alice_priv = generate_keypair()
bob_pub, bob_priv = generate_keypair()

# They exchange public keys (these can be sent over insecure channels)

# Both derive the same shared secret
alice_secret = derive_shared_secret(alice_priv, bob_pub)
bob_secret = derive_shared_secret(bob_priv, alice_pub)

assert alice_secret == bob_secret

# They can now use this shared secret as a key for symmetric encryption
print(f"Shared secret: {alice_secret.hex()}")
```

## Security Properties

MatrixTrap provides the following security properties (for an educational system):

- **Asymmetry**: Different keys for encryption and decryption
- **Trapdoor Function**: Computing P = L × D × R is easy, decomposing P is hard
- **One-wayness**: Cannot recover private matrices from public matrix
- **Semantic Security**: Random blinding prevents ciphertext pattern analysis
- **Signature Authentication**: Verifies both authenticity and integrity
- **Key Exchange**: Derives shared secrets without transmitting them

## Running the Demo

Run the included demo to see all features in action:

```bash
python demo.py
```

This demonstrates:
- Key generation
- Encryption/decryption
- Digital signatures
- Key exchange
- Key serialization
- Library structure overview
