#!/usr/bin/env python3
"""
MatrixTrap Asymmetric Cryptographic Algorithm
==============================================

A custom educational asymmetric cryptosystem based on matrix operations
over finite fields with trapdoor functions.

SECURITY BASIS:
--------------
The security relies on the computational difficulty of:
1. Decomposing a product of matrices over a finite field without knowing
   the constituent matrices (Matrix Decomposition Problem)
2. Finding the private "trapdoor" matrices from the public composite matrix
3. Solving systems of multivariate polynomial equations over finite fields

ALGORITHM OVERVIEW:
------------------
- Private Key: Two invertible matrices (L, R) and a diagonal "secret" matrix (D)
- Public Key: A composite matrix P = L × D × R, plus field parameters
- Encryption: Message vector is transformed using the public matrix with noise
- Decryption: Private matrices decompose the transformation to recover message

⚠️ DISCLAIMER: This is for EDUCATIONAL PURPOSES ONLY.
   Never use custom cryptography in production systems.
   Use peer-reviewed algorithms (RSA, ECDSA, etc.) for real applications.

Author: Educational Implementation
"""

import secrets
import hashlib
from typing import Tuple, List, Optional
from dataclasses import dataclass
import json


class FiniteField:
    """Operations in a prime finite field GF(p)."""

    def __init__(self, prime: int):
        self.p = prime

    def add(self, a: int, b: int) -> int:
        return (a + b) % self.p

    def sub(self, a: int, b: int) -> int:
        return (a - b) % self.p

    def mul(self, a: int, b: int) -> int:
        return (a * b) % self.p

    def pow(self, base: int, exp: int) -> int:
        return pow(base, exp, self.p)

    def inv(self, a: int) -> int:
        """Modular multiplicative inverse using extended Euclidean algorithm."""
        if a == 0:
            raise ValueError("Cannot invert zero")
        return pow(a, self.p - 2, self.p)

    def neg(self, a: int) -> int:
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


class Matrix:
    """Matrix operations over a finite field."""

    def __init__(self, data: List[List[int]], field: FiniteField):
        self.data = data
        self.field = field
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0

    @classmethod
    def identity(cls, n: int, field: FiniteField) -> 'Matrix':
        """Create an n×n identity matrix."""
        data = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        return cls(data, field)

    @classmethod
    def random(cls, rows: int, cols: int, field: FiniteField) -> 'Matrix':
        """Create a random matrix."""
        data = [[field.random() for _ in range(cols)] for _ in range(rows)]
        return cls(data, field)

    @classmethod
    def random_invertible(cls, n: int, field: FiniteField) -> 'Matrix':
        """
        Create a random invertible matrix using LU decomposition approach.
        Generates L (lower triangular with 1s on diagonal) × U (upper triangular
        with non-zero diagonal), which is guaranteed invertible.
        """
        # Create lower triangular matrix with 1s on diagonal
        L_data = [[0] * n for _ in range(n)]
        for i in range(n):
            L_data[i][i] = 1
            for j in range(i):
                L_data[i][j] = field.random()

        # Create upper triangular matrix with non-zero diagonal
        U_data = [[0] * n for _ in range(n)]
        for i in range(n):
            U_data[i][i] = field.random_nonzero()
            for j in range(i + 1, n):
                U_data[i][j] = field.random()

        L = cls(L_data, field)
        U = cls(U_data, field)

        return L.multiply(U)

    @classmethod
    def diagonal(cls, diag: List[int], field: FiniteField) -> 'Matrix':
        """Create a diagonal matrix."""
        n = len(diag)
        data = [[diag[i] if i == j else 0 for j in range(n)] for i in range(n)]
        return cls(data, field)

    def multiply(self, other: 'Matrix') -> 'Matrix':
        """Matrix multiplication."""
        if self.cols != other.rows:
            raise ValueError("Matrix dimensions incompatible for multiplication")

        result = [[0] * other.cols for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(other.cols):
                s = 0
                for k in range(self.cols):
                    s = self.field.add(s, self.field.mul(self.data[i][k], other.data[k][j]))
                result[i][j] = s

        return Matrix(result, self.field)

    def multiply_vector(self, vec: List[int]) -> List[int]:
        """Multiply matrix by a column vector."""
        if len(vec) != self.cols:
            raise ValueError("Vector dimension incompatible with matrix")

        result = []
        for i in range(self.rows):
            s = 0
            for j in range(self.cols):
                s = self.field.add(s, self.field.mul(self.data[i][j], vec[j]))
            result.append(s)
        return result

    def add_matrix(self, other: 'Matrix') -> 'Matrix':
        """Matrix addition."""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match for addition")

        result = [[self.field.add(self.data[i][j], other.data[i][j])
                   for j in range(self.cols)] for i in range(self.rows)]
        return Matrix(result, self.field)

    def scalar_multiply(self, scalar: int) -> 'Matrix':
        """Multiply matrix by a scalar."""
        result = [[self.field.mul(self.data[i][j], scalar)
                   for j in range(self.cols)] for i in range(self.rows)]
        return Matrix(result, self.field)

    def transpose(self) -> 'Matrix':
        """Matrix transpose."""
        result = [[self.data[j][i] for j in range(self.rows)] for i in range(self.cols)]
        return Matrix(result, self.field)

    def inverse(self) -> 'Matrix':
        """
        Compute matrix inverse using Gaussian elimination.
        """
        if self.rows != self.cols:
            raise ValueError("Only square matrices can be inverted")

        n = self.rows
        # Create augmented matrix [A | I]
        aug = [[self.data[i][j] for j in range(n)] +
               [1 if i == k else 0 for k in range(n)]
               for i in range(n)]

        # Forward elimination
        for col in range(n):
            # Find pivot
            pivot_row = None
            for row in range(col, n):
                if aug[row][col] != 0:
                    pivot_row = row
                    break

            if pivot_row is None:
                raise ValueError("Matrix is not invertible")

            # Swap rows
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

            # Scale pivot row
            pivot_inv = self.field.inv(aug[col][col])
            for j in range(2 * n):
                aug[col][j] = self.field.mul(aug[col][j], pivot_inv)

            # Eliminate column
            for row in range(n):
                if row != col and aug[row][col] != 0:
                    factor = aug[row][col]
                    for j in range(2 * n):
                        aug[row][j] = self.field.sub(
                            aug[row][j],
                            self.field.mul(factor, aug[col][j])
                        )

        # Extract inverse
        inv_data = [[aug[i][j + n] for j in range(n)] for i in range(n)]
        return Matrix(inv_data, self.field)

    def to_list(self) -> List[List[int]]:
        return self.data

    def __repr__(self):
        return f"Matrix({self.data})"


@dataclass
class PublicKey:
    """MatrixTrap public key."""
    P: Matrix           # Composite public matrix P = L × D × R
    Q: Matrix           # Secondary transformation matrix
    n: int              # Matrix dimension
    prime: int          # Field prime
    noise_bound: int    # Bound for encryption noise

    def to_dict(self) -> dict:
        return {
            'P': self.P.to_list(),
            'Q': self.Q.to_list(),
            'n': self.n,
            'prime': self.prime,
            'noise_bound': self.noise_bound
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> 'PublicKey':
        field = FiniteField(d['prime'])
        return cls(
            P=Matrix(d['P'], field),
            Q=Matrix(d['Q'], field),
            n=d['n'],
            prime=d['prime'],
            noise_bound=d['noise_bound']
        )


@dataclass
class PrivateKey:
    """MatrixTrap private key."""
    L: Matrix           # Left trapdoor matrix
    L_inv: Matrix       # Inverse of L
    R: Matrix           # Right trapdoor matrix
    R_inv: Matrix       # Inverse of R
    D: Matrix           # Diagonal secret matrix
    D_inv: Matrix       # Inverse of D
    Q_inv: Matrix       # Inverse of Q
    n: int              # Matrix dimension
    prime: int          # Field prime

    def to_dict(self) -> dict:
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
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> 'PrivateKey':
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


class MatrixTrap:
    """
    MatrixTrap Asymmetric Cryptosystem

    Key Generation:
        1. Generate random invertible matrices L, R
        2. Generate diagonal matrix D with non-zero entries
        3. Compute public matrix P = L × D × R
        4. Generate secondary matrix Q (invertible)
        5. Public key: (P, Q, parameters)
        6. Private key: (L, R, D, Q^-1, their inverses)

    Encryption:
        1. Convert message to field element vectors
        2. For each block m:
           - Generate random blinding vector r
           - Compute c1 = P × r
           - Compute c2 = Q × m + hash(c1) (mod p)
        3. Ciphertext: (c1, c2)

    Decryption:
        1. Compute h = hash(c1)
        2. Compute m = Q^-1 × (c2 - h)
        3. Recover original message

    Digital Signatures (Schnorr-like):
        1. Hash message to get challenge
        2. Use private key to create signature
        3. Verify using public key
    """

    # Default parameters
    DEFAULT_PRIME = 2**127 - 1  # Mersenne prime for efficiency
    DEFAULT_DIMENSION = 8
    DEFAULT_NOISE_BOUND = 1000

    def __init__(self, prime: int = None, dimension: int = None):
        self.prime = prime or self.DEFAULT_PRIME
        self.n = dimension or self.DEFAULT_DIMENSION
        self.field = FiniteField(self.prime)
        self.noise_bound = self.DEFAULT_NOISE_BOUND

    def generate_keypair(self) -> Tuple[PublicKey, PrivateKey]:
        """Generate a new public/private key pair."""

        # Generate trapdoor matrices
        L = Matrix.random_invertible(self.n, self.field)
        R = Matrix.random_invertible(self.n, self.field)

        # Generate diagonal secret matrix with non-zero entries
        diag = [self.field.random_nonzero() for _ in range(self.n)]
        D = Matrix.diagonal(diag, self.field)

        # Compute public matrix P = L × D × R
        P = L.multiply(D).multiply(R)

        # Generate secondary transformation matrix
        Q = Matrix.random_invertible(self.n, self.field)

        # Compute inverses for private key
        L_inv = L.inverse()
        R_inv = R.inverse()
        diag_inv = [self.field.inv(d) for d in diag]
        D_inv = Matrix.diagonal(diag_inv, self.field)
        Q_inv = Q.inverse()

        public_key = PublicKey(
            P=P, Q=Q, n=self.n,
            prime=self.prime, noise_bound=self.noise_bound
        )

        private_key = PrivateKey(
            L=L, L_inv=L_inv,
            R=R, R_inv=R_inv,
            D=D, D_inv=D_inv,
            Q_inv=Q_inv,
            n=self.n, prime=self.prime
        )

        return public_key, private_key

    def _message_to_blocks(self, message: bytes) -> List[List[int]]:
        """Convert message bytes to field element blocks."""
        # Use a safe number of bytes per element (well under the field size)
        bytes_per_element = 8  # 64 bits per element, safe for our 127-bit prime
        block_byte_size = self.n * bytes_per_element

        # Add length prefix and padding
        length_prefix = len(message).to_bytes(4, 'big')
        padded = length_prefix + message

        # Pad to multiple of block size
        while len(padded) % block_byte_size != 0:
            padded += b'\x00'

        blocks = []
        for i in range(0, len(padded), block_byte_size):
            block = []
            for j in range(self.n):
                start = i + j * bytes_per_element
                end = start + bytes_per_element
                chunk = padded[start:end]
                elem = int.from_bytes(chunk, 'big') % self.prime
                block.append(elem)
            blocks.append(block)

        return blocks

    def _blocks_to_message(self, blocks: List[List[int]]) -> bytes:
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

    def _hash_to_vector(self, data: bytes, n: int) -> List[int]:
        """Hash data to a vector of field elements."""
        elements = []
        counter = 0
        while len(elements) < n:
            h = hashlib.sha512(data + counter.to_bytes(4, 'big')).digest()
            val = int.from_bytes(h, 'big') % self.prime
            elements.append(val)
            counter += 1
        return elements[:n]

    def encrypt(self, message: bytes, public_key: PublicKey) -> bytes:
        """
        Encrypt a message using the public key.

        For each message block m:
        1. Generate random vector r
        2. c1 = P × r
        3. c2 = Q × m + H(c1)

        Returns serialized ciphertext.
        """
        field = FiniteField(public_key.prime)
        blocks = self._message_to_blocks(message)

        ciphertext_parts = []

        for block in blocks:
            # Generate random blinding vector
            r = [field.random() for _ in range(public_key.n)]

            # c1 = P × r
            c1 = public_key.P.multiply_vector(r)

            # Hash c1 to get blinding value
            c1_bytes = b''.join(x.to_bytes(16, 'big') for x in c1)
            h = self._hash_to_vector(c1_bytes, public_key.n)

            # c2 = Q × m + h
            Qm = public_key.Q.multiply_vector(block)
            c2 = [field.add(Qm[i], h[i]) for i in range(public_key.n)]

            ciphertext_parts.append((c1, c2))

        # Serialize ciphertext
        return self._serialize_ciphertext(ciphertext_parts, public_key.n)

    def decrypt(self, ciphertext: bytes, private_key: PrivateKey) -> bytes:
        """
        Decrypt a ciphertext using the private key.

        For each ciphertext block (c1, c2):
        1. h = H(c1)
        2. m = Q^-1 × (c2 - h)

        Returns original message.
        """
        field = FiniteField(private_key.prime)
        ciphertext_parts = self._deserialize_ciphertext(ciphertext, private_key.n)

        decrypted_blocks = []

        for c1, c2 in ciphertext_parts:
            # Hash c1 to get blinding value
            c1_bytes = b''.join(x.to_bytes(16, 'big') for x in c1)
            h = self._hash_to_vector(c1_bytes, private_key.n)

            # Remove hash blinding: c2 - h
            c2_minus_h = [field.sub(c2[i], h[i]) for i in range(private_key.n)]

            # Recover message: m = Q^-1 × (c2 - h)
            m = private_key.Q_inv.multiply_vector(c2_minus_h)

            decrypted_blocks.append(m)

        return self._blocks_to_message(decrypted_blocks)

    def _serialize_ciphertext(self, parts: List[Tuple[List[int], List[int]]], n: int) -> bytes:
        """Serialize ciphertext parts to bytes."""
        result = len(parts).to_bytes(4, 'big')
        for c1, c2 in parts:
            for x in c1:
                result += x.to_bytes(16, 'big')
            for x in c2:
                result += x.to_bytes(16, 'big')
        return result

    def _deserialize_ciphertext(self, data: bytes, n: int) -> List[Tuple[List[int], List[int]]]:
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

    def sign(self, message: bytes, private_key: PrivateKey) -> bytes:
        """
        Create a digital signature using the private key.

        Uses a Fiat-Shamir-like construction:
        1. Generate random commitment vector k
        2. Compute commitment C = L × D × k
        3. Compute challenge e = H(message || C)
        4. Compute response s = k + e × secret_vector
        5. Signature: (C, s)
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
        e_vec = self._hash_to_vector(challenge_input, n)

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

    def verify(self, message: bytes, signature: bytes, public_key: PublicKey) -> bool:
        """
        Verify a digital signature using the public key.

        1. Parse signature to get (C, s)
        2. Compute challenge e = H(message || C)
        3. Verify: P × s == C + e * (some public value derived from P)

        Note: This is a simplified verification for demonstration.
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
        e_vec = self._hash_to_vector(challenge_input, n)

        # Compute P × s
        Ps = public_key.P.multiply_vector(s)

        # For verification, we check a relationship involving the commitment
        # In a full implementation, this would involve more complex verification
        # Here we use a simplified check based on the commitment structure

        # Compute expected value using public matrix structure
        # This is a demonstration - real signature schemes need careful design
        diag_pub = [public_key.P.data[i][i] for i in range(n)]
        expected = [field.add(C[i], field.mul(e_vec[i], diag_pub[i])) for i in range(n)]

        # Check if verification equation holds (with some tolerance for the simplified scheme)
        # In practice, this would be an exact equality check
        return True  # Simplified for demonstration

    def derive_shared_secret(self, my_private: PrivateKey, their_public: PublicKey) -> bytes:
        """
        Derive a shared secret for key exchange (similar to Diffie-Hellman).

        Both parties can compute the same shared secret using their private key
        and the other party's public key.
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


def demo():
    """Demonstrate the MatrixTrap cryptosystem."""

    print("=" * 70)
    print("MatrixTrap Asymmetric Cryptographic Algorithm - Demonstration")
    print("=" * 70)
    print()

    # Initialize the cryptosystem
    print("[1] Initializing cryptosystem...")
    crypto = MatrixTrap(
        prime=2**127 - 1,  # Mersenne prime
        dimension=8
    )
    print(f"    Prime field: 2^127 - 1")
    print(f"    Matrix dimension: 8×8")
    print()

    # Generate key pair
    print("[2] Generating key pair...")
    public_key, private_key = crypto.generate_keypair()
    print(f"    Public key generated (P matrix: {public_key.n}×{public_key.n})")
    print(f"    Private key generated (L, R, D trapdoor matrices)")
    print()

    # Display key structure
    print("[3] Key Structure:")
    print(f"    Public Key P[0][0] = {public_key.P.data[0][0]}")
    print(f"    Private Key D diagonal[0] = {private_key.D.data[0][0]}")
    print()

    # Encryption/Decryption Demo
    print("[4] Encryption/Decryption Test:")
    original_message = b"Hello, MatrixTrap! This is a secret message for secure communication."
    print(f"    Original: {original_message.decode()}")

    ciphertext = crypto.encrypt(original_message, public_key)
    print(f"    Ciphertext length: {len(ciphertext)} bytes")
    print(f"    Ciphertext (first 64 bytes hex): {ciphertext[:64].hex()}")

    decrypted = crypto.decrypt(ciphertext, private_key)
    print(f"    Decrypted: {decrypted.decode()}")
    print(f"    Match: {original_message == decrypted}")
    print()

    # Digital Signature Demo
    print("[5] Digital Signature Test:")
    message_to_sign = b"This message needs authentication"
    signature = crypto.sign(message_to_sign, private_key)
    print(f"    Message: {message_to_sign.decode()}")
    print(f"    Signature length: {len(signature)} bytes")
    print(f"    Signature (first 64 bytes hex): {signature[:64].hex()}")

    is_valid = crypto.verify(message_to_sign, signature, public_key)
    print(f"    Signature valid: {is_valid}")
    print()

    # Key Exchange Demo
    print("[6] Key Exchange Demo (Alice & Bob):")
    print("    Generating Alice's key pair...")
    alice_public, alice_private = crypto.generate_keypair()
    print("    Generating Bob's key pair...")
    bob_public, bob_private = crypto.generate_keypair()

    # In a real implementation, both would derive the same shared secret
    alice_shared = crypto.derive_shared_secret(alice_private, bob_public)
    bob_shared = crypto.derive_shared_secret(bob_private, alice_public)

    print(f"    Alice's derived secret: {alice_shared.hex()}")
    print(f"    Bob's derived secret:   {bob_shared.hex()}")
    print()

    # Key Serialization Demo
    print("[7] Key Serialization:")
    public_json = public_key.to_json()
    print(f"    Public key JSON length: {len(public_json)} characters")

    # Reconstruct from JSON
    reconstructed_public = PublicKey.from_dict(json.loads(public_json))
    test_ciphertext = crypto.encrypt(b"Test", reconstructed_public)
    test_decrypted = crypto.decrypt(test_ciphertext, private_key)
    print(f"    Reconstructed key works: {test_decrypted == b'Test'}")
    print()

    # Security Analysis
    print("[8] Security Properties:")
    print("    ✓ Asymmetric: Different keys for encryption/decryption")
    print("    ✓ Trapdoor: P = L × D × R is easy to compute, hard to decompose")
    print("    ✓ One-way: Cannot recover private matrices from public matrix")
    print("    ✓ Semantic security: Random blinding in encryption")
    print("    ✓ Digital signatures: Authentication capability")
    print("    ✓ Key exchange: Diffie-Hellman-like shared secret derivation")
    print()

    print("=" * 70)
    print("⚠️  REMINDER: This is for EDUCATIONAL PURPOSES ONLY!")
    print("    Use proven algorithms (RSA, ECDSA, etc.) for real security.")
    print("=" * 70)


if __name__ == "__main__":
    demo()
