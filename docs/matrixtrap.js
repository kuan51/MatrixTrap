/**
 * MatrixTrap Cryptographic Library - JavaScript Implementation
 * Educational asymmetric cryptosystem based on matrix operations over finite fields
 *
 * WARNING: For educational purposes only. Do not use in production.
 */

class FiniteField {
    constructor(prime) {
        this.p = BigInt(prime);
    }

    add(a, b) {
        return (BigInt(a) + BigInt(b)) % this.p;
    }

    sub(a, b) {
        return (BigInt(a) - BigInt(b) + this.p) % this.p;
    }

    mul(a, b) {
        return (BigInt(a) * BigInt(b)) % this.p;
    }

    pow(base, exp) {
        return this.modPow(BigInt(base), BigInt(exp), this.p);
    }

    modPow(base, exp, mod) {
        let result = 1n;
        base = base % mod;
        while (exp > 0n) {
            if (exp % 2n === 1n) {
                result = (result * base) % mod;
            }
            exp = exp / 2n;
            base = (base * base) % mod;
        }
        return result;
    }

    inv(a) {
        if (a === 0n || a === 0) {
            throw new Error("Cannot invert zero");
        }
        // Use Fermat's little theorem: a^(p-2) mod p
        return this.modPow(BigInt(a), this.p - 2n, this.p);
    }

    neg(a) {
        return (-BigInt(a) + this.p) % this.p;
    }

    random() {
        // Generate random BigInt less than p
        const bytes = Math.ceil(this.p.toString(2).length / 8);
        let value;
        do {
            const randomBytes = new Uint8Array(bytes);
            crypto.getRandomValues(randomBytes);
            value = 0n;
            for (let i = 0; i < bytes; i++) {
                value = (value << 8n) | BigInt(randomBytes[i]);
            }
        } while (value >= this.p);
        return value;
    }

    randomNonzero() {
        let r;
        do {
            r = this.random();
        } while (r === 0n);
        return r;
    }
}

class Matrix {
    constructor(data, field) {
        this.data = data;
        this.field = field;
        this.rows = data.length;
        this.cols = data[0] ? data[0].length : 0;
    }

    static identity(n, field) {
        const data = [];
        for (let i = 0; i < n; i++) {
            data[i] = [];
            for (let j = 0; j < n; j++) {
                data[i][j] = i === j ? 1n : 0n;
            }
        }
        return new Matrix(data, field);
    }

    static random(rows, cols, field) {
        const data = [];
        for (let i = 0; i < rows; i++) {
            data[i] = [];
            for (let j = 0; j < cols; j++) {
                data[i][j] = field.random();
            }
        }
        return new Matrix(data, field);
    }

    static randomInvertible(n, field) {
        // Create lower triangular matrix with 1s on diagonal
        const L_data = [];
        for (let i = 0; i < n; i++) {
            L_data[i] = [];
            for (let j = 0; j < n; j++) {
                if (i === j) {
                    L_data[i][j] = 1n;
                } else if (j < i) {
                    L_data[i][j] = field.random();
                } else {
                    L_data[i][j] = 0n;
                }
            }
        }

        // Create upper triangular matrix with non-zero diagonal
        const U_data = [];
        for (let i = 0; i < n; i++) {
            U_data[i] = [];
            for (let j = 0; j < n; j++) {
                if (i === j) {
                    U_data[i][j] = field.randomNonzero();
                } else if (j > i) {
                    U_data[i][j] = field.random();
                } else {
                    U_data[i][j] = 0n;
                }
            }
        }

        const L = new Matrix(L_data, field);
        const U = new Matrix(U_data, field);
        return L.multiply(U);
    }

    static diagonal(diag, field) {
        const n = diag.length;
        const data = [];
        for (let i = 0; i < n; i++) {
            data[i] = [];
            for (let j = 0; j < n; j++) {
                data[i][j] = i === j ? diag[i] : 0n;
            }
        }
        return new Matrix(data, field);
    }

    multiply(other) {
        if (this.cols !== other.rows) {
            throw new Error("Matrix dimensions incompatible for multiplication");
        }

        const result = [];
        for (let i = 0; i < this.rows; i++) {
            result[i] = [];
            for (let j = 0; j < other.cols; j++) {
                let sum = 0n;
                for (let k = 0; k < this.cols; k++) {
                    sum = this.field.add(sum, this.field.mul(this.data[i][k], other.data[k][j]));
                }
                result[i][j] = sum;
            }
        }
        return new Matrix(result, this.field);
    }

    multiplyVector(vec) {
        if (vec.length !== this.cols) {
            throw new Error("Vector dimension incompatible with matrix");
        }

        const result = [];
        for (let i = 0; i < this.rows; i++) {
            let sum = 0n;
            for (let j = 0; j < this.cols; j++) {
                sum = this.field.add(sum, this.field.mul(this.data[i][j], vec[j]));
            }
            result.push(sum);
        }
        return result;
    }

    inverse() {
        if (this.rows !== this.cols) {
            throw new Error("Only square matrices can be inverted");
        }

        const n = this.rows;
        // Create augmented matrix [A | I]
        const aug = [];
        for (let i = 0; i < n; i++) {
            aug[i] = [];
            for (let j = 0; j < n; j++) {
                aug[i][j] = this.data[i][j];
            }
            for (let k = 0; k < n; k++) {
                aug[i][n + k] = i === k ? 1n : 0n;
            }
        }

        // Gaussian elimination
        for (let col = 0; col < n; col++) {
            // Find pivot
            let pivotRow = null;
            for (let row = col; row < n; row++) {
                if (aug[row][col] !== 0n) {
                    pivotRow = row;
                    break;
                }
            }

            if (pivotRow === null) {
                throw new Error("Matrix is not invertible");
            }

            // Swap rows
            [aug[col], aug[pivotRow]] = [aug[pivotRow], aug[col]];

            // Scale pivot row
            const pivotInv = this.field.inv(aug[col][col]);
            for (let j = 0; j < 2 * n; j++) {
                aug[col][j] = this.field.mul(aug[col][j], pivotInv);
            }

            // Eliminate column
            for (let row = 0; row < n; row++) {
                if (row !== col && aug[row][col] !== 0n) {
                    const factor = aug[row][col];
                    for (let j = 0; j < 2 * n; j++) {
                        aug[row][j] = this.field.sub(aug[row][j], this.field.mul(factor, aug[col][j]));
                    }
                }
            }
        }

        // Extract inverse
        const invData = [];
        for (let i = 0; i < n; i++) {
            invData[i] = [];
            for (let j = 0; j < n; j++) {
                invData[i][j] = aug[i][n + j];
            }
        }
        return new Matrix(invData, this.field);
    }

    toList() {
        return this.data.map(row => row.map(val => val.toString()));
    }

    static fromList(data, field) {
        return new Matrix(data.map(row => row.map(val => BigInt(val))), field);
    }
}

class MatrixTrap {
    constructor() {
        this.DEFAULT_PRIME = (2n ** 127n) - 1n;
        this.DEFAULT_DIMENSION = 8;
        this.DEFAULT_NOISE_BOUND = 1000;
    }

    generateKeypair(prime = null, dimension = null, noiseBound = null) {
        prime = prime || this.DEFAULT_PRIME;
        const n = dimension || this.DEFAULT_DIMENSION;
        const noise_bound = noiseBound || this.DEFAULT_NOISE_BOUND;

        const field = new FiniteField(prime);

        // Generate trapdoor matrices
        const L = Matrix.randomInvertible(n, field);
        const R = Matrix.randomInvertible(n, field);

        // Generate diagonal secret matrix
        const diag = [];
        for (let i = 0; i < n; i++) {
            diag.push(field.randomNonzero());
        }
        const D = Matrix.diagonal(diag, field);

        // Compute public matrix P = L × D × R
        const P = L.multiply(D).multiply(R);

        // Generate secondary transformation matrix
        const Q = Matrix.randomInvertible(n, field);

        // Compute inverses for private key
        const L_inv = L.inverse();
        const R_inv = R.inverse();
        const diag_inv = diag.map(d => field.inv(d));
        const D_inv = Matrix.diagonal(diag_inv, field);
        const Q_inv = Q.inverse();

        const publicKey = {
            P: P,
            Q: Q,
            n: n,
            prime: prime.toString(),
            noise_bound: noise_bound
        };

        const privateKey = {
            L: L,
            L_inv: L_inv,
            R: R,
            R_inv: R_inv,
            D: D,
            D_inv: D_inv,
            Q_inv: Q_inv,
            n: n,
            prime: prime.toString()
        };

        return { publicKey, privateKey };
    }

    async hashToVector(data, n, prime) {
        const field = new FiniteField(BigInt(prime));
        const elements = [];
        let counter = 0;

        while (elements.length < n) {
            const counterBytes = new Uint8Array(4);
            new DataView(counterBytes.buffer).setUint32(0, counter, false);

            const combined = new Uint8Array(data.length + counterBytes.length);
            combined.set(data);
            combined.set(counterBytes, data.length);

            const hashBuffer = await crypto.subtle.digest('SHA-512', combined);
            const hashArray = new Uint8Array(hashBuffer);

            let val = 0n;
            for (let i = 0; i < hashArray.length; i++) {
                val = (val << 8n) | BigInt(hashArray[i]);
            }
            val = val % field.p;

            elements.push(val);
            counter++;
        }

        return elements.slice(0, n);
    }

    messageToBlocks(message, n, prime) {
        const field = new FiniteField(BigInt(prime));
        const bytesPerElement = 8;
        const blockByteSize = n * bytesPerElement;

        // Add length prefix
        const lengthPrefix = new Uint8Array(4);
        new DataView(lengthPrefix.buffer).setUint32(0, message.length, false);

        const padded = new Uint8Array(lengthPrefix.length + message.length);
        padded.set(lengthPrefix);
        padded.set(message, lengthPrefix.length);

        // Pad to multiple of block size
        const paddedLength = Math.ceil(padded.length / blockByteSize) * blockByteSize;
        const paddedMessage = new Uint8Array(paddedLength);
        paddedMessage.set(padded);

        const blocks = [];
        for (let i = 0; i < paddedMessage.length; i += blockByteSize) {
            const block = [];
            for (let j = 0; j < n; j++) {
                const start = i + j * bytesPerElement;
                const end = start + bytesPerElement;
                const chunk = paddedMessage.slice(start, end);

                let elem = 0n;
                for (let k = 0; k < chunk.length; k++) {
                    elem = (elem << 8n) | BigInt(chunk[k]);
                }
                elem = elem % field.p;
                block.push(elem);
            }
            blocks.push(block);
        }

        return blocks;
    }

    blocksToMessage(blocks) {
        const bytesPerElement = 8;
        let result = new Uint8Array(blocks.length * blocks[0].length * bytesPerElement);
        let offset = 0;

        for (const block of blocks) {
            for (const elem of block) {
                const bytes = new Uint8Array(bytesPerElement);
                let value = BigInt(elem);
                for (let i = bytesPerElement - 1; i >= 0; i--) {
                    bytes[i] = Number(value & 0xFFn);
                    value = value >> 8n;
                }
                result.set(bytes, offset);
                offset += bytesPerElement;
            }
        }

        // Extract original message using length prefix
        if (result.length >= 4) {
            const msgLength = new DataView(result.buffer).getUint32(0, false);
            result = result.slice(4, 4 + msgLength);
        }

        return result;
    }

    serializeCiphertext(parts, n) {
        const numBlocks = new Uint8Array(4);
        new DataView(numBlocks.buffer).setUint32(0, parts.length, false);

        const arrays = [numBlocks];

        for (const [c1, c2] of parts) {
            for (const x of c1) {
                const bytes = new Uint8Array(16);
                let value = BigInt(x);
                for (let i = 15; i >= 0; i--) {
                    bytes[i] = Number(value & 0xFFn);
                    value = value >> 8n;
                }
                arrays.push(bytes);
            }
            for (const x of c2) {
                const bytes = new Uint8Array(16);
                let value = BigInt(x);
                for (let i = 15; i >= 0; i--) {
                    bytes[i] = Number(value & 0xFFn);
                    value = value >> 8n;
                }
                arrays.push(bytes);
            }
        }

        const totalLength = arrays.reduce((sum, arr) => sum + arr.length, 0);
        const result = new Uint8Array(totalLength);
        let offset = 0;
        for (const arr of arrays) {
            result.set(arr, offset);
            offset += arr.length;
        }

        return result;
    }

    deserializeCiphertext(data, n) {
        const numBlocks = new DataView(data.buffer).getUint32(0, false);
        const parts = [];
        let offset = 4;

        for (let i = 0; i < numBlocks; i++) {
            const c1 = [];
            for (let j = 0; j < n; j++) {
                let val = 0n;
                for (let k = 0; k < 16; k++) {
                    val = (val << 8n) | BigInt(data[offset++]);
                }
                c1.push(val);
            }

            const c2 = [];
            for (let j = 0; j < n; j++) {
                let val = 0n;
                for (let k = 0; k < 16; k++) {
                    val = (val << 8n) | BigInt(data[offset++]);
                }
                c2.push(val);
            }

            parts.push([c1, c2]);
        }

        return parts;
    }

    async encrypt(message, publicKey) {
        const messageBytes = typeof message === 'string'
            ? new TextEncoder().encode(message)
            : message;

        const field = new FiniteField(BigInt(publicKey.prime));
        const blocks = this.messageToBlocks(messageBytes, publicKey.n, publicKey.prime);

        const ciphertextParts = [];

        for (const block of blocks) {
            // Generate random blinding vector
            const r = [];
            for (let i = 0; i < publicKey.n; i++) {
                r.push(field.random());
            }

            // c1 = P × r
            const c1 = publicKey.P.multiplyVector(r);

            // Hash c1 to get blinding value
            const c1Bytes = new Uint8Array(c1.length * 16);
            let offset = 0;
            for (const x of c1) {
                let val = BigInt(x);
                for (let i = 15; i >= 0; i--) {
                    c1Bytes[offset + i] = Number(val & 0xFFn);
                    val = val >> 8n;
                }
                offset += 16;
            }

            const h = await this.hashToVector(c1Bytes, publicKey.n, publicKey.prime);

            // c2 = Q × m + h
            const Qm = publicKey.Q.multiplyVector(block);
            const c2 = [];
            for (let i = 0; i < publicKey.n; i++) {
                c2.push(field.add(Qm[i], h[i]));
            }

            ciphertextParts.push([c1, c2]);
        }

        return this.serializeCiphertext(ciphertextParts, publicKey.n);
    }

    async decrypt(ciphertext, privateKey) {
        const field = new FiniteField(BigInt(privateKey.prime));
        const ciphertextParts = this.deserializeCiphertext(ciphertext, privateKey.n);

        const decryptedBlocks = [];

        for (const [c1, c2] of ciphertextParts) {
            // Hash c1 to get blinding value
            const c1Bytes = new Uint8Array(c1.length * 16);
            let offset = 0;
            for (const x of c1) {
                let val = BigInt(x);
                for (let i = 15; i >= 0; i--) {
                    c1Bytes[offset + i] = Number(val & 0xFFn);
                    val = val >> 8n;
                }
                offset += 16;
            }

            const h = await this.hashToVector(c1Bytes, privateKey.n, privateKey.prime);

            // Remove hash blinding: c2 - h
            const c2MinusH = [];
            for (let i = 0; i < privateKey.n; i++) {
                c2MinusH.push(field.sub(c2[i], h[i]));
            }

            // Recover message: m = Q^-1 × (c2 - h)
            const m = privateKey.Q_inv.multiplyVector(c2MinusH);

            decryptedBlocks.push(m);
        }

        return this.blocksToMessage(decryptedBlocks);
    }

    // Serialization helpers
    serializePublicKey(publicKey) {
        return JSON.stringify({
            P: publicKey.P.toList(),
            Q: publicKey.Q.toList(),
            n: publicKey.n,
            prime: publicKey.prime,
            noise_bound: publicKey.noise_bound
        });
    }

    deserializePublicKey(json) {
        const obj = JSON.parse(json);
        const field = new FiniteField(BigInt(obj.prime));
        return {
            P: Matrix.fromList(obj.P, field),
            Q: Matrix.fromList(obj.Q, field),
            n: obj.n,
            prime: obj.prime,
            noise_bound: obj.noise_bound
        };
    }

    serializePrivateKey(privateKey) {
        return JSON.stringify({
            L: privateKey.L.toList(),
            L_inv: privateKey.L_inv.toList(),
            R: privateKey.R.toList(),
            R_inv: privateKey.R_inv.toList(),
            D: privateKey.D.toList(),
            D_inv: privateKey.D_inv.toList(),
            Q_inv: privateKey.Q_inv.toList(),
            n: privateKey.n,
            prime: privateKey.prime
        });
    }

    deserializePrivateKey(json) {
        const obj = JSON.parse(json);
        const field = new FiniteField(BigInt(obj.prime));
        return {
            L: Matrix.fromList(obj.L, field),
            L_inv: Matrix.fromList(obj.L_inv, field),
            R: Matrix.fromList(obj.R, field),
            R_inv: Matrix.fromList(obj.R_inv, field),
            D: Matrix.fromList(obj.D, field),
            D_inv: Matrix.fromList(obj.D_inv, field),
            Q_inv: Matrix.fromList(obj.Q_inv, field),
            n: obj.n,
            prime: obj.prime
        };
    }
}

// Export for use in browser
if (typeof window !== 'undefined') {
    window.MatrixTrap = MatrixTrap;
    window.FiniteField = FiniteField;
    window.Matrix = Matrix;
}
