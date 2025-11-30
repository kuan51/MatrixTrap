/**
 * MatrixTrap Web Application
 * Main application logic for the encryption/decryption interface
 */

class MatrixTrapApp {
    constructor() {
        this.matrixTrap = new MatrixTrap();
        this.publicKey = null;
        this.privateKey = null;
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Key management
        document.getElementById('generateKeysBtn').addEventListener('click', () => this.generateKeys());
        document.getElementById('copyPublicKeyBtn').addEventListener('click', () => this.copyPublicKey());
        document.getElementById('copyPrivateKeyBtn').addEventListener('click', () => this.copyPrivateKey());
        document.getElementById('loadPublicKeyBtn').addEventListener('click', () => this.loadPublicKey());
        document.getElementById('loadPrivateKeyBtn').addEventListener('click', () => this.loadPrivateKey());
        document.getElementById('downloadKeysBtn').addEventListener('click', () => this.downloadKeys());

        // Encryption
        document.getElementById('encryptBtn').addEventListener('click', () => this.encryptMessage());
        document.getElementById('clearPlaintextBtn').addEventListener('click', () => {
            document.getElementById('plaintextInput').value = '';
            document.getElementById('ciphertextOutput').value = '';
            this.hideStatus('encryptStatus');
        });
        document.getElementById('copyCiphertextBtn').addEventListener('click', () => this.copyCiphertext());

        // Decryption
        document.getElementById('decryptBtn').addEventListener('click', () => this.decryptMessage());
        document.getElementById('clearCiphertextBtn').addEventListener('click', () => {
            document.getElementById('ciphertextInput').value = '';
            document.getElementById('plaintextOutput').value = '';
            this.hideStatus('decryptStatus');
        });
        document.getElementById('copyPlaintextBtn').addEventListener('click', () => this.copyPlaintext());

        // Quick transfer
        document.getElementById('transferBtn').addEventListener('click', () => this.transferCiphertext());
    }

    async generateKeys() {
        const btn = document.getElementById('generateKeysBtn');
        const originalText = btn.textContent;

        try {
            btn.disabled = true;
            btn.innerHTML = 'Generating Keys<span class="loading"></span>';

            // Add small delay to show loading state
            await new Promise(resolve => setTimeout(resolve, 100));

            const { publicKey, privateKey } = this.matrixTrap.generateKeypair();
            this.publicKey = publicKey;
            this.privateKey = privateKey;

            // Display keys
            const publicKeyJson = this.matrixTrap.serializePublicKey(publicKey);
            const privateKeyJson = this.matrixTrap.serializePrivateKey(privateKey);

            document.getElementById('publicKeyDisplay').value = publicKeyJson;
            document.getElementById('privateKeyDisplay').value = privateKeyJson;

            this.showStatus('keyStatus', 'Keypair generated successfully!', 'success');
        } catch (error) {
            this.showStatus('keyStatus', `Error generating keys: ${error.message}`, 'error');
            console.error('Key generation error:', error);
        } finally {
            btn.disabled = false;
            btn.textContent = originalText;
        }
    }

    async copyPublicKey() {
        const publicKeyText = document.getElementById('publicKeyDisplay').value;
        if (!publicKeyText) {
            this.showStatus('keyStatus', 'No public key to copy. Generate keys first.', 'warning');
            return;
        }

        try {
            await navigator.clipboard.writeText(publicKeyText);
            this.showStatus('keyStatus', 'Public key copied to clipboard!', 'success');
        } catch (error) {
            this.showStatus('keyStatus', 'Failed to copy to clipboard', 'error');
        }
    }

    async copyPrivateKey() {
        const privateKeyText = document.getElementById('privateKeyDisplay').value;
        if (!privateKeyText) {
            this.showStatus('keyStatus', 'No private key to copy. Generate keys first.', 'warning');
            return;
        }

        try {
            await navigator.clipboard.writeText(privateKeyText);
            this.showStatus('keyStatus', 'Private key copied to clipboard! Keep it secret!', 'success');
        } catch (error) {
            this.showStatus('keyStatus', 'Failed to copy to clipboard', 'error');
        }
    }

    loadPublicKey() {
        const input = prompt('Paste the public key JSON:');
        if (!input) return;

        try {
            this.publicKey = this.matrixTrap.deserializePublicKey(input);
            document.getElementById('publicKeyDisplay').value = input;
            this.showStatus('keyStatus', 'Public key loaded successfully!', 'success');
        } catch (error) {
            this.showStatus('keyStatus', `Error loading public key: ${error.message}`, 'error');
            console.error('Public key load error:', error);
        }
    }

    loadPrivateKey() {
        const input = prompt('Paste the private key JSON:');
        if (!input) return;

        try {
            this.privateKey = this.matrixTrap.deserializePrivateKey(input);
            document.getElementById('privateKeyDisplay').value = input;
            this.showStatus('keyStatus', 'Private key loaded successfully!', 'success');
        } catch (error) {
            this.showStatus('keyStatus', `Error loading private key: ${error.message}`, 'error');
            console.error('Private key load error:', error);
        }
    }

    downloadKeys() {
        if (!this.publicKey || !this.privateKey) {
            this.showStatus('keyStatus', 'No keys to download. Generate keys first.', 'warning');
            return;
        }

        const publicKeyJson = this.matrixTrap.serializePublicKey(this.publicKey);
        const privateKeyJson = this.matrixTrap.serializePrivateKey(this.privateKey);

        const keysData = {
            publicKey: JSON.parse(publicKeyJson),
            privateKey: JSON.parse(privateKeyJson),
            generated: new Date().toISOString()
        };

        const blob = new Blob([JSON.stringify(keysData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `matrixtrap-keys-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        this.showStatus('keyStatus', 'Keys downloaded successfully!', 'success');
    }

    async encryptMessage() {
        const plaintext = document.getElementById('plaintextInput').value;

        if (!plaintext) {
            this.showStatus('encryptStatus', 'Please enter a message to encrypt.', 'warning');
            return;
        }

        if (!this.publicKey) {
            this.showStatus('encryptStatus', 'No public key available. Generate or load keys first.', 'error');
            return;
        }

        const btn = document.getElementById('encryptBtn');
        const originalText = btn.textContent;

        try {
            btn.disabled = true;
            btn.innerHTML = 'Encrypting<span class="loading"></span>';

            const ciphertext = await this.matrixTrap.encrypt(plaintext, this.publicKey);
            const base64Ciphertext = this.arrayBufferToBase64(ciphertext);

            document.getElementById('ciphertextOutput').value = base64Ciphertext;
            this.showStatus('encryptStatus', 'Message encrypted successfully!', 'success');
        } catch (error) {
            this.showStatus('encryptStatus', `Encryption error: ${error.message}`, 'error');
            console.error('Encryption error:', error);
        } finally {
            btn.disabled = false;
            btn.textContent = originalText;
        }
    }

    async decryptMessage() {
        const base64Ciphertext = document.getElementById('ciphertextInput').value;

        if (!base64Ciphertext) {
            this.showStatus('decryptStatus', 'Please enter an encrypted message to decrypt.', 'warning');
            return;
        }

        if (!this.privateKey) {
            this.showStatus('decryptStatus', 'No private key available. Generate or load keys first.', 'error');
            return;
        }

        const btn = document.getElementById('decryptBtn');
        const originalText = btn.textContent;

        try {
            btn.disabled = true;
            btn.innerHTML = 'Decrypting<span class="loading"></span>';

            const ciphertext = this.base64ToArrayBuffer(base64Ciphertext);
            const plaintext = await this.matrixTrap.decrypt(ciphertext, this.privateKey);
            const message = new TextDecoder().decode(plaintext);

            document.getElementById('plaintextOutput').value = message;
            this.showStatus('decryptStatus', 'Message decrypted successfully!', 'success');
        } catch (error) {
            this.showStatus('decryptStatus', `Decryption error: ${error.message}`, 'error');
            console.error('Decryption error:', error);
        } finally {
            btn.disabled = false;
            btn.textContent = originalText;
        }
    }

    async copyCiphertext() {
        const ciphertext = document.getElementById('ciphertextOutput').value;
        if (!ciphertext) {
            this.showStatus('encryptStatus', 'No ciphertext to copy.', 'warning');
            return;
        }

        try {
            await navigator.clipboard.writeText(ciphertext);
            this.showStatus('encryptStatus', 'Ciphertext copied to clipboard!', 'success');
        } catch (error) {
            this.showStatus('encryptStatus', 'Failed to copy to clipboard', 'error');
        }
    }

    async copyPlaintext() {
        const plaintext = document.getElementById('plaintextOutput').value;
        if (!plaintext) {
            this.showStatus('decryptStatus', 'No plaintext to copy.', 'warning');
            return;
        }

        try {
            await navigator.clipboard.writeText(plaintext);
            this.showStatus('decryptStatus', 'Plaintext copied to clipboard!', 'success');
        } catch (error) {
            this.showStatus('decryptStatus', 'Failed to copy to clipboard', 'error');
        }
    }

    transferCiphertext() {
        const ciphertext = document.getElementById('ciphertextOutput').value;
        if (!ciphertext) {
            alert('No encrypted message to transfer. Encrypt a message first.');
            return;
        }

        document.getElementById('ciphertextInput').value = ciphertext;

        // Scroll to decryption section
        document.getElementById('ciphertextInput').scrollIntoView({
            behavior: 'smooth',
            block: 'center'
        });

        // Show brief success message
        this.showStatus('decryptStatus', 'Encrypted message transferred! Click "Decrypt Message" to decrypt.', 'success');
        setTimeout(() => this.hideStatus('decryptStatus'), 3000);
    }

    // Utility functions
    arrayBufferToBase64(buffer) {
        let binary = '';
        const bytes = new Uint8Array(buffer);
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }

    base64ToArrayBuffer(base64) {
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }
        return bytes;
    }

    showStatus(elementId, message, type) {
        const statusElement = document.getElementById(elementId);
        statusElement.textContent = message;
        statusElement.className = `status-message ${type}`;
        statusElement.style.display = 'block';

        // Auto-hide success messages after 5 seconds
        if (type === 'success') {
            setTimeout(() => this.hideStatus(elementId), 5000);
        }
    }

    hideStatus(elementId) {
        const statusElement = document.getElementById(elementId);
        statusElement.style.display = 'none';
        statusElement.className = 'status-message';
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new MatrixTrapApp();
});
