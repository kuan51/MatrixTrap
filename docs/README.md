# MatrixTrap Web Application

This is a web-based encryption and decryption tool using the MatrixTrap cryptographic algorithm.

## Usage

Visit the GitHub Pages site to use the application:
https://kuan51.github.io/MatrixTrap/

## Features

- **Key Generation**: Generate new public/private keypairs
- **Message Encryption**: Encrypt messages using a public key
- **Message Decryption**: Decrypt messages using a private key
- **Key Management**: Save, load, and share keys
- **Educational**: Learn about asymmetric cryptography

## How It Works

1. Click "Generate New Keypair" to create encryption keys
2. Share your public key with others (safe to distribute)
3. Keep your private key secret (needed for decryption)
4. Enter a message and click "Encrypt Message"
5. Share the encrypted message (Base64 encoded)
6. To decrypt, paste the encrypted message and click "Decrypt Message"

## Security Warning

⚠️ **This is for educational purposes only!**

MatrixTrap is a custom cryptographic algorithm designed for learning. Never use custom cryptography in production systems. Use established, peer-reviewed algorithms like RSA, ECDSA, or AES for real security needs.

## Files

- `index.html` - Main application interface
- `matrixtrap.js` - JavaScript implementation of the MatrixTrap algorithm
- `app.js` - Application logic and UI handling
- `styles.css` - Styling for the web application

## Technology

- Pure JavaScript (ES6+)
- Web Crypto API for random number generation
- BigInt for large number arithmetic
- Responsive design with CSS Grid and Flexbox

## License

See the main repository LICENSE file.
