import SEAL from 'node-seal';

/**
 * CKKS Encryption Utility
 * Provides an interface to encrypt numerical data using the CKKS scheme.
 * As requested, prints the results to the console.
 */
export async function encryptPatientDataCKKS(data: Record<string, any>) {
    // Initialize SEAL
    const seal = await SEAL({
        locateFile: (path: string) => `/${path}`
    });

    // Create CKKS context parameters
    const schemeType = seal.SchemeType.ckks;
    const polyModulusDegree = 8192;
    const bitSizes = Int32Array.from([60, 40, 40, 60]);

    const parms = new seal.EncryptionParameters(schemeType);
    parms.setPolyModulusDegree(polyModulusDegree);
    parms.setCoeffModulus(
        seal.CoeffModulus.Create(polyModulusDegree, bitSizes)
    );

    const context = new seal.SEALContext(
        parms,
        true,
        seal.SecLevelType.tc128
    );

    if (!context.parametersSet()) {
        throw new Error('Could not set SEAL parameters');
    }

    // Key creation
    const keyGenerator = new seal.KeyGenerator(context);
    const publicKey = keyGenerator.createPublicKey();
    const encryptor = new seal.Encryptor(context, publicKey);
    const encoder = new seal.CKKSEncoder(context);
    const scale = Math.pow(2.0, 40);

    const encryptedResults: Record<string, string> = {};

    // Encrypt numeric values
    for (const [key, value] of Object.entries(data)) {
        if (typeof value === 'number' && value !== null) {
            const plainText = new seal.Plaintext();
            const cipherText = new seal.Ciphertext();

            encoder.encode(Float64Array.from([value]), scale, plainText);
            encryptor.encrypt(plainText, cipherText);

            // Save ciphertext to string (truncated for display)
            const base64 = cipherText.saveToBase64(seal.ComprModeType.none);
            encryptedResults[key] = base64.substring(0, 100) + '... (truncated)';
        } else {
            encryptedResults[key] = '[Skipped: Non-numeric]';
        }
    }

    console.group('%c--- CKKS Homomorphic Encryption (Demo) ---', 'color: #3b82f6; font-weight: bold;');
    console.log('Original Clinical Payload:', data);
    console.log('Encrypted Patient Info (Ciphertexts):', encryptedResults);
    console.groupEnd();

    return encryptedResults;
}
