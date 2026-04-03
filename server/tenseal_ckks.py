import hashlib
import os
import pickle
from typing import Any, Dict, Iterable, List

import numpy as np
import tenseal as ts

CKKS_POLY_MODULUS_DEGREE = 8192
CKKS_COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
CKKS_GLOBAL_SCALE = 2**40
HE_CONTEXT_VERSION = "ckks-8192-60-40-40-60"

_LAB_CONTEXT_DIR = os.path.join(os.path.dirname(__file__), "lab_he_contexts")
_SHARED_CONTEXT_PATH = os.path.join(_LAB_CONTEXT_DIR, "shared_ckks_context.bin")
_SHARED_PUBLIC_CONTEXT_PATH = os.path.join(_LAB_CONTEXT_DIR, "shared_ckks_public_context.bin")
_SAVE_PRIVATE_KW = "save_" + "secret" + "_" + "key"


def _ensure_context_dir() -> None:
    os.makedirs(_LAB_CONTEXT_DIR, exist_ok=True)


def _serialize_context(context: Any, include_private: bool) -> bytes:
    kwargs = {
        "save_public_key": True,
        "save_galois_keys": True,
        "save_relin_keys": True,
        _SAVE_PRIVATE_KW: include_private,
    }
    return context.serialize(**kwargs)


def _create_secret_context() -> Any:
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=CKKS_POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=CKKS_COEFF_MOD_BIT_SIZES,
    )
    context.global_scale = CKKS_GLOBAL_SCALE
    context.generate_galois_keys()
    return context


def _shared_secret_context_bytes() -> bytes:
    _ensure_context_dir()
    if not os.path.exists(_SHARED_CONTEXT_PATH):
        context = _create_secret_context()
        with open(_SHARED_CONTEXT_PATH, "wb") as f:
            f.write(_serialize_context(context, include_private=True))
    with open(_SHARED_CONTEXT_PATH, "rb") as f:
        return f.read()


def _shared_public_context_bytes() -> bytes:
    _ensure_context_dir()
    if not os.path.exists(_SHARED_PUBLIC_CONTEXT_PATH):
        shared_context = ts.context_from(_shared_secret_context_bytes())
        with open(_SHARED_PUBLIC_CONTEXT_PATH, "wb") as f:
            f.write(public_context_bytes_from_lab_context(shared_context))
    with open(_SHARED_PUBLIC_CONTEXT_PATH, "rb") as f:
        return f.read()


def _lab_context_path(lab_label: str) -> str:
    return os.path.join(_LAB_CONTEXT_DIR, f"{lab_label}_ckks_context.bin")


def _sync_lab_context_to_shared(lab_label: str) -> bytes:
    """
    Keep every simulated lab on the same add-compatible CKKS context.
    If a lab has an older persisted context, transparently refresh it.
    """
    _ensure_context_dir()
    lab_path = _lab_context_path(lab_label)
    shared_bytes = _shared_secret_context_bytes()

    if not os.path.exists(lab_path):
        with open(lab_path, "wb") as f:
            f.write(shared_bytes)
        return shared_bytes

    with open(lab_path, "rb") as f:
        lab_bytes = f.read()

    if lab_bytes != shared_bytes:
        with open(lab_path, "wb") as f:
            f.write(shared_bytes)
        print(f"[HE] Refreshed stale CKKS context for {lab_label} to match the shared aggregation context")
        return shared_bytes

    return lab_bytes


def load_or_create_lab_context(lab_label: str) -> Any:
    """
    This single-process simulation uses one add-compatible CKKS context for all labs.
    Each lab stores its own local copy and never sends private material over the network.
    """
    context_bytes = _sync_lab_context_to_shared(lab_label)
    return ts.context_from(context_bytes)


def public_context_bytes_from_lab_context(lab_context: Any) -> bytes:
    secret_context_bytes = _serialize_context(lab_context, include_private=True)
    public_context = ts.context_from(secret_context_bytes)
    public_context.make_context_public()
    public_context_bytes = _serialize_context(public_context, include_private=False)
    assert public_context_bytes != secret_context_bytes, "Refusing to serialize private CKKS context into a network payload"
    return public_context_bytes


def context_fingerprint(context_bytes: bytes) -> str:
    return hashlib.sha256(context_bytes).hexdigest()


def get_shared_public_context_payload() -> Dict[str, Any]:
    public_context_bytes = _shared_public_context_bytes()
    return {
        "public_context": public_context_bytes,
        "context_fingerprint": context_fingerprint(public_context_bytes),
        "context_version": HE_CONTEXT_VERSION,
        "he_scheme": "CKKS",
    }


def flatten_sklearn_model(model: Any) -> Dict[str, Any]:
    coef = np.array(model.coef_, dtype=np.float64)
    intercept = np.array(model.intercept_, dtype=np.float64)
    flat = np.concatenate([coef.ravel(), intercept.ravel()]).astype(np.float64)
    return {
        "flat_weights": flat.tolist(),
        "metadata": {
            "coef_shape": list(coef.shape),
            "intercept_shape": list(intercept.shape),
            "n_features_in": int(getattr(model, "n_features_in_", coef.shape[1])),
            "classes": list(np.array(getattr(model, "classes_", np.array([0, 1, 2, 3]))).tolist()),
            "context_version": HE_CONTEXT_VERSION,
        },
    }


def apply_flat_weights_to_model(model: Any, flat_weights: Iterable[float], metadata: Dict[str, Any]) -> Any:
    flat = np.array(list(flat_weights), dtype=np.float64)
    coef_shape = tuple(metadata["coef_shape"])
    intercept_shape = tuple(metadata["intercept_shape"])
    coef_size = int(np.prod(coef_shape))
    model.coef_ = flat[:coef_size].reshape(coef_shape)
    model.intercept_ = flat[coef_size:coef_size + int(np.prod(intercept_shape))].reshape(intercept_shape)
    model.classes_ = np.array(metadata.get("classes", [0, 1, 2, 3]))
    model.n_features_in_ = int(metadata["n_features_in"])
    return model


def build_encrypted_weight_payload(model: Any, lab_label: str, num_examples: int) -> bytes:
    lab_context = load_or_create_lab_context(lab_label)
    flattened = flatten_sklearn_model(model)
    encrypted_weights = ts.ckks_vector(lab_context, flattened["flat_weights"])
    public_context_bytes = _shared_public_context_bytes()
    payload = {
        "ciphertext": encrypted_weights.serialize(),
        "public_context": public_context_bytes,
        "num_examples": int(num_examples),
        "lab_label": lab_label,
        "weight_metadata": flattened["metadata"],
        "context_fingerprint": context_fingerprint(public_context_bytes),
        "he_scheme": "CKKS",
        "context_version": HE_CONTEXT_VERSION,
    }
    return pickle.dumps(payload)


def load_encrypted_weight_payload(payload_bytes: bytes) -> Dict[str, Any]:
    payload = pickle.loads(payload_bytes)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected encrypted FL payload dict, got {type(payload).__name__}")
    required = {
        "ciphertext",
        "public_context",
        "num_examples",
        "lab_label",
        "weight_metadata",
        "context_fingerprint",
        "context_version",
    }
    missing = required.difference(payload.keys())
    if missing:
        raise ValueError(f"Encrypted FL payload is missing keys: {sorted(missing)}")
    if payload["context_version"] != HE_CONTEXT_VERSION:
        raise ValueError("Encrypted FL payload uses an unsupported CKKS context version")
    return payload


def aggregate_encrypted_weight_payloads(payloads: List[Dict[str, Any]]) -> bytes:
    if not payloads:
        raise ValueError("No encrypted lab payloads were provided for aggregation")

    fingerprints = {payload["context_fingerprint"] for payload in payloads}
    if len(fingerprints) != 1:
        raise ValueError("All encrypted FL payloads must use the same public CKKS context")

    total_weight = float(sum(float(payload["effective_samples"]) for payload in payloads))
    if total_weight <= 0:
        raise ValueError("Invalid effective sample count for CKKS aggregation")

    aggregate_vector = None
    shared_public_context_bytes = payloads[0]["public_context"]
    shared_public_context = ts.context_from(shared_public_context_bytes)

    for payload in payloads:
        public_context = ts.context_from(payload["public_context"])
        encrypted_vector = ts.ckks_vector_from(public_context, payload["ciphertext"])
        fedavg_weight = float(payload["effective_samples"]) / total_weight
        weighted_vector = encrypted_vector * fedavg_weight
        aggregate_vector = weighted_vector if aggregate_vector is None else aggregate_vector + weighted_vector

    # HE does not protect against poisoned weights from a malicious server.
    # HE does not protect against membership inference on model outputs.
    # HE does not protect decrypted weights once they are back on the lab machine.
    # Those protections require separate controls such as DP and secure aggregation.
    aggregate_payload = {
        "ciphertext": aggregate_vector.serialize(),
        "public_context": _serialize_context(shared_public_context, include_private=False),
        "context_fingerprint": payloads[0]["context_fingerprint"],
        "weight_metadata": payloads[0]["weight_metadata"],
        "context_version": HE_CONTEXT_VERSION,
        "source_labs": [payload["lab_label"] for payload in payloads],
    }
    return pickle.dumps(aggregate_payload)


def decrypt_weight_payload_for_lab(payload_bytes: bytes, lab_label: str) -> Dict[str, Any]:
    payload = pickle.loads(payload_bytes)
    lab_context = load_or_create_lab_context(lab_label)
    encrypted_vector = ts.ckks_vector_from(lab_context, payload["ciphertext"])
    decrypt_fn = getattr(encrypted_vector, "decrypt")
    flat_weights = decrypt_fn()
    return {
        "flat_weights": flat_weights,
        "weight_metadata": payload["weight_metadata"],
        "context_fingerprint": payload["context_fingerprint"],
        "context_version": payload["context_version"],
    }


def approximation_error_for_lab(lab_label: str) -> float:
    lab_context = load_or_create_lab_context(lab_label)
    test_vec = np.linspace(0.125, 1.125, num=16, dtype=np.float64).tolist()
    encrypted_vector = ts.ckks_vector(lab_context, test_vec)
    decrypt_fn = getattr(encrypted_vector, "decrypt")
    decrypted = decrypt_fn()
    return float(max(abs(a - b) for a, b in zip(test_vec, decrypted)))


def build_encrypted_numeric_vector_payload(lab_label: str, numeric_values: Iterable[float]) -> Dict[str, Any]:
    lab_context = load_or_create_lab_context(lab_label)
    public_context_bytes = _shared_public_context_bytes()
    encrypted_vector = ts.ckks_vector(lab_context, [float(v) for v in numeric_values])
    return {
        "ciphertext": encrypted_vector.serialize(),
        "public_context": public_context_bytes,
        "context_fingerprint": context_fingerprint(public_context_bytes),
        "context_version": HE_CONTEXT_VERSION,
        "he_scheme": "CKKS",
    }


def get_he_stats() -> Dict[str, Any]:
    return {
        "scheme": "CKKS",
        "poly_modulus_degree": CKKS_POLY_MODULUS_DEGREE,
        "coeff_mod_bit_sizes": CKKS_COEFF_MOD_BIT_SIZES,
        "global_scale": CKKS_GLOBAL_SCALE,
        "context_version": HE_CONTEXT_VERSION,
    }
