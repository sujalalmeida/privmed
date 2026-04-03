import base64
from typing import Any, Dict, Iterable, List

import tenseal as ts

from tenseal_ckks import (
    HE_CONTEXT_VERSION,
    build_encrypted_numeric_vector_payload,
    get_shared_public_context_payload,
    load_or_create_lab_context,
)

REPORT_NUMERIC_FIELDS = [
    'age',
    'bmi',
    'systolic_bp',
    'diastolic_bp',
    'heart_rate',
    'fasting_glucose',
    'hba1c',
    'total_cholesterol',
    'ldl_cholesterol',
    'hdl_cholesterol',
    'triglycerides',
    'max_heart_rate',
    'st_depression',
]

REPORT_PLAINTEXT_FIELDS = [
    'lab_id',
    'patient_id_hash',
    'prediction',
    'confidence',
    'clinical_reasoning',
]

SUMMARY_NUMERIC_FIELDS = [
    'age',
    'bmi',
    'systolic_bp',
    'diastolic_bp',
    'fasting_glucose',
    'hba1c',
]


def _b64encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode('ascii')


def _b64decode(raw: str) -> bytes:
    return base64.b64decode(raw.encode('ascii'))


def _safe_float(value: Any) -> float | None:
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_plaintext_metadata(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'lab_id': report.get('lab_id'),
        'patient_id_hash': report.get('patient_id_hash'),
        'prediction': report.get('prediction'),
        'confidence': _safe_float(report.get('confidence')),
        'clinical_reasoning': report.get('clinical_reasoning') or '',
    }


def encrypt_patient_report(report_dict: Dict[str, Any], lab_label: str) -> Dict[str, Any]:
    vector_values: List[float] = []
    presence_counts: List[int] = []

    for field in REPORT_NUMERIC_FIELDS:
        numeric_value = _safe_float(report_dict.get(field))
        if numeric_value is None:
            vector_values.append(0.0)
            presence_counts.append(0)
        else:
            vector_values.append(float(numeric_value))
            presence_counts.append(1)

    if not any(presence_counts):
        raise ValueError("No numeric report fields were available for homomorphic encryption")

    payload = build_encrypted_numeric_vector_payload(lab_label, vector_values)

    return {
        'field_order': REPORT_NUMERIC_FIELDS,
        'ciphertext_b64': _b64encode(payload['ciphertext']),
        'value_presence': presence_counts,
        'context_fingerprint': payload['context_fingerprint'],
        'context_version': payload['context_version'],
        'he_scheme': payload['he_scheme'],
    }


def decrypt_patient_report(encrypted_report: Dict[str, Any], lab_label: str) -> Dict[str, float]:
    lab_context = load_or_create_lab_context(lab_label)
    encrypted_vector = ts.ckks_vector_from(lab_context, _b64decode(encrypted_report['ciphertext_b64']))
    decrypt_fn = getattr(encrypted_vector, 'decrypt')
    values = decrypt_fn()
    field_order = encrypted_report.get('field_order') or REPORT_NUMERIC_FIELDS
    presence = encrypted_report.get('value_presence') or [1] * len(field_order)
    decrypted: Dict[str, float] = {}

    for idx, field in enumerate(field_order):
        if idx >= len(values) or idx >= len(presence) or int(presence[idx]) <= 0:
            continue
        decrypted[field] = float(values[idx])

    return decrypted


def compute_encrypted_aggregate_payloads(encrypted_reports: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    bundles = list(encrypted_reports)
    if not bundles:
        raise ValueError("No encrypted reports were provided for aggregation")

    fingerprints = {bundle.get('context_fingerprint') for bundle in bundles}
    if len(fingerprints) != 1:
        raise ValueError("All encrypted patient reports must use the same CKKS public context")

    shared_context = get_shared_public_context_payload()
    public_context = ts.context_from(shared_context['public_context'])
    field_order = bundles[0].get('field_order') or REPORT_NUMERIC_FIELDS
    aggregate_vector = None
    aggregate_counts = [0] * len(field_order)

    for bundle in bundles:
        encrypted_vector = ts.ckks_vector_from(public_context, _b64decode(bundle['ciphertext_b64']))
        aggregate_vector = encrypted_vector if aggregate_vector is None else aggregate_vector + encrypted_vector
        bundle_counts = bundle.get('value_presence') or [1] * len(field_order)
        aggregate_counts = [
            aggregate_counts[idx] + int(bundle_counts[idx] if idx < len(bundle_counts) else 0)
            for idx in range(len(field_order))
        ]

    return {
        'ciphertext_b64': _b64encode(aggregate_vector.serialize()),
        'field_order': field_order,
        'value_presence_totals': aggregate_counts,
        'context_fingerprint': shared_context['context_fingerprint'],
        'context_version': shared_context['context_version'],
        'he_scheme': shared_context['he_scheme'],
    }


def _bucket_age_range(avg_age: float | None) -> str | None:
    if avg_age is None:
        return None
    lower = int(avg_age // 10) * 10
    upper = lower + 10
    return f"{lower}-{upper} years"


def decrypt_aggregate_stats(aggregate_payload: Dict[str, Any], lab_label: str) -> Dict[str, Any]:
    lab_context = load_or_create_lab_context(lab_label)
    averages: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    encrypted_vector = ts.ckks_vector_from(lab_context, _b64decode(aggregate_payload['ciphertext_b64']))
    decrypt_fn = getattr(encrypted_vector, 'decrypt')
    values = decrypt_fn()
    field_order = aggregate_payload.get('field_order') or REPORT_NUMERIC_FIELDS
    count_totals = aggregate_payload.get('value_presence_totals') or [0] * len(field_order)

    for idx, field in enumerate(field_order):
        if idx >= len(values):
            continue
        count = int(count_totals[idx] if idx < len(count_totals) else 0)
        counts[field] = count
        if count > 0:
            averages[field] = float(values[idx]) / count

    rounded_averages = {
        field: round(value, 2)
        for field, value in averages.items()
        if field in SUMMARY_NUMERIC_FIELDS
    }

    return {
        'averages': rounded_averages,
        'counts': counts,
        'age_band': _bucket_age_range(averages.get('age')),
        'summary_fields': SUMMARY_NUMERIC_FIELDS,
    }
