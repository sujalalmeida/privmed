import os
import sys
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SERVER_DIR = os.path.join(REPO_ROOT, 'server')
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from privacy.report_encryption import (  # noqa: E402
    compute_encrypted_aggregate_payloads,
    decrypt_aggregate_stats,
    decrypt_patient_report,
    encrypt_patient_report,
)


class ReportEncryptionTests(unittest.TestCase):
    def test_encrypt_then_decrypt_patient_report(self):
        report = {
            'age': 45,
            'bmi': 27.4,
            'systolic_bp': 138,
            'diastolic_bp': 88,
            'fasting_glucose': 124,
            'hba1c': 6.2,
        }

        encrypted = encrypt_patient_report(report, 'lab_A')
        self.assertIn('ciphertext_b64', encrypted)
        self.assertIn('field_order', encrypted)
        self.assertNotEqual(
            encrypted['ciphertext_b64'],
            '45',
        )
        self.assertGreater(len(encrypted['ciphertext_b64']), 100)

        decrypted = decrypt_patient_report(encrypted, 'lab_A')
        self.assertAlmostEqual(decrypted['age'], 45.0, places=3)
        self.assertAlmostEqual(decrypted['bmi'], 27.4, places=3)
        self.assertAlmostEqual(decrypted['fasting_glucose'], 124.0, places=3)

    def test_aggregate_stats_match_plaintext_means(self):
        report_a = encrypt_patient_report({
            'age': 40,
            'bmi': 24.0,
            'systolic_bp': 120,
            'diastolic_bp': 80,
            'fasting_glucose': 100,
            'hba1c': 5.8,
        }, 'lab_A')
        report_b = encrypt_patient_report({
            'age': 60,
            'bmi': 30.0,
            'systolic_bp': 140,
            'diastolic_bp': 90,
            'fasting_glucose': 140,
            'hba1c': 7.0,
        }, 'lab_A')

        aggregate_payload = compute_encrypted_aggregate_payloads([report_a, report_b])
        stats = decrypt_aggregate_stats(aggregate_payload, 'lab_A')

        self.assertAlmostEqual(stats['averages']['age'], 50.0, places=2)
        self.assertAlmostEqual(stats['averages']['bmi'], 27.0, places=2)
        self.assertAlmostEqual(stats['averages']['fasting_glucose'], 120.0, places=2)
        self.assertEqual(stats['age_band'], '40-50 years')


if __name__ == '__main__':
    unittest.main()
