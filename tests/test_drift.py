"""
GOLIATH Production ML Drift Detection - Integration Tests
"""

import os
import sys
import unittest
import numpy as np

# Ensure root path in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from goliath_drift.drift_detector import ProductionDriftDetector

class TestGoliathDriftDetection(unittest.TestCase):

    def test_01_healthy_stream_no_drift_alert(self):
        """Verifies clean stream conforming to baseline does not raise false alarms."""
        np.random.seed(42)
        ref_conf = np.random.beta(a=5, b=2, size=1000)
        ref_ar = np.random.normal(loc=0.5, scale=0.08, size=1000)

        detector = ProductionDriftDetector(
            reference_confidences=ref_conf.tolist(),
            reference_aspect_ratios=ref_ar.tolist(),
        )

        # Ingest 100 samples matching distribution
        for _ in range(100):
            conf = float(np.random.beta(a=5, b=2))
            w = float(np.random.uniform(20, 60))
            ar = float(np.random.normal(loc=0.5, scale=0.08))
            h = w / max(ar, 1e-3)
            detector.record_detection(confidence=conf, bbox=[0, 0, w, h])

        report = detector.evaluate_drift()
        self.assertEqual(report["status"], "HEALTHY")
        self.assertFalse(report["drift_detected"])
        self.assertEqual(report["action_required"], "NOOP")

    def test_02_confidence_degradation_triggers_retraining(self):
        """Verifies confidence decay > 5% triggers automated alert & retraining flag."""
        np.random.seed(42)
        ref_conf = np.random.beta(a=8, b=2, size=1000)  # High confidence baseline (~0.80)
        detector = ProductionDriftDetector(reference_confidences=ref_conf.tolist())

        # Ingest degraded stream with lower confidence (~0.60)
        for _ in range(50):
            conf = float(np.random.uniform(0.40, 0.65))
            detector.record_detection(confidence=conf, bbox=[0, 0, 50, 100])

        report = detector.evaluate_drift()
        self.assertEqual(report["status"], "DRIFT_DETECTED")
        self.assertTrue(report["drift_detected"])
        self.assertTrue(report["confidence_drop_pct"] > 5.0)
        self.assertEqual(report["action_required"], "TRIGGER_SAGEMAKER_RETRAINING")


if __name__ == "__main__":
    unittest.main()
