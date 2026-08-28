"""
GOLIATH Multi-Object Tracking (ByteTrack) - Integration & Benchmark Tests
"""

import os
import sys
import time
import unittest
import numpy as np

# Ensure root path in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from goliath_tracking.kalman_filter import KalmanFilter
from goliath_tracking.byte_tracker import ByteTracker, STrack
from goliath_tracking.trajectory_manager import TrajectoryAndZoneManager

class TestGoliathTracking(unittest.TestCase):

    def setUp(self):
        STrack.reset_counter()

    def test_01_kalman_filter_state_evolution(self):
        """Verifies Kalman filter velocity estimation and measurement projection."""
        kf = KalmanFilter()
        measurement = np.array([100.0, 100.0, 0.5, 200.0]) # [x, y, a, h]
        mean, cov = kf.initiate(measurement)

        self.assertEqual(len(mean), 8)
        self.assertEqual(cov.shape, (8, 8))

        # Predict next frame
        pred_mean, pred_cov = kf.predict(mean, cov)
        self.assertEqual(len(pred_mean), 8)

        # Update with new shifted measurement (moving right and down)
        new_measurement = np.array([105.0, 103.0, 0.5, 200.0])
        upd_mean, upd_cov = kf.update(pred_mean, pred_cov, new_measurement)
        
        # Velocity in X should be positive
        vx = upd_mean[4]
        self.assertTrue(vx > 0.0)

    def test_02_bytetrack_occlusion_recovery_no_id_switch(self):
        """
        Critical Test: Object is tracked at high confidence, occluded (confidence drops to 0.2),
        and emerges again. Verifies that ByteTrack recovers the track without spawning a new ID.
        """
        tracker = ByteTracker(track_thresh=0.5, low_conf_thresh=0.1, match_thresh=0.8)

        # Frame 1: High confidence detection
        det1 = np.array([[100, 100, 150, 200, 0.90, 0]]) # [x1, y1, x2, y2, score, class]
        tracks = tracker.update(det1)
        self.assertEqual(len(tracks), 1)
        original_track_id = tracks[0].track_id

        # Frames 2-5: Moving with high confidence
        for i in range(1, 5):
            det = np.array([[100 + i * 5, 100 + i * 2, 150 + i * 5, 200 + i * 2, 0.88, 0]])
            tracks = tracker.update(det)
            self.assertEqual(len(tracks), 1)
            self.assertEqual(tracks[0].track_id, original_track_id)

        # Frames 6-8: Severe Occlusion (Confidence drops to 0.20 - below standard 0.5 threshold)
        for i in range(5, 8):
            det_low = np.array([[100 + i * 5, 100 + i * 2, 150 + i * 5, 200 + i * 2, 0.20, 0]])
            tracks = tracker.update(det_low)
            # ByteTrack Stage 2 must re-associate this low-confidence box
            self.assertEqual(len(tracks), 1)
            self.assertEqual(tracks[0].track_id, original_track_id)

        # Frame 9: Clears occlusion (Confidence returns to 0.92)
        det_clear = np.array([[100 + 8 * 5, 100 + 8 * 2, 150 + 8 * 5, 200 + 8 * 2, 0.92, 0]])
        tracks = tracker.update(det_clear)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, original_track_id, "Track ID must remain invariant across occlusion!")

    def test_03_trajectory_and_zone_violation(self):
        """Verifies trajectory recording, speed computation, and polygon intrusion detection."""
        # Restricted polygon: Box from (100, 100) to (300, 300)
        restricted_zone = [(100, 100), (300, 100), (300, 300), (100, 300)]
        traj_mgr = TrajectoryAndZoneManager(restricted_polygon_coords=restricted_zone)

        # Point outside zone: [x1=10, y1=10, x2=50, y2=50] -> foot=(30, 50)
        res1 = traj_mgr.process_track(
            track_id=1,
            class_id=0,
            tlbr=np.array([10, 10, 50, 50]),
            timestamp_ms=1000,
        )
        self.assertFalse(res1["in_restricted_zone"])

        # Point enters zone: [x1=150, y1=150, x2=200, y2=250] -> foot=(175, 250) (inside polygon)
        res2 = traj_mgr.process_track(
            track_id=1,
            class_id=0,
            tlbr=np.array([150, 150, 200, 250]),
            timestamp_ms=1100,
        )
        self.assertTrue(res2["in_restricted_zone"])
        self.assertTrue(res2["speed"] > 0)
        self.assertEqual(len(res2["trajectory_points"]), 2)

    def test_04_bytetrack_latency_benchmark(self):
        """Measures tracking latency across 50 simultaneous objects per frame."""
        tracker = ByteTracker()
        
        # Create 50 simulated objects
        np.random.seed(42)
        n_objects = 50
        base_boxes = np.zeros((n_objects, 6))
        for i in range(n_objects):
            x = np.random.uniform(50, 500)
            y = np.random.uniform(50, 400)
            base_boxes[i] = [x, y, x + 40, y + 80, 0.85, 0]

        # Warm up
        tracker.update(base_boxes)

        # Benchmark 100 frames
        latencies = []
        for frame in range(100):
            # Apply slight velocity jitter
            boxes = base_boxes.copy()
            boxes[:, 0] += np.random.normal(1.0, 0.5, n_objects)
            boxes[:, 1] += np.random.normal(0.5, 0.5, n_objects)
            boxes[:, 2] = boxes[:, 0] + 40
            boxes[:, 3] = boxes[:, 1] + 80

            start = time.perf_counter()
            tracker.update(boxes)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        print(f"\n[Benchmark] ByteTrack with 50 tracks: Avg Latency = {avg_latency:.2f}ms, P95 = {p95_latency:.2f}ms")
        
        # Must easily satisfy frame tracking budget (< 5.0ms for 50 tracks on CPU)
        self.assertTrue(avg_latency < 5.0, f"Average latency {avg_latency:.2f}ms exceeds 5ms limit")


if __name__ == "__main__":
    unittest.main()
