"""
GOLIATH Fleet Orchestration - Comprehensive Integration Test Suite
Verifies gRPC Control Plane, Model Registry, Ring Buffer, and Edge Client.
"""

import os
import sys
import time
import unittest
import threading

# Ensure root path in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from proto.v1 import fleet_control_pb2
from goliath_control.node_manager import NodeLifecycleManager
from goliath_control.telemetry_ingest import TelemetryIngestionEngine
from goliath_control.model_registry import ModelRegistry
from goliath_control.server import GoliathControlServer
from goliath_edge.ring_buffer import EdgeTelemetryRingBuffer
from goliath_edge.client import GoliathEdgeClient

class TestGoliathFleetOrchestration(unittest.TestCase):

    def test_01_model_registry_sha256_and_manifest(self):
        """Validates SHA-256 computation, manifest packaging, and canary distribution."""
        registry = ModelRegistry(storage_root="test_models_storage")
        
        # Test existing yolov8n.onnx
        model_path = os.path.join(ROOT_DIR, "yolov8n.onnx")
        if os.path.exists(model_path):
            manifest = registry.register_model_version(
                model_name="yolov8n-threat-detector",
                version="v1.1.0-fp16",
                artifact_path=model_path,
                input_shape=[1, 3, 640, 640],
                confidence_threshold=0.5,
                iou_threshold=0.45,
                target_accelerators=["webgl", "webgpu", "cuda"],
            )

            self.assertEqual(manifest["version"], "v1.1.0-fp16")
            self.assertEqual(len(manifest["sha256"]), 64)
            self.assertTrue(manifest["file_size_bytes"] > 0)

            # Test canary assignment
            registry.set_canary_rollout("v1.1.0-fp16", 1.0)
            assigned = registry.assign_model_for_node("node-edge-001", "wasm32", "webgl")
            self.assertIsNotNone(assigned)
            self.assertEqual(assigned["version"], "v1.1.0-fp16")

    def test_02_edge_ring_buffer_priority_eviction(self):
        """Verifies ring buffer preserves critical intrusion alerts when overflowing."""
        buf = EdgeTelemetryRingBuffer(max_capacity=5)

        # Push 4 ambient events
        for i in range(4):
            buf.push(f"ambient_{i}", is_critical=False)
        self.assertEqual(buf.size(), 4)

        # Push 2 critical events -> Should evict 1 ambient
        buf.push("critical_1", is_critical=True)
        buf.push("critical_2", is_critical=True)
        
        self.assertEqual(buf.size(), 5)
        self.assertEqual(buf.dropped_ambient_count, 1)

        # Drain batch
        batch = buf.pop_batch(max_batch_size=10)
        # Critical items must be returned first
        self.assertEqual(batch[0], "critical_1")
        self.assertEqual(batch[1], "critical_2")
        self.assertTrue(buf.is_empty())

    def test_03_node_lifecycle_state_machine(self):
        """Validates node registration, authentication, and heartbeat tracking."""
        mgr = NodeLifecycleManager(heartbeat_timeout_seconds=2.0)
        state = mgr.register_node(
            node_id="edge-camera-alpha",
            cluster_id="us-east-1",
            firmware_version="2.4.0",
            active_model_version="v1.0.0",
            hardware_arch="arm64",
            accelerator="cuda_12_2",
            total_ram_mb=8192,
            available_ram_mb=6144,
            compute_score=3.5,
            tags={"zone": "vault-perimeter"},
        )

        self.assertEqual(state.status, "HEALTHY")
        self.assertTrue(mgr.validate_session("edge-camera-alpha", state.session_token))

        # Queue a directive
        mgr.queue_directive("edge-camera-alpha", {
            "action": "ACTION_HOT_SWAP_MODEL",
            "target_model_version": "v1.1.0-fp16",
        })

        # Record heartbeat and verify popped directive
        directives = mgr.record_heartbeat(
            node_id="edge-camera-alpha",
            session_token=state.session_token,
            timestamp_utc_ms=int(time.time() * 1000),
            status_name="HEALTHY",
            fps=28.5,
            latency_ms=35.1,
            memory_mb=1024.0,
            tracks_count=3,
            detections_since_last=10,
        )

        self.assertEqual(len(directives), 1)
        self.assertEqual(directives[0]["action"], "ACTION_HOT_SWAP_MODEL")

    def test_04_end_to_end_grpc_control_and_telemetry(self):
        """Spins up live gRPC Control Plane, connects Edge Client, exchanges streams."""
        test_port = 50055
        server = GoliathControlServer(host="127.0.0.1", port=test_port)
        server.start()
        time.sleep(0.5)

        received_model_updates = []
        def on_model_update(version, manifest_url):
            received_model_updates.append(version)

        client = GoliathEdgeClient(
            node_id="node-test-integration-01",
            server_address=f"127.0.0.1:{test_port}",
            active_model_version="yolov8n-v1.0.0",
        )
        client.set_model_update_handler(on_model_update)

        success = client.start()
        self.assertTrue(success)
        self.assertIsNotNone(client.session_token)

        # Queue hot swap directive on server
        server.node_manager.queue_directive("node-test-integration-01", {
            "action": "ACTION_HOT_SWAP_MODEL",
            "target_model_version": "yolov8n-v2.0.0-int8",
            "target_model_manifest_url": "https://cdn.internal/v2.0.0/manifest.json",
        })

        # Push detection events from edge
        client.record_detection(
            track_id=101,
            class_name="person",
            confidence=0.92,
            bbox=[0.2, 0.3, 0.4, 0.8],
            threat_level=2,  # CRITICAL_INTRUSION
            in_restricted_zone=True,
        )
        client.record_detection(
            track_id=102,
            class_name="backpack",
            confidence=0.81,
            bbox=[0.25, 0.5, 0.35, 0.7],
            threat_level=0,  # Ambient
            in_restricted_zone=False,
        )

        # Wait for control stream tick and telemetry batch drain
        time.sleep(4.0)

        # Verify server ingested the critical intrusion
        metrics = server.telemetry_engine.get_ingest_metrics()
        self.assertTrue(metrics["total_events_ingested"] >= 2)
        self.assertTrue(metrics["total_critical_intrusions"] >= 1)

        # Verify model update directive was processed by edge client
        self.assertIn("yolov8n-v2.0.0-int8", received_model_updates)

        # Cleanup
        client.stop()
        server.stop(grace_period=1.0)


if __name__ == "__main__":
    unittest.main()
