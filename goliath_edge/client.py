"""
GOLIATH Edge SDK - Production Edge Node Client
Manages gRPC bidirectional streams, automatic reconnection, ring-buffer telemetry, and live model updates.
"""

import time
import logging
import threading
import queue
from typing import Optional, Callable, Dict, Any, List
import grpc

from proto.v1 import fleet_control_pb2
from proto.v1 import fleet_control_pb2_grpc
from goliath_edge.ring_buffer import EdgeTelemetryRingBuffer

logger = logging.getLogger("goliath.edge.client")

class GoliathEdgeClient:
    """
    Production-grade edge client connecting to GOLIATH Control Plane.
    """

    def __init__(
        self,
        node_id: str,
        server_address: str = "localhost:50051",
        cluster_id: str = "default-cluster",
        firmware_version: str = "1.0.0",
        active_model_version: str = "yolov8n-v1.0.0",
        hardware_profile: Optional[Dict[str, Any]] = None,
        heartbeat_interval_sec: float = 3.0,
    ):
        self.node_id = node_id
        self.server_address = server_address
        self.cluster_id = cluster_id
        self.firmware_version = firmware_version
        self.active_model_version = active_model_version
        self.hardware_profile = hardware_profile or {
            "arch": "x86_64",
            "accelerator": "cpu",
            "total_ram_mb": 8192,
            "available_ram_mb": 4096,
            "compute_score": 1.0,
        }
        self.heartbeat_interval_sec = heartbeat_interval_sec

        self.session_token: Optional[str] = None
        self.ring_buffer = EdgeTelemetryRingBuffer(max_capacity=5000)
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[fleet_control_pb2_grpc.FleetControlServiceStub] = None

        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._telemetry_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._on_model_update_cb: Optional[Callable[[str, str], None]] = None
        self._on_config_update_cb: Optional[Callable[[str], None]] = None

        # Edge state telemetry
        self.current_fps: float = 0.0
        self.current_latency_ms: float = 0.0
        self.current_memory_mb: float = 0.0
        self.active_tracks: int = 0
        self.batch_counter: int = 0

    def set_model_update_handler(self, callback: Callable[[str, str], None]):
        self._on_model_update_cb = callback

    def set_config_update_handler(self, callback: Callable[[str], None]):
        self._on_config_update_cb = callback

    def start(self) -> bool:
        """Connects, registers, and starts background stream workers."""
        logger.info(f"[EdgeClient] Connecting to GOLIATH Control Plane at {self.server_address}...")
        self._channel = grpc.insecure_channel(
            self.server_address,
            options=[
                ("grpc.max_receive_message_length", 32 * 1024 * 1024),
                ("grpc.max_send_message_length", 32 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 30000),
            ],
        )
        self._stub = fleet_control_pb2_grpc.FleetControlServiceStub(self._channel)

        if not self._register_node():
            logger.error("[EdgeClient] Initial node registration failed.")
            return False

        self._running = True
        self._heartbeat_thread = threading.Thread(target=self._run_control_stream, daemon=True)
        self._telemetry_thread = threading.Thread(target=self._run_telemetry_stream, daemon=True)
        self._heartbeat_thread.start()
        self._telemetry_thread.start()
        logger.info("[EdgeClient] Background streams initialized successfully.")
        return True

    def stop(self):
        self._running = False
        if self._channel:
            self._channel.close()
        logger.info("[EdgeClient] Stopped.")

    def record_detection(
        self,
        track_id: int,
        class_name: str,
        confidence: float,
        bbox: List[float],
        threat_level: int,
        in_restricted_zone: bool,
        snapshot_jpeg: bytes = b"",
    ):
        """Pushes detection event to the prioritized local ring buffer."""
        evt = fleet_control_pb2.DetectionEvent(
            timestamp_epoch_ms=int(time.time() * 1000),
            track_id=track_id,
            class_name=class_name,
            confidence=confidence,
            bbox=bbox,
            threat_level=threat_level,
            in_restricted_zone=in_restricted_zone,
            optional_snapshot_jpeg=snapshot_jpeg,
        )
        is_critical = (threat_level == 2) or in_restricted_zone
        self.ring_buffer.push(evt, is_critical=is_critical)

    def _register_node(self) -> bool:
        req = fleet_control_pb2.NodeRegistrationRequest(
            node_id=self.node_id,
            cluster_id=self.cluster_id,
            firmware_version=self.firmware_version,
            active_model_version=self.active_model_version,
            hardware=fleet_control_pb2.HardwareProfile(
                arch=self.hardware_profile.get("arch", "x86_64"),
                accelerator=self.hardware_profile.get("accelerator", "cpu"),
                total_ram_mb=self.hardware_profile.get("total_ram_mb", 4096),
                available_ram_mb=self.hardware_profile.get("available_ram_mb", 2048),
                compute_score=self.hardware_profile.get("compute_score", 1.0),
            ),
        )
        try:
            resp = self._stub.RegisterNode(req, timeout=10.0)
            if resp.registered:
                self.session_token = resp.session_token
                logger.info(
                    f"[EdgeClient] Registered successfully! Session={self.session_token}, Model={resp.assigned_model_version}"
                )
                if resp.assigned_model_version != self.active_model_version and self._on_model_update_cb:
                    self._on_model_update_cb(resp.assigned_model_version, resp.model_manifest_url)
                return True
            else:
                logger.error(f"[EdgeClient] Registration rejected: {resp.message}")
                return False
        except Exception as e:
            logger.error(f"[EdgeClient] Registration RPC failed: {e}")
            return False

    def _run_control_stream(self):
        while self._running:
            try:
                heartbeat_queue = queue.Queue()

                def heartbeat_generator():
                    while self._running:
                        try:
                            item = heartbeat_queue.get(timeout=self.heartbeat_interval_sec)
                            yield item
                        except queue.Empty:
                            # Periodic heartbeat
                            yield fleet_control_pb2.HeartbeatMessage(
                                node_id=self.node_id,
                                session_token=self.session_token,
                                timestamp_utc_epoch_ms=int(time.time() * 1000),
                                status=fleet_control_pb2.NodeStatus.NODE_STATUS_HEALTHY,
                                inference_fps=self.current_fps,
                                inference_latency_ms=self.current_latency_ms,
                                memory_usage_mb=self.current_memory_mb,
                                active_tracks_count=self.active_tracks,
                                total_detections_since_last=0,
                            )

                response_iterator = self._stub.StreamControl(heartbeat_generator())
                for directive in response_iterator:
                    self._handle_directive(directive)

            except Exception as e:
                if self._running:
                    logger.warning(f"[EdgeClient] Control stream interrupted ({e}). Reconnecting in 3s...")
                    time.sleep(3.0)
                    self._register_node()

    def _run_telemetry_stream(self):
        while self._running:
            try:
                def telemetry_generator():
                    while self._running:
                        batch_events = self.ring_buffer.pop_batch(max_batch_size=50)
                        if batch_events:
                            self.batch_counter += 1
                            yield fleet_control_pb2.TelemetryBatch(
                                node_id=self.node_id,
                                session_token=self.session_token,
                                batch_sequence_id=self.batch_counter,
                                events=batch_events,
                            )
                        else:
                            time.sleep(0.05)  # 50ms poll loop

                response_iterator = self._stub.StreamTelemetry(telemetry_generator())
                for ack in response_iterator:
                    if not ack.success:
                        logger.warning(f"[EdgeClient] Telemetry batch {ack.batch_sequence_id} rejected: {ack.error_message}")

            except Exception as e:
                if self._running:
                    logger.warning(f"[EdgeClient] Telemetry stream interrupted ({e}). Backing off 2s...")
                    time.sleep(2.0)

    def _handle_directive(self, directive: fleet_control_pb2.CommandDirective):
        action_name = fleet_control_pb2.CommandDirective.Action.Name(directive.action)
        logger.info(f"[EdgeClient] Received directive: {action_name} (ID: {directive.directive_id})")

        if directive.action == fleet_control_pb2.CommandDirective.ACTION_HOT_SWAP_MODEL:
            logger.info(f"[EdgeClient] Triggering live model swap to: {directive.target_model_version}")
            self.active_model_version = directive.target_model_version
            if self._on_model_update_cb:
                self._on_model_update_cb(directive.target_model_version, directive.target_model_manifest_url)

        elif directive.action == fleet_control_pb2.CommandDirective.ACTION_UPDATE_CONFIG:
            if self._on_config_update_cb:
                self._on_config_update_cb(directive.config_payload_json)
