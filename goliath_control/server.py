"""
GOLIATH Control Plane - High-Performance gRPC Server
Coordinates node registration, bi-directional control streams, and high-throughput telemetry ingestion.
"""

import time
import logging
import grpc
from concurrent import futures
from typing import Iterator

from proto.v1 import fleet_control_pb2
from proto.v1 import fleet_control_pb2_grpc
from goliath_control.node_manager import NodeLifecycleManager
from goliath_control.telemetry_ingest import TelemetryIngestionEngine
from goliath_control.model_registry import ModelRegistry

logger = logging.getLogger("goliath.control.server")

class FleetControlServicer(fleet_control_pb2_grpc.FleetControlServiceServicer):
    """
    gRPC Servicer implementing the core control plane protocol.
    """

    def __init__(
        self,
        node_manager: NodeLifecycleManager,
        telemetry_engine: TelemetryIngestionEngine,
        model_registry: ModelRegistry,
    ):
        self.node_manager = node_manager
        self.telemetry_engine = telemetry_engine
        self.model_registry = model_registry

    def RegisterNode(
        self,
        request: fleet_control_pb2.NodeRegistrationRequest,
        context: grpc.ServicerContext,
    ) -> fleet_control_pb2.NodeRegistrationResponse:
        try:
            logger.info(f"[FleetServer] RegisterNode received for node_id={request.node_id}")
            
            # Model assignment strategy
            assigned_manifest = self.model_registry.assign_model_for_node(
                node_id=request.node_id,
                hardware_arch=request.hardware.arch,
                accelerator=request.hardware.accelerator,
            )

            assigned_version = assigned_manifest["version"] if assigned_manifest else "yolov8n-v1.0.0"
            manifest_url = assigned_manifest["manifest_url"] if assigned_manifest else "https://cdn.goliath-sec.internal/models/yolov8n-v1.0.0/manifest.json"

            node_state = self.node_manager.register_node(
                node_id=request.node_id,
                cluster_id=request.cluster_id or "default-cluster",
                firmware_version=request.firmware_version,
                active_model_version=request.active_model_version,
                hardware_arch=request.hardware.arch,
                accelerator=request.hardware.accelerator,
                total_ram_mb=request.hardware.total_ram_mb,
                available_ram_mb=request.hardware.available_ram_mb,
                compute_score=request.hardware.compute_score,
                tags=dict(request.tags),
                assigned_model_version=assigned_version,
                manifest_url=manifest_url,
            )

            return fleet_control_pb2.NodeRegistrationResponse(
                registered=True,
                session_token=node_state.session_token,
                heartbeat_interval_ms=5000,
                assigned_model_version=assigned_version,
                model_manifest_url=manifest_url,
                message="Registration successful. Node state: HEALTHY.",
            )
        except Exception as e:
            logger.error(f"[FleetServer] Registration error: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return fleet_control_pb2.NodeRegistrationResponse(
                registered=False,
                message=f"Registration failed: {e}"
            )

    def StreamControl(
        self,
        request_iterator: Iterator[fleet_control_pb2.HeartbeatMessage],
        context: grpc.ServicerContext,
    ) -> Iterator[fleet_control_pb2.CommandDirective]:
        logger.info("[FleetServer] Edge node connected to bi-directional StreamControl")
        try:
            for heartbeat in request_iterator:
                # Map protobuf status enum to string
                status_name = fleet_control_pb2.NodeStatus.Name(heartbeat.status)
                
                directives = self.node_manager.record_heartbeat(
                    node_id=heartbeat.node_id,
                    session_token=heartbeat.session_token,
                    timestamp_utc_ms=heartbeat.timestamp_utc_epoch_ms,
                    status_name=status_name,
                    fps=heartbeat.inference_fps,
                    latency_ms=heartbeat.inference_latency_ms,
                    memory_mb=heartbeat.memory_usage_mb,
                    tracks_count=heartbeat.active_tracks_count,
                    detections_since_last=heartbeat.total_detections_since_last,
                )

                if directives is None:
                    # Unauthenticated node
                    context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid session token")
                    return

                # Send any queued directives back to edge node
                for d in directives:
                    action_enum = getattr(fleet_control_pb2.CommandDirective, d.get("action", "ACTION_NOOP"), fleet_control_pb2.CommandDirective.ACTION_NOOP)
                    yield fleet_control_pb2.CommandDirective(
                        directive_id=d.get("directive_id", str(time.time())),
                        action=action_enum,
                        target_model_version=d.get("target_model_version", ""),
                        target_model_manifest_url=d.get("target_model_manifest_url", ""),
                        config_payload_json=d.get("config_payload_json", "{}"),
                    )

        except grpc.RpcError as rpc_err:
            logger.warning(f"[FleetServer] StreamControl RPC terminated: {rpc_err}")
        except Exception as e:
            logger.error(f"[FleetServer] StreamControl exception: {e}", exc_info=True)

    def StreamTelemetry(
        self,
        request_iterator: Iterator[fleet_control_pb2.TelemetryBatch],
        context: grpc.ServicerContext,
    ) -> Iterator[fleet_control_pb2.TelemetryAck]:
        try:
            for batch in request_iterator:
                if not self.node_manager.validate_session(batch.node_id, batch.session_token):
                    yield fleet_control_pb2.TelemetryAck(
                        batch_sequence_id=batch.batch_sequence_id,
                        success=False,
                        error_message="Invalid session credentials",
                        server_received_time_ms=int(time.time() * 1000),
                    )
                    continue

                ingest_res = self.telemetry_engine.ingest_batch(
                    node_id=batch.node_id,
                    batch_sequence_id=batch.batch_sequence_id,
                    events=batch.events,
                )

                yield fleet_control_pb2.TelemetryAck(
                    batch_sequence_id=batch.batch_sequence_id,
                    success=True,
                    error_message="",
                    server_received_time_ms=ingest_res["server_received_time_ms"],
                )
        except grpc.RpcError as rpc_err:
            logger.info(f"[FleetServer] StreamTelemetry RPC terminated cleanly: {rpc_err}")
        except Exception as e:
            logger.error(f"[FleetServer] StreamTelemetry error: {e}", exc_info=True)


class GoliathControlServer:
    """
    Production gRPC Control Plane Server with graceful lifecycle management.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 50051,
        max_workers: int = 32,
    ):
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.node_manager = NodeLifecycleManager()
        self.telemetry_engine = TelemetryIngestionEngine()
        self.model_registry = ModelRegistry()
        self._server = None

    def start(self):
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers),
            options=[
                ("grpc.max_receive_message_length", 32 * 1024 * 1024),  # 32MB max message
                ("grpc.max_send_message_length", 32 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
                ("grpc.http2.min_ping_interval_without_data_ms", 5000),
            ],
        )

        servicer = FleetControlServicer(
            node_manager=self.node_manager,
            telemetry_engine=self.telemetry_engine,
            model_registry=self.model_registry,
        )

        fleet_control_pb2_grpc.add_FleetControlServiceServicer_to_server(servicer, self._server)
        bind_address = f"{self.host}:{self.port}"
        self._server.add_insecure_port(bind_address)
        self._server.start()
        logger.info(f"[FleetServer] GOLIATH Control Plane gRPC Server active on {bind_address}")

    def stop(self, grace_period: float = 5.0):
        if self._server:
            logger.info(f"[FleetServer] Initiating graceful shutdown (grace_period={grace_period}s)...")
            self._server.stop(grace_period)
            logger.info("[FleetServer] Shutdown complete.")
