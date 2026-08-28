"""
GOLIATH Control Plane - Node Lifecycle & State Manager
Tracks edge node states, heartbeats, and orchestrates command directives.
"""

import time
import logging
import uuid
import threading
from typing import Dict, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger("goliath.control.node_manager")

@dataclass
class EdgeNodeState:
    node_id: str
    cluster_id: str
    session_token: str
    firmware_version: str
    active_model_version: str
    hardware_arch: str
    accelerator: str
    total_ram_mb: int
    available_ram_mb: int
    compute_score: float
    tags: Dict[str, str] = field(default_factory=dict)
    status: str = "PROVISIONING"
    last_heartbeat_utc_ms: int = 0
    inference_fps: float = 0.0
    inference_latency_ms: float = 0.0
    memory_usage_mb: float = 0.0
    active_tracks_count: int = 0
    pending_directives: List[dict] = field(default_factory=list)


class NodeLifecycleManager:
    """
    Thread-safe fleet state manager maintaining real-time edge telemetry state,
    detecting offline/degraded nodes, and queuing asynchronous directives.
    """

    def __init__(self, heartbeat_timeout_seconds: float = 15.0):
        self._nodes: Dict[str, EdgeNodeState] = {}
        self._token_to_node_id: Dict[str, str] = {}
        self._lock = threading.RLock()
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds

    def register_node(
        self,
        node_id: str,
        cluster_id: str,
        firmware_version: str,
        active_model_version: str,
        hardware_arch: str,
        accelerator: str,
        total_ram_mb: int,
        available_ram_mb: int,
        compute_score: float,
        tags: Dict[str, str],
        assigned_model_version: str = "yolov8n-v1.0.0",
        manifest_url: str = "https://cdn.goliath-sec.internal/models/yolov8n-v1.0.0/manifest.json",
    ) -> EdgeNodeState:
        session_token = str(uuid.uuid4())
        now_ms = int(time.time() * 1000)

        with self._lock:
            # Invalidate old session token if re-registering
            if node_id in self._nodes:
                old_token = self._nodes[node_id].session_token
                self._token_to_node_id.pop(old_token, None)

            node = EdgeNodeState(
                node_id=node_id,
                cluster_id=cluster_id,
                session_token=session_token,
                firmware_version=firmware_version,
                active_model_version=active_model_version,
                hardware_arch=hardware_arch,
                accelerator=accelerator,
                total_ram_mb=total_ram_mb,
                available_ram_mb=available_ram_mb,
                compute_score=compute_score,
                tags=tags,
                status="HEALTHY",
                last_heartbeat_utc_ms=now_ms,
            )
            self._nodes[node_id] = node
            self._token_to_node_id[session_token] = node_id

            logger.info(
                f"[FleetControl] Registered node={node_id} (arch={hardware_arch}, acc={accelerator}) session={session_token}"
            )
            return node

    def validate_session(self, node_id: str, session_token: str) -> bool:
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return False
            return node.session_token == session_token

    def record_heartbeat(
        self,
        node_id: str,
        session_token: str,
        timestamp_utc_ms: int,
        status_name: str,
        fps: float,
        latency_ms: float,
        memory_mb: float,
        tracks_count: int,
        detections_since_last: int,
    ) -> Optional[List[dict]]:
        with self._lock:
            if not self.validate_session(node_id, session_token):
                logger.warning(f"[FleetControl] Rejected unauthenticated heartbeat for node={node_id}")
                return None

            node = self._nodes[node_id]
            node.last_heartbeat_utc_ms = timestamp_utc_ms or int(time.time() * 1000)
            node.status = status_name
            node.inference_fps = fps
            node.inference_latency_ms = latency_ms
            node.memory_usage_mb = memory_mb
            node.active_tracks_count = tracks_count

            # Pop pending directives
            directives = list(node.pending_directives)
            node.pending_directives.clear()
            return directives

    def queue_directive(self, node_id: str, directive: dict) -> bool:
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return False
            node.pending_directives.append(directive)
            logger.info(f"[FleetControl] Queued directive {directive.get('action')} for node={node_id}")
            return True

    def get_fleet_summary(self) -> dict:
        now_ms = int(time.time() * 1000)
        timeout_ms = self.heartbeat_timeout_seconds * 1000

        with self._lock:
            healthy = 0
            degraded = 0
            offline = 0
            total = len(self._nodes)

            for node in self._nodes.values():
                age_ms = now_ms - node.last_heartbeat_utc_ms
                if age_ms > timeout_ms:
                    node.status = "OFFLINE"
                    offline += 1
                elif node.status == "DEGRADED":
                    degraded += 1
                else:
                    healthy += 1

            return {
                "total_nodes": total,
                "healthy_nodes": healthy,
                "degraded_nodes": degraded,
                "offline_nodes": offline,
            }
