"""
GOLIATH Control Plane - Telemetry Ingestion Pipeline
High-throughput micro-batch ingestion engine with priority threat triage and metrics.
"""

import time
import logging
import threading
from typing import List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger("goliath.control.telemetry")

@dataclass
class ProcessedDetection:
    node_id: str
    timestamp_epoch_ms: int
    track_id: int
    class_name: str
    confidence: float
    bbox: List[float]
    threat_level: int
    in_restricted_zone: bool
    has_snapshot: bool


class TelemetryIngestionEngine:
    """
    Micro-batched stream ingestion engine designed for 50,000+ events/sec across 1,000+ edge nodes.
    Separates high-priority intrusion events from ambient telemetry to guarantee zero alert drop.
    """

    def __init__(self, max_buffer_size: int = 100000):
        self.max_buffer_size = max_buffer_size
        self._critical_events: List[ProcessedDetection] = []
        self._ambient_events: List[ProcessedDetection] = []
        self._lock = threading.Lock()
        
        # Telemetry Metrics
        self.total_batches_ingested = 0
        self.total_events_ingested = 0
        self.total_critical_intrusions = 0
        self.last_batch_time_epoch_ms = int(time.time() * 1000)

    def ingest_batch(self, node_id: str, batch_sequence_id: int, events: List[Any]) -> Dict[str, Any]:
        critical_count = 0
        ambient_count = 0
        now_ms = int(time.time() * 1000)

        processed_critical: List[ProcessedDetection] = []
        processed_ambient: List[ProcessedDetection] = []

        for evt in events:
            # ThreatLevel 2 = CRITICAL_INTRUSION, 1 = SUSPICIOUS, 0 = NONE
            is_critical = (evt.threat_level == 2) or evt.in_restricted_zone
            record = ProcessedDetection(
                node_id=node_id,
                timestamp_epoch_ms=evt.timestamp_epoch_ms,
                track_id=evt.track_id,
                class_name=evt.class_name,
                confidence=evt.confidence,
                bbox=list(evt.bbox),
                threat_level=evt.threat_level,
                in_restricted_zone=evt.in_restricted_zone,
                has_snapshot=len(evt.optional_snapshot_jpeg) > 0,
            )

            if is_critical:
                processed_critical.append(record)
                critical_count += 1
            else:
                processed_ambient.append(record)
                ambient_count += 1

        with self._lock:
            self._critical_events.extend(processed_critical)
            self._ambient_events.extend(processed_ambient)

            # Evict ambient if buffer overflows, NEVER drop critical
            if len(self._ambient_events) > self.max_buffer_size:
                evict_count = len(self._ambient_events) - self.max_buffer_size
                self._ambient_events = self._ambient_events[evict_count:]

            self.total_batches_ingested += 1
            self.total_events_ingested += (critical_count + ambient_count)
            self.total_critical_intrusions += critical_count
            self.last_batch_time_epoch_ms = now_ms

        if critical_count > 0:
            logger.warning(
                f"[TelemetryIngest] CRITICAL INTRUSION ALERT on node={node_id}: {critical_count} critical events!"
            )

        return {
            "success": True,
            "ingested_events": len(events),
            "critical_events": critical_count,
            "server_received_time_ms": now_ms,
        }

    def flush_critical_events(self) -> List[ProcessedDetection]:
        with self._lock:
            events = self._critical_events
            self._critical_events = []
            return events

    def get_ingest_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_batches_ingested": self.total_batches_ingested,
                "total_events_ingested": self.total_events_ingested,
                "total_critical_intrusions": self.total_critical_intrusions,
                "buffered_critical_count": len(self._critical_events),
                "buffered_ambient_count": len(self._ambient_events),
                "last_batch_epoch_ms": self.last_batch_time_epoch_ms,
            }
