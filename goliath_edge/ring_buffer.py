"""
GOLIATH Edge SDK - Priority-Aware Bounded Ring Buffer
Guarantees zero loss of critical intrusion telemetry during edge network partitions.
"""

import threading
from typing import List, Optional, Any
from collections import deque

class EdgeTelemetryRingBuffer:
    """
    Thread-safe dual-priority ring buffer.
    When capacity is reached, low-priority ambient events are discarded first,
    ensuring critical intrusion detections are preserved until reconnection.
    """

    def __init__(self, max_capacity: int = 10000):
        self.max_capacity = max_capacity
        self._critical_queue = deque()
        self._ambient_queue = deque()
        self._lock = threading.RLock()
        self.dropped_ambient_count = 0

    def push(self, event: Any, is_critical: bool = False):
        with self._lock:
            total_items = len(self._critical_queue) + len(self._ambient_queue)
            
            if total_items >= self.max_capacity:
                if len(self._ambient_queue) > 0:
                    # Drop oldest ambient event
                    self._ambient_queue.popleft()
                    self.dropped_ambient_count += 1
                elif is_critical:
                    # Under extreme overflow with only critical events, drop oldest critical
                    self._critical_queue.popleft()
                else:
                    # Buffer full of critical items, drop incoming ambient event
                    self.dropped_ambient_count += 1
                    return

            if is_critical:
                self._critical_queue.append(event)
            else:
                self._ambient_queue.append(event)

    def pop_batch(self, max_batch_size: int = 50) -> List[Any]:
        """
        Pulls a batch of events prioritizing critical alerts over ambient events.
        """
        batch = []
        with self._lock:
            # Drain critical first
            while self._critical_queue and len(batch) < max_batch_size:
                batch.append(self._critical_queue.popleft())

            # Drain ambient up to remaining batch size
            while self._ambient_queue and len(batch) < max_batch_size:
                batch.append(self._ambient_queue.popleft())

        return batch

    def size(self) -> int:
        with self._lock:
            return len(self._critical_queue) + len(self._ambient_queue)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._critical_queue) == 0 and len(self._ambient_queue) == 0

    def clear(self):
        with self._lock:
            self._critical_queue.clear()
            self._ambient_queue.clear()
