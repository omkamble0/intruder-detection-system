"""
GOLIATH Multi-Object Tracking - Trajectory & Zone Violation Engine
Maintains smoothed motion trajectories, velocity vectors, and polygon zone intrusion detection.
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from shapely.geometry import Point, Polygon

class TrajectoryRecord:
    def __init__(self, track_id: int, class_id: int):
        self.track_id = track_id
        self.class_id = class_id
        self.points: List[Tuple[float, float, int]] = []  # (x, y, timestamp_ms)
        self.first_seen_ms: int = int(time.time() * 1000)
        self.last_seen_ms: int = self.first_seen_ms
        self.in_zone_duration_ms: int = 0
        self.has_triggered_alert: bool = False

    def add_point(self, x: float, y: float, timestamp_ms: Optional[int] = None):
        now_ms = timestamp_ms or int(time.time() * 1000)
        self.points.append((x, y, now_ms))
        self.last_seen_ms = now_ms

        # Keep max 100 history points to bound memory
        if len(self.points) > 100:
            self.points = self.points[-100:]

    @property
    def current_velocity(self) -> Tuple[float, float]:
        """Calculates instantaneous velocity in pixels/second."""
        if len(self.points) < 2:
            return (0.0, 0.0)
        x1, y1, t1 = self.points[-2]
        x2, y2, t2 = self.points[-1]
        dt = (t2 - t1) / 1000.0
        if dt <= 0:
            return (0.0, 0.0)
        return ((x2 - x1) / dt, (y2 - y1) / dt)


class TrajectoryAndZoneManager:
    """
    Monitors trajectories of active tracks against restricted polygon boundaries.
    """

    def __init__(self, restricted_polygon_coords: Optional[List[Tuple[float, float]]] = None):
        self._trajectories: Dict[int, TrajectoryRecord] = {}
        self.restricted_polygon: Optional[Polygon] = None
        if restricted_polygon_coords and len(restricted_polygon_coords) >= 3:
            self.restricted_polygon = Polygon(restricted_polygon_coords)

    def set_restricted_zone(self, coords: List[Tuple[float, float]]):
        if len(coords) >= 3:
            self.restricted_polygon = Polygon(coords)

    def process_track(
        self,
        track_id: int,
        class_id: int,
        tlbr: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Updates trajectory and checks for zone violation.
        tlbr: [x1, y1, x2, y2]
        """
        now_ms = timestamp_ms or int(time.time() * 1000)
        center_x = float((tlbr[0] + tlbr[2]) / 2.0)
        center_y = float((tlbr[1] + tlbr[3]) / 2.0)
        bottom_center = (center_x, float(tlbr[3]))  # Ground foot point for accurate zone containment

        if track_id not in self._trajectories:
            self._trajectories[track_id] = TrajectoryRecord(track_id, class_id)

        record = self._trajectories[track_id]
        record.add_point(center_x, center_y, now_ms)

        # Check zone violation
        is_in_zone = False
        if self.restricted_polygon:
            point = Point(bottom_center)
            is_in_zone = self.restricted_polygon.contains(point)

        if is_in_zone:
            record.in_zone_duration_ms += (now_ms - record.last_seen_ms)
        else:
            record.in_zone_duration_ms = 0

        vx, vy = record.current_velocity
        speed = float(np.sqrt(vx * vx + vy * vy))

        return {
            "track_id": track_id,
            "center": (center_x, center_y),
            "bottom_center": bottom_center,
            "velocity": (vx, vy),
            "speed": speed,
            "in_restricted_zone": is_in_zone,
            "dwell_time_ms": record.in_zone_duration_ms,
            "trajectory_points": [(p[0], p[1]) for p in record.points],
        }

    def cleanup_stale_tracks(self, active_track_ids: List[int], max_age_ms: int = 10000):
        """Purges tracks that have disappeared from view."""
        now_ms = int(time.time() * 1000)
        active_set = set(active_track_ids)
        stale_keys = [
            tid for tid, rec in self._trajectories.items()
            if tid not in active_set and (now_ms - rec.last_seen_ms) > max_age_ms
        ]
        for tid in stale_keys:
            del self._trajectories[tid]
