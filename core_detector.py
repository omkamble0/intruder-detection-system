import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO

from goliath_tracking.byte_tracker import ByteTracker
from goliath_tracking.trajectory_manager import TrajectoryAndZoneManager

class SecuritySystem:
    def __init__(self):
        # Initialize YOLOv8
        self.model = YOLO('yolov8n.pt') 
        self.zone_scale = 1.0       
        self.y_offset = 0
        self.x_offset = 0
        
        # Threat & Tracking State
        self.threat_mode = True 
        self.tracker = ByteTracker(track_thresh=0.5, low_conf_thresh=0.1, match_thresh=0.8)
        self.trajectory_manager = TrajectoryAndZoneManager()

    def update_calibration(self, zone_scale=100, y_offset=0, x_offset=0):
        self.zone_scale = zone_scale / 100.0
        self.y_offset = y_offset
        self.x_offset = x_offset

    def process_frame(self, frame):
        height, width, _ = frame.shape
        intruders_in_zone = 0
        total_people = 0

        # ==========================================
        # 1. CALCULATE DYNAMIC TARGETING ZONE
        # ==========================================
        center_x = int(width / 2) + self.x_offset
        center_y = int(height / 2) + self.y_offset

        base_box_h = int(height * 0.5)
        base_box_w = int(width * 0.5)
        final_box_h = int(base_box_h * self.zone_scale)
        final_box_w = int(base_box_w * self.zone_scale)

        zx1 = center_x - final_box_w // 2
        zy1 = center_y - final_box_h // 2
        zx2 = center_x + final_box_w // 2
        zy2 = center_y + final_box_h // 2

        # Update restricted zone in trajectory manager
        self.trajectory_manager.set_restricted_zone([
            (zx1, zy1), (zx2, zy1), (zx2, zy2), (zx1, zy2)
        ])

        # ==========================================
        # 2. RUN AI OBJECT DETECTION (YOLOv8)
        # ==========================================
        results = self.model(frame, verbose=False)
        raw_detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                # Person class = 0
                if cls_id == 0:
                    px1, py1, px2, py2 = map(float, box.xyxy[0])
                    raw_detections.append([px1, py1, px2, py2, conf, cls_id])

        # ==========================================
        # 3. RUN BYTETRACK MULTI-OBJECT TRACKING
        # ==========================================
        if len(raw_detections) > 0:
            dets_array = np.array(raw_detections, dtype=float)
        else:
            dets_array = np.empty((0, 6), dtype=float)

        active_tracks = self.tracker.update(dets_array)
        total_people = len(active_tracks)

        active_ids = []
        for track in active_tracks:
            track_id = track.track_id
            active_ids.append(track_id)
            tlbr = track.tlbr
            px1, py1, px2, py2 = map(int, tlbr)
            
            # Process trajectory & zone intrusion
            traj_data = self.trajectory_manager.process_track(
                track_id=track_id,
                class_id=0,
                tlbr=tlbr,
            )

            is_intruder = traj_data["in_restricted_zone"]
            speed = traj_data["speed"]
            pts = traj_data["trajectory_points"]

            # Draw trajectory polyline
            if len(pts) > 1:
                for i in range(1, len(pts)):
                    pt_a = (int(pts[i-1][0]), int(pts[i-1][1]))
                    pt_b = (int(pts[i][0]), int(pts[i][1]))
                    line_color = (0, 0, 255) if (is_intruder and self.threat_mode) else (0, 255, 200)
                    cv2.line(frame, pt_a, pt_b, line_color, 2)

            if self.threat_mode:
                if is_intruder:
                    intruders_in_zone += 1
                    # Red Threat Box
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                    cv2.putText(frame, f"TRACK #{track_id} [INTRUDER]", (px1, max(15, py1 - 10)), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, f"SPD: {speed:.1f}px/s | DWELL: {traj_data['dwell_time_ms']/1000:.1f}s", 
                                (px1, py2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
                else:
                    # Blue/Yellow Monitoring Box
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 200, 0), 1)
                    cv2.putText(frame, f"TRACK #{track_id} (CLEAR)", (px1, max(15, py1 - 10)), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 200, 0), 1, cv2.LINE_AA)
            else:
                # Passive Surveillance Mode (Green)
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                cv2.putText(frame, f"TRACK #{track_id}", (px1, max(15, py1 - 10)), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        self.trajectory_manager.cleanup_stale_tracks(active_ids)

        # ==========================================
        # 4. DRAW PROFESSIONAL SOC HUD OVERLAYS
        # ==========================================
        if self.threat_mode:
            # Active Yellow Restricted Zone
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 255, 255), 2)
            cv2.putText(frame, "RESTRICTED SECURITY ZONE", (zx1 + 10, zy1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            # Passive Cyan Zone (Dimmed)
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 200, 0), 1)
            
        cv2.circle(frame, (center_x, center_y), 4, (0, 255, 0), -1)

        # Semi-Transparent Data HUD Box
        overlay = frame.copy()
        box_x1, box_y1 = 15, 15
        box_x2, box_y2 = 420, 100
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (10, 10, 15), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (255, 200, 0), 1)

        # Write HUD metrics
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"GOLIATH IDS | TIME: {current_time}", (25, 38), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"ACTIVE TRACKS: {total_people} | ENGINE: BYTETRACK-v1.0", (25, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 200), 1, cv2.LINE_AA)
        
        if self.threat_mode:
            if intruders_in_zone > 0:
                cv2.putText(frame, f"STATUS: [CRITICAL] {intruders_in_zone} INTRUDER(S) IN ZONE", (25, 85), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
                # Red screen border flash
                red_overlay = frame.copy()
                cv2.rectangle(red_overlay, (0, 0), (width, height), (0, 0, 255), 6)
                frame = cv2.addWeighted(red_overlay, 0.3, frame, 0.7, 0)
            else:
                cv2.putText(frame, "STATUS: [SECURE] NO PERIMETER VIOLATIONS", (25, 85), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"STATUS: [PASSIVE MONITORING] {total_people} TARGETS", (25, 85), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)

        return frame