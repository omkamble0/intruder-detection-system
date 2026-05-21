import cv2
import sqlite3
import hashlib
import datetime
import os
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
import numpy as np
import settings

class SecurityLogger:
    def __init__(self):
        self.conn = sqlite3.connect(settings.DATABASE_NAME, check_same_thread=False)
        self.cursor = self.conn.cursor()
        # UPGRADE: Added 'image_path' to our database table
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS logs 
                             (id INTEGER PRIMARY KEY, timestamp TEXT, event TEXT, integrity_hash TEXT, image_path TEXT)''')
        self.conn.commit()

    def log_event(self, event_text, frame, coords):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # UPGRADE: Crop the intruder's image and save it
        x1, y1, x2, y2 = map(int, coords)
        # Add a buffer around the face/body, ensuring we don't go outside image bounds
        h, w = frame.shape[:2]
        cropped_img = frame[max(0, y1-20):min(h, y2+20), max(0, x1-20):min(w, x2+20)]
        
        filename = f"threat_{file_timestamp}.jpg"
        filepath = os.path.join(settings.SNAPSHOT_DIR, filename)
        cv2.imwrite(filepath, cropped_img)

        # Hash and Database insertion
        raw_data = f"{timestamp}-{event_text}-{filename}"
        secure_hash = hashlib.sha256(raw_data.encode()).hexdigest()
        self.cursor.execute("INSERT INTO logs (timestamp, event, integrity_hash, image_path) VALUES (?, ?, ?, ?)", 
                           (timestamp, event_text, secure_hash, filename))
        self.conn.commit()
        print(f"[SECURE LOG]: Saved Threat Image -> {filename}")

class SecuritySystem:
    def __init__(self):
        self.model = YOLO(settings.MODEL_WEIGHTS)
        self.logger = SecurityLogger()
        self.zone_coords = settings.RESTRICTED_ZONE
        self.zone_polygon = Polygon(self.zone_coords)
        self.last_alert_time = datetime.datetime.now()
        
        # UPGRADE: Dynamic Cooldown Variable
        self.cooldown_seconds = settings.LOG_COOLDOWN_SECONDS 

    def is_inside_zone(self, box):
        x1, y1, x2, y2 = box
        center_point = Point((x1 + x2) / 2, (y1 + y2) / 2)
        return self.zone_polygon.contains(center_point)

    def process_frame(self, frame):
        # We need a clean copy of the frame to save the image WITHOUT the lines drawn on it
        clean_frame = frame.copy() 
        results = self.model(frame, conf=settings.CONFIDENCE_THRESHOLD, verbose=False)
        
        pts = np.array(self.zone_coords, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

        for r in results:
            for box in r.boxes:
                coords = box.xyxy[0].tolist()
                if int(box.cls[0]) == 0: 
                    track_x = int((coords[0] + coords[2]) / 2)
                    track_y = int((coords[1] + coords[3]) / 2)
                    cv2.circle(frame, (track_x, track_y), 8, (255, 0, 0), -1) 

                    if self.is_inside_zone(coords):
                        cv2.rectangle(frame, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (0, 0, 255), 4)
                        cv2.putText(frame, "INTRUDER!", (int(coords[0]), int(coords[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        
                        # UPGRADE: Now uses the dynamic self.cooldown_seconds
                        if (datetime.datetime.now() - self.last_alert_time).seconds >= self.cooldown_seconds:
                            self.logger.log_event("Intruder Detected", clean_frame, coords)
                            self.last_alert_time = datetime.datetime.now()
                    else:
                        cv2.rectangle(frame, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (0, 255, 0), 2)
        return frame
    