import cv2
import sqlite3
import hashlib
import datetime
import os
import time
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
import settings

class SecurityLogger:
    def __init__(self):
        self.conn = sqlite3.connect(settings.DATABASE_NAME, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS logs 
                             (id INTEGER PRIMARY KEY, timestamp TEXT, event TEXT, integrity_hash TEXT, image_path TEXT, user_id INTEGER)''')
        self.conn.commit()

    def log_event(self, event_text, frame, coords, user_id):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        x1, y1, x2, y2 = map(int, coords)
        h, w = frame.shape[:2]
        cropped_img = frame[max(0, y1-20):min(h, y2+20), max(0, x1-20):min(w, x2+20)]
        
        filename = f"threat_{user_id}_{file_timestamp}.jpg"
        filepath = os.path.join(settings.SNAPSHOT_DIR, filename)
        cv2.imwrite(filepath, cropped_img)

        raw_data = f"{timestamp}-{event_text}-{filename}-{user_id}"
        secure_hash = hashlib.sha256(raw_data.encode()).hexdigest()
        
        self.cursor.execute("INSERT INTO logs (timestamp, event, integrity_hash, image_path, user_id) VALUES (?, ?, ?, ?, ?)", 
                           (timestamp, event_text, secure_hash, filename, user_id))
        self.conn.commit()
        print(f"[🚨 THREAT LOGGED] Image saved to Database for User ID: {user_id}")

class HeadlessWorker:
    def __init__(self, stream_url, user_id):
        self.stream_url = stream_url
        self.user_id = user_id
        print("[⚙️] Loading YOLOv8 AI Model...")
        self.model = YOLO(settings.MODEL_WEIGHTS)
        self.logger = SecurityLogger()
        self.zone_polygon = Polygon(settings.RESTRICTED_ZONE)
        self.last_alert_time = datetime.datetime.now()
        self.cooldown_seconds = settings.LOG_COOLDOWN_SECONDS

    def run(self):
        print(f"[📡] Connecting to Remote Camera Stream: {self.stream_url}")
        
        # Connect to the phone's IP camera (Add /video to the end of the IP Webcam URL)
        cap = cv2.VideoCapture(f"{self.stream_url}/video")
        
        if not cap.isOpened():
            print("[❌] ERROR: Could not connect to the stream. Check your IP address and Wi-Fi.")
            return

        print("[✅] Connection Established! AI analyzing frames in the background...")
        print("[⚡] Press Ctrl+C in this terminal to stop the worker.")

        while True:
            success, frame = cap.read()
            if not success:
                print("[!] Stream dropped. Retrying...")
                time.sleep(2)
                cap = cv2.VideoCapture(f"{self.stream_url}/video")
                continue

            clean_frame = frame.copy()
            # Run inference silently (verbose=False)
            results = self.model(frame, conf=settings.CONFIDENCE_THRESHOLD, verbose=False)

            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0: # 0 is the class for 'person'
                        coords = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = coords
                        center = Point((x1 + x2) / 2, (y1 + y2) / 2)

                        if self.zone_polygon.contains(center):
                            if (datetime.datetime.now() - self.last_alert_time).seconds >= self.cooldown_seconds:
                                self.logger.log_event("Intruder Detected", clean_frame, coords, self.user_id)
                                self.last_alert_time = datetime.datetime.now()

if __name__ == "__main__":
    print("=== ZERO-TRUST SOC: EDGE WORKER INITIALIZATION ===")
    
    # 1. Ask for the User ID (In a real production app, this is passed via a secure token)
    operative_id = input("Enter your authorized User ID (e.g., 1): ")
    
    # 2. Ask for the Phone Camera URL
    camera_ip = input("Enter your phone's IP Camera URL (e.g., http://192.168.1.5:8080): ")
    
    # Start the background worker
    worker = HeadlessWorker(stream_url=camera_ip, user_id=int(operative_id))
    worker.run()
    