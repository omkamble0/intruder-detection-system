import os

# AI & Camera Settings
MODEL_WEIGHTS = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.5
CAMERA_INDEX = 0

# Database & Storage
DATABASE_NAME = "security_audit.db"
LOG_COOLDOWN_SECONDS = 5
# This is where the intruder face snapshots will be saved
SNAPSHOT_DIR = os.path.join("static", "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True) 

# Security Zone Coordinates
RESTRICTED_ZONE = [(100, 100), (540, 100), (540, 380), (100, 380)]
