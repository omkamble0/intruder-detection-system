# main.py
import os
import sys
import webbrowser
from threading import Timer
from flask import Flask, render_template, session, request, jsonify
from werkzeug.utils import secure_filename

# Ensure root directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(ROOT_DIR, "templates")
for p in [ROOT_DIR, TEMPLATES_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from cloud_db import supabase
from routes.camera_api import camera_bp
from goliath_control.server import GoliathControlServer
from goliath_drift.drift_detector import ProductionDriftDetector

app = Flask(__name__)
app.secret_key = 'super_secret_enterprise_key_123' 
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Register Camera Streaming Blueprint
app.register_blueprint(camera_bp)

# Global Enterprise Control Plane & Drift Detector
control_server = GoliathControlServer(host="0.0.0.0", port=50051)
drift_detector = ProductionDriftDetector()

@app.route('/')
def dashboard():
    """The main entry point for the UI."""
    if 'user_id' in session:
        try:
            response = supabase.table('activity_logs').select("*").eq('user_id', session['user_id']).order('timestamp', desc=True).limit(50).execute()
            user_logs = [(log['tool_used'], log['action_details'], log['timestamp'][:16].replace('T', ' ')) for log in response.data]
        except Exception:
            user_logs = []
        username = session.get('username', 'Authenticated Operative')
        logged_in = True
    else:
        user_logs = []
        username = "Guest Operative"
        logged_in = False
        
    return render_template('index.html', logs=user_logs, username=username, logged_in=logged_in)

@app.route('/upload_sim', methods=['POST'])
def upload_sim():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
    
    if file and file.filename.endswith('.mp4'):
        filename = "simulation.mp4" 
        file.save(os.path.join('static', filename))
        return jsonify({"status": "success", "message": "Simulation updated successfully!"})
    
    return jsonify({"status": "error", "message": "Invalid file type. Only MP4 allowed."}), 400

# ==========================================
# 🛰️ ENTERPRISE CONTROL PLANE APIs
# ==========================================
@app.route('/api/fleet/status', methods=['GET'])
def get_fleet_status():
    summary = control_server.node_manager.get_fleet_summary()
    telemetry_metrics = control_server.telemetry_engine.get_ingest_metrics()
    return jsonify({
        "control_plane": "ONLINE",
        "grpc_port": 50051,
        "fleet": summary,
        "telemetry": telemetry_metrics,
    })

@app.route('/api/mlops/drift', methods=['GET'])
def get_drift_status():
    report = drift_detector.evaluate_drift()
    return jsonify(report)

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    # Start background gRPC Control Plane Server
    try:
        control_server.start()
    except Exception as e:
        print(f"[WARN] Control server start: {e}")

    if os.environ.get("WERKZEUG_RUN_MAIN") is None:
        Timer(1.5, open_browser).start()
    app.run(debug=True, port=5000, threaded=True)