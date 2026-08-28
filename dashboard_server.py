from flask import Flask, jsonify, render_template, Response, request, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from functools import wraps
import cv2
import os
import numpy as np
import time
import hashlib
import math
import webbrowser
from threading import Timer
import joblib

# Local architecture modules
from core_detector import SecuritySystem
import settings
from supabase import create_client, Client

app = Flask(__name__)
app.secret_key = 'super_secret_enterprise_key_123' 

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ai_system = SecuritySystem()
camera_active = False
camera_source = 0 

# ==========================================
# ☁️ SUPABASE CLOUD CONNECTION
# ==========================================
SUPABASE_URL = "https://raghqovbiobywbjbykmj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJhZ2hxb3ZiaW9ieXdiamJ5a21qIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzkzNjg2MzUsImV4cCI6MjA5NDk0NDYzNX0.1keZ2Ku1nyVSga_XGiVUkvZ4qexYHM3vJnKznrE-_v4"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def log_activity(user_id, tool, details):
    """Logs actions to the cloud database if a valid user session exists."""
    if not user_id or user_id == "guest":
        return
    try:
        supabase.table('activity_logs').insert({
            "user_id": user_id, 
            "tool_used": tool, 
            "action_details": details
        }).execute()
    except Exception as e:
        print(f"Cloud Logging Error: {e}")

# ==========================================
# 🖥️ OPEN MAIN DASHBOARD (GUEST-ACCESSIBLE)
# ==========================================
@app.route('/')
def dashboard():
    # Detect if session exists, else initialize guest parameters
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
        
    return render_template('index.html', camera_active=camera_active, logs=user_logs, username=username, logged_in=logged_in)

# ==========================================
# 🔐 SYSTEM AUTHENTICATION PIPELINE
# ==========================================
@app.route('/auth', methods=['POST'])
def handle_auth():
    action = request.form.get('action') 
    email = request.form.get('username')
    password = request.form.get('password')
    
    if action == 'register':
        try:
            response = supabase.auth.sign_up({"email": email, "password": password})
            if response.user:
                session['user_id'] = response.user.id
                session['username'] = email
                flash('Clearance profile established successfully!', 'success')
            else:
                flash('Registration failed. Check details.', 'error')
        except Exception as e:
            flash(f'Auth Gateway Error: {str(e)}', 'error')

    elif action == 'login':
        try:
            response = supabase.auth.sign_in_with_password({"email": email, "password": password})
            if response.user:
                session['user_id'] = response.user.id
                session['username'] = email
                flash('Access granted. Profile sync complete.', 'success')
            else:
                flash('Verification failed.', 'error')
        except Exception as e:
            flash('Invalid credentials or rate limit exceeded.', 'error')
            
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('dashboard'))

# ==========================================
# 🌐 WEB THREAT INTELLIGENCE ENGINE
# ==========================================
@app.route('/api/scan_url', methods=['POST'])
def scan_url():
    data = request.json
    url = data.get('url', '')
    if not url:
        return jsonify({'status': 'error', 'message': 'No URL provided'})

    try:
        if not os.path.exists('url_vectorizer.pkl') or not os.path.exists('url_model.pkl'):
            return jsonify({'status': 'error', 'message': 'AI Models missing.'})

        vectorizer = joblib.load('url_vectorizer.pkl')
        model = joblib.load('url_model.pkl')

        X_input = vectorizer.transform([url])
        prediction = model.predict(X_input)[0]
        confidence = float(max(model.predict_proba(X_input)[0]))

        result_text = "PHISHING / MALICIOUS" if prediction == 1 else "SAFE / CLEAN"
        
        current_uid = session.get('user_id', 'guest')
        log_activity(current_uid, "Web Intelligence", f"Scanned URL: {url} | Result: {result_text}")

        return jsonify({'status': 'success', 'url': url, 'result': result_text, 'confidence': confidence})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# ==========================================
# 📷 CAMERA NODE INFRASTRUCTURE
# ==========================================
def generate_video_stream():
    global camera_active, camera_source
    cap = None
    while True:
        if camera_active:
            if cap is None:
                cap = cv2.VideoCapture(camera_source) 
            success, frame = cap.read()
            if success:
                processed_frame = ai_system.process_frame(frame)
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                cap.release()
                cap = None
                time.sleep(1)
        else:
            if cap is not None:
                cap.release()
                cap = None
            standby_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(standby_frame, "NODE OFFLINE", (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            ret, buffer = cv2.imencode('.jpg', standby_frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera', methods=['POST'])
def toggle_camera_api():
    global camera_active, camera_source
    data = request.json
    
    if data.get('action') == 'start':
        raw_source = data.get('url', '0')
        
        # Try to convert to an integer (for local webcams like '0' or '1')
        # If it fails, keep it as a string (for IP cameras like http://192.168.1.1/video)
        try:
            camera_source = int(raw_source)
        except ValueError:
            camera_source = raw_source

        # Set the AI Threat Mode based on UI Toggle
        ai_system.threat_mode = data.get('threat_mode', True)
        camera_active = True
        
        log_activity(session.get('user_id', 'guest'), "Surveillance Nexus", f"Activated Node: {camera_source}")
    elif data.get('action') == 'update_threat':
        # Live toggle update without restarting the camera
        ai_system.threat_mode = data.get('threat_mode', True)
    else:
        camera_active = False
        log_activity(session.get('user_id', 'guest'), "Surveillance Nexus", "Deactivated Node")
        
    return jsonify({"status": "active" if camera_active else "inactive"})

@app.route('/api/calibrate', methods=['POST'])
def calibrate_camera():
    data = request.json
    ai_system.update_calibration(
        zone_scale=data.get('zone', 100),
        y_offset=data.get('y_offset', 0),
        x_offset=data.get('x_offset', 0)
    )
    if data.get('save'):
        log_activity(session.get('user_id', 'guest'), "Surveillance Nexus", "Updated Matrix Calibration")
    return jsonify({"status": "calibrated"})

# ==========================================
# 🦠 MALWARE SANDBOX ENGINE
# ==========================================
def calculate_shannon_entropy(filepath):
    with open(filepath, 'rb') as f:
        byte_data = f.read()
    if not byte_data:
        return 0.0
    byte_counts = [0] * 256 
    for byte in byte_data:
        byte_counts[byte] += 1
    length = len(byte_data)
    entropy = 0
    for count in byte_counts:
        if count > 0:
            probability = count / length
            entropy -= probability * math.log(probability, 2)
    return round(entropy, 2)

@app.route('/api/scan_file', methods=['POST'])
def scan_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    file_hash = sha256_hash.hexdigest()
    entropy = calculate_shannon_entropy(filepath)
    
    threat_level = "CLEAN"
    malicious_score = 0
    suspicious_score = 0
    
    if entropy > 7.5:
        threat_level = "MALICIOUS"
        malicious_score = 1
    elif entropy > 7.0:
        threat_level = "SUSPICIOUS"
        suspicious_score = 1

    if any(filename.lower().endswith(ext) for ext in ['.exe', '.bat', '.cmd', '.vbs', '.ps1', '.sh']):
        suspicious_score += 1
        if threat_level == "CLEAN":
            threat_level = "SUSPICIOUS"

    stats = {"malicious": malicious_score, "suspicious": suspicious_score, "undetected": 1 if threat_level == "CLEAN" else 0, "entropy": entropy}
    log_activity(session.get('user_id', 'guest'), "Malware Sandbox", f"Scanned: {filename} | Result: {threat_level}")

    return jsonify({'status': 'success', 'stats': stats, 'filename': filename, 'hash': file_hash, 'threat_level': threat_level})

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    if os.environ.get("WERKZEUG_RUN_MAIN") is None:
        Timer(1.5, open_browser).start()
    app.run(debug=True, port=5000, threaded=True)