from flask import Flask, jsonify, render_template, Response, request, session, redirect, url_for, flash
from functools import wraps
import sqlite3
import cv2
import os
import numpy as np
import time
import hashlib
from core_detector import SecuritySystem
import settings

app = Flask(__name__)
# REQUIRED FOR SESSIONS: A cryptographic key to sign the browser cookies
app.secret_key = 'super_secret_enterprise_key_123' 

ai_system = SecuritySystem()
camera_active = False

# ==========================================
# DEPARTMENT 1: THE SECURITY GATEKEEPER
# ==========================================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            # If they don't have a session, bounce them back to the login screen
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Hash the typed password to compare it with the secure database hash
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        conn = sqlite3.connect(settings.DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND password_hash=?", (username, password_hash))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            # Grant the VIP Pass
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid Authentication Credentials.')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    # Destroy the VIP Pass
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('login'))

# ==========================================
# DEPARTMENT 2: DATABASE & VIDEO GENERATION
# ==========================================
def get_logs():
    try:
        conn = sqlite3.connect(settings.DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM logs ORDER BY id DESC")
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        return []

def generate_video_stream():
    global camera_active
    cap = None
    
    while True:
        if camera_active:
            if cap is None:
                cap = cv2.VideoCapture(settings.CAMERA_INDEX)
            
            success, frame = cap.read()
            if success:
                processed_frame = ai_system.process_frame(frame)
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            if cap is not None:
                cap.release()
                cap = None
            
            # Generate the "SYSTEM OFFLINE" screen
            standby_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(standby_frame, "SYSTEM OFFLINE", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            ret, buffer = cv2.imencode('.jpg', standby_frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1)

# ==========================================
# DEPARTMENT 3: PROTECTED API ROUTES
# ==========================================

@app.route('/')
@login_required
def dashboard():
    logs = get_logs()
    return render_template('index.html', logs=logs, total_logs=len(logs), 
                           current_cooldown=ai_system.cooldown_seconds,
                           camera_active=camera_active)

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/config', methods=['POST'])
@login_required
def update_config():
    data = request.json
    ai_system.cooldown_seconds = int(data.get('cooldown', ai_system.cooldown_seconds))
    return jsonify({"status": "success"})

@app.route('/api/camera', methods=['POST'])
@login_required
def toggle_camera():
    global camera_active
    camera_active = not camera_active
    status_text = "active" if camera_active else "inactive"
    print(f"📷 HARDWARE CONTROL: Camera is now {status_text.upper()}")
    return jsonify({"status": status_text})

@app.route('/api/logs/<int:log_id>', methods=['DELETE'])
@login_required
def delete_log(log_id):
    try:
        conn = sqlite3.connect(settings.DATABASE_NAME)
        cursor = conn.cursor()
        
        cursor.execute("SELECT image_path FROM logs WHERE id=?", (log_id,))
        result = cursor.fetchone()
        
        if result and result[0]:
            filepath = os.path.join(settings.SNAPSHOT_DIR, result[0])
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"🗑️ DELETED FILE: {filepath}")
                
        cursor.execute("DELETE FROM logs WHERE id=?", (log_id,))
        conn.commit()
        conn.close()
        
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Error deleting log: {e}")
        return jsonify({"status": "error"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)
    