"""
Flask application for student attendance system with automatic marking
"""
import cv2
import os
import sys
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, session, redirect, url_for
from authlib.integrations.flask_client import OAuth
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'default_secret_dev_only')

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from core.face_recognizer import FaceRecognizerCV, encode_face_from_image_cv
from core.face_stabilizer import FaceStabilizer
from database.db_manager import DatabaseManager

# OAuth Configuration
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.environ.get('GOOGLE_CLIENT_ID'),
    client_secret=os.environ.get('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)

# Global instances
db_manager = DatabaseManager()
face_recognizer = FaceRecognizerCV()
face_stabilizer = FaceStabilizer(window_size=5) # Stabilize over last 5 frames

# Track which students have been marked today (to show confirmation only once)
marked_today = set()

# Admin Auth Decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def init_system():
    """Initialize the attendance system"""
    global face_recognizer, marked_today
    
    print("Initializing Student Attendance System...")
    
    # Sync Admin Credentials from .env
    admin_user = os.environ.get('ADMIN_USERNAME', 'admin')
    admin_pass = os.environ.get('ADMIN_PASSWORD', 'admin123')
    db_manager.sync_admin(admin_user, admin_pass)
    
    # Load all students from database
    students = db_manager.get_all_students()
    face_recognizer.load_students(students)
    
    # Reset daily tracking
    marked_today.clear()
    
    print(f"System ready with {len(students)} registered students")

# Removed local OpenCV AttendanceSystem in favor of Browser WebRTC + /process_frame

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if 'user' in session:
        return redirect(url_for('index'))
    
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if db_manager.verify_admin(username, password):
            session['user'] = {
                'name': username,
                'email': f"{username}@local",
                'picture': f"https://ui-avatars.com/api/?name={username}"
            }
            return redirect(url_for('index'))
        else:
            error = "Invalid username or password"
            
    return render_template('login.html', 
                           bypass_enabled=os.environ.get('BYPASS_AUTH') == 'true',
                           error=error)

@app.route('/login/test')
def login_test():
    """Bypass login for development/testing"""
    if os.environ.get('BYPASS_AUTH') == 'true':
        session['user'] = {
            'name': 'Test Admin',
            'email': 'admin@example.com',
            'picture': 'https://ui-avatars.com/api/?name=Test+Admin'
        }
        return redirect(url_for('index'))
    return redirect(url_for('login'))

@app.route('/login/google')
def login_google():
    """Redirect to Google for authentication"""
    redirect_uri = url_for('auth', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/auth')
def auth():
    """Handle Google authentication callback"""
    token = google.authorize_access_token()
    user = token.get('userinfo')
    if user:
        session['user'] = user
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    """Logout user"""
    session.pop('user', None)
    return redirect(url_for('index'))

@app.route('/')
def index():
    """Main attendance marking page (Public)"""
    students = db_manager.get_all_students()
    attendance_today = db_manager.get_attendance_today()
    
    return render_template('index.html', 
                          total_students=len(students),
                          present_today=len(attendance_today),
                          attendance_list=attendance_today,
                          user=session.get('user'))

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process a single frame from the browser camera"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data'}), 400
        
        # Decode base64 image
        import base64
        import numpy as np
        import cv2
        
        header, encoded = data['image'].split(",", 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        # Recognize faces
        raw_faces = face_recognizer.recognize_faces(frame)
        
        # Stabilize faces (Temporal smoothing)
        recognized_faces = face_stabilizer.update(raw_faces)
        
        marked_student = None
        
        # Auto-mark attendance for recognized faces
        for face in recognized_faces:
            student_id = face['id']
            student_name = face['name']
            confidence = face['confidence']
            
            # Threshold for attendance (same as before)
            if confidence > 75:
                if db_manager.mark_attendance(student_id):
                    marked_student = student_name
                    print(f"✅ Attendance marked: {student_name} (ID: {student_id})")
        
        # New Feature: Stranger Detection
        if not recognized_faces:
            # Check for generic faces to detect strangers
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_recognizer.face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                # Log a "stranger" if no known faces match
                db_manager.log_stranger(data['image'], 0) # 0 confidence = unknown
                print("🚨 Stranger detected!")

        # Format location for frontend (Top, Right, Bottom, Left)
        faces_output = []
        now_time = datetime.now().strftime('%H:%M:%S')
        for face in recognized_faces:
            faces_output.append({
                'name': face['name'],
                'location': face['location'],
                'confidence': face['confidence'],
                'time': now_time
            })
            
        return jsonify({
            'success': True, 
            'faces': faces_output,
            'marked_student': marked_student
        })
            
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/attendance/today')
def get_attendance_today():
    """Get today's attendance as JSON"""
    attendance = db_manager.get_attendance_today()
    return jsonify(attendance)

@app.route('/students')
@admin_required
def list_students():
    """List all registered students"""
    students = db_manager.get_all_students()
    # Don't send face encodings to frontend
    students_clean = [{'id': s['id'], 'name': s['name'], 'enrollment_no': s['enrollment_no']} 
                      for s in students]
    return render_template('students.html', students=students_clean, user=session.get('user'))

@app.route('/students/add', methods=['POST'])
@admin_required
def add_student():
    """Add a new student with face image"""
    try:
        name = request.form['name']
        enrollment_no = request.form['enrollment_no']
        
        # Get uploaded image
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        
        # Save temporarily
        temp_path = f'temp_{enrollment_no}.jpg'
        image_file.save(temp_path)
        
        # Extract face encoding
        face_encoding = encode_face_from_image_cv(temp_path)
        
        if face_encoding is None:
            os.remove(temp_path)
            return jsonify({'error': 'No face detected in image'}), 400
        
        # Add to database
        student_id = db_manager.add_student(name, enrollment_no, face_encoding)
        
        # Clean up
        os.remove(temp_path)
        
        if student_id:
            # Reload face recognizer
            init_system()
            return jsonify({'success': True, 'student_id': student_id})
        else:
            return jsonify({'error': 'Enrollment number already exists'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/students/delete/<int:student_id>', methods=['DELETE'])
@admin_required
def delete_student(student_id):
    """Delete a student and reload face recognizer"""
    try:
        success = db_manager.delete_student(student_id)
        
        if success:
            # Reload face recognizer
            init_system()
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Student not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analytics')
@admin_required
def analytics():
    """Analytics dashboard with charts"""
    return render_template('analytics.html')

@app.route('/api/stats')
@admin_required
def api_stats():
    """API for chart data"""
    try:
        # 1. Daily Trend (last 30 days)
        daily_query = '''
            SELECT session_date as date, COUNT(*) as count
            FROM attendance
            GROUP BY session_date
            ORDER BY session_date DESC
            LIMIT 30
        '''
        daily_data = db_manager._execute(daily_query, fetchall=True)
        # Convert to list if sqlite (psycopg2 returns list of dicts)
        daily_list = [dict(row) for row in daily_data] if daily_data else []
        
        # 2. Hourly Distribution
        hourly_query = '''
            SELECT 
                EXTRACT(HOUR FROM marked_at) as hour, 
                COUNT(*) as count
            FROM attendance
            GROUP BY hour
            ORDER BY hour
        ''' if db_manager.use_postgres else '''
            SELECT strftime('%H', marked_at) as hour, COUNT(*) as count
            FROM attendance
            GROUP BY hour
            ORDER BY hour
        '''
        hourly_data = db_manager._execute(hourly_query, fetchall=True)
        hourly_list = [dict(row) for row in hourly_data] if hourly_data else []
        
        return jsonify({
            'daily': daily_list,
            'hourly': hourly_list
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reports')
@admin_required
def reports():
    """Attendance reports page"""
    attendance_data = db_manager.get_attendance_report()
    return render_template('reports.html', attendance=attendance_data)

@app.route('/reports/export')
@admin_required
def export_reports():
    """Export attendance data as professional Excel file"""
    try:
        import pandas as pd
        import io
        from flask import make_response
        
        # Get standardized attendance data
        data = db_manager.get_attendance_report()
        df = pd.DataFrame(data)
        
        # Calculate summaries
        summary_query = '''
            SELECT s.name, s.enrollment_no, COUNT(a.id) as days_present
            FROM students s
            LEFT JOIN attendance a ON s.id = a.student_id
            GROUP BY s.id, s.name, s.enrollment_no
        '''
        summary_rows = db_manager._execute(summary_query, fetchall=True)
        df_summary = pd.DataFrame([dict(row) for row in summary_rows]) if summary_rows else pd.DataFrame()
        
        # Create Excel in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            if not df.empty:
                # Rename columns for professional look
                df.columns = [c.replace('_', ' ').title() for c in df.columns]
                df.to_excel(writer, index=False, sheet_name='Daily Attendance')
            else:
                # Still create sheet with headers even if empty
                pd.DataFrame(columns=['Name', 'Enrollment No', 'Date', 'Time']).to_excel(writer, index=False, sheet_name='Daily Attendance')
                
            if not df_summary.empty:
                df_summary.columns = [c.replace('_', ' ').title() for c in df_summary.columns]
                df_summary.to_excel(writer, index=False, sheet_name='Student Summary')
            else:
                pd.DataFrame(columns=['Name', 'Enrollment No', 'Days Present']).to_excel(writer, index=False, sheet_name='Student Summary')
        
        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = f"attachment; filename=attendance_report_{datetime.now().strftime('%Y%m%d')}.xlsx"
        response.headers["Content-type"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        return response
        
    except Exception as e:
        print(f"Excel Export error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/reports/export/csv')
@admin_required
def export_csv():
    """Export attendance data as CSV file"""
    try:
        import pandas as pd
        import io
        from flask import make_response
        
        # Get standardized attendance data
        data = db_manager.get_attendance_report()
        df = pd.DataFrame(data)
        
        if not df.empty:
            df.columns = [c.replace('_', ' ').title() for c in df.columns]
        else:
            df = pd.DataFrame(columns=['Name', 'Enrollment No', 'Date', 'Time'])
            
        output = io.StringIO()
        df.to_csv(output, index=False)
        
        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = f"attachment; filename=attendance_{datetime.now().strftime('%Y%m%d')}.csv"
        response.headers["Content-type"] = "text/csv"
        return response
        
    except Exception as e:
        print(f"CSV Export error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Initialize system
    init_system()
    
    # Get local IP
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "127.0.0.1"

    # Try to find the specific 192.168.x.x IP if possible
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        pass
    
    print("\n" + "="*60)
    print(" STUDENT ATTENDANCE SYSTEM")
    print("="*60)
    print(f" Registered students: {len(db_manager.get_all_students())}")
    print(f" Present today: {len(db_manager.get_attendance_today())}")
    print(" ACCESS URLs (New Port 8080):")
    print(f" 🏠 Local:      http://127.0.0.1:8080")
    
    # Print all available interfaces
    try:
        import socket
        hostname = socket.gethostname()
        print(f" 🔍 Hostname:   {hostname}")
        
        # Method 1: Get all IPs via getaddrinfo
        addresses = socket.getaddrinfo(hostname, None)
        seen_ips = set()
        for addr in addresses:
            ip = addr[4][0]
            if ip not in seen_ips and ':' not in ip and ip != '127.0.0.1': # IPv4 only, no loopback
                seen_ips.add(ip)
                print(f" 🌐 Network IP: http://{ip}:8080")
                
        # Method 2: Connect to external to find primary route (redundant but safe)
        if not seen_ips:
             s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
             try:
                 s.connect(("8.8.8.8", 80))
                 primary_ip = s.getsockname()[0]
                 print(f" 🌐 Primary IP: http://{primary_ip}:8080")
                 s.close()
             except:
                 pass
                 
    except Exception as e:
        print(f" ❌ IP Search Error: {e}")
        
    print("-" * 60)
    print(" Camera will auto-mark attendance when faces are recognized")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
