"""
Database manager for student attendance system (supports SQLite and PostgreSQL)
"""
import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor
import pickle
import os
import json
from datetime import datetime
from urllib.parse import urlparse
from werkzeug.security import generate_password_hash, check_password_hash

class DatabaseManager:
    def __init__(self, db_path='database/attendance.db'):
        self.db_url = os.environ.get('DATABASE_URL')
        self.db_path = db_path
        
        if not self.db_url:
            # Local SQLite mode
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            self.use_postgres = False
        else:
            # Cloud PostgreSQL mode
            self.use_postgres = True
            
        self.init_database()
    
    def get_connection(self):
        """Create database connection based on mode"""
        if self.use_postgres:
            return psycopg2.connect(self.db_url)
        else:
            return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Initialize database with tables (generic SQL)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # SQLite vs PostgreSQL syntax differences
        auto_inc = "SERIAL PRIMARY KEY" if self.use_postgres else "INTEGER PRIMARY KEY AUTOINCREMENT"
        blob_type = "BYTEA" if self.use_postgres else "BLOB"
        ts_default = "CURRENT_TIMESTAMP"
        
        # Students table
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS students (
                id {auto_inc},
                name TEXT NOT NULL,
                enrollment_no TEXT UNIQUE NOT NULL,
                face_encoding {blob_type} NOT NULL,
                created_at TIMESTAMP DEFAULT {ts_default}
            )
        ''')
        
        # Attendance table
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS attendance (
                id {auto_inc},
                student_id INTEGER NOT NULL,
                marked_at TIMESTAMP DEFAULT {ts_default},
                session_date DATE,
                status TEXT DEFAULT 'present',
                FOREIGN KEY (student_id) REFERENCES students(id)
            )
        ''')
        
        # Stranger Log table (New Feature)
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS stranger_logs (
                id {auto_inc},
                snapshot_base64 TEXT,
                confidence FLOAT,
                detected_at TIMESTAMP DEFAULT {ts_default}
            )
        ''')

        # Admins table (New Feature)
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS admins (
                id {auto_inc},
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT {ts_default}
            )
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Database initialized successfully ({'PostgreSQL' if self.use_postgres else 'SQLite'})")
    
    def _execute(self, query, params=(), fetchone=False, fetchall=False):
        """Helper to run queries and handle connection lifecycle"""
        conn = self.get_connection()
        # For Postgres we want results as dicts if possible
        if self.use_postgres:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
        else:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
        try:
            # Postgres use %s, SQLite uses ?
            if not self.use_postgres:
                query = query.replace('%s', '?')
            
            cursor.execute(query, params)
            
            result = None
            if fetchone:
                result = cursor.fetchone()
            elif fetchall:
                result = cursor.fetchall()
            
            conn.commit()
            return result
        except Exception as e:
            print(f"Database Error: {e}")
            conn.rollback()
            return None
        finally:
            cursor.close()
            conn.close()

    def add_student(self, name, enrollment_no, face_encoding):
        """Add a new student with face encoding"""
        try:
            encoded_face = pickle.dumps(face_encoding)
            placeholder = '?' if not self.use_postgres else '%s'
            binary_data = encoded_face if not self.use_postgres else psycopg2.Binary(encoded_face)
            
            query = f"INSERT INTO students (name, enrollment_no, face_encoding) VALUES ({placeholder}, {placeholder}, {placeholder})"
            
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, (name, enrollment_no, binary_data))
            
            student_id = None
            if self.use_postgres:
                cursor.execute("SELECT LASTVAL()")
                student_id = cursor.fetchone()[0]
            else:
                student_id = cursor.lastrowid
                
            conn.commit()
            conn.close()
            return student_id
        except Exception as e:
            print(f"Error adding student: {e}")
            return None
    
    def get_all_students(self):
        """Retrieve all students"""
        rows = self._execute('SELECT id, name, enrollment_no, face_encoding FROM students', fetchall=True)
        
        students = []
        if rows:
            for row in rows:
                # Handle binary data difference
                face_encoding_blob = row['face_encoding']
                if hasattr(face_encoding_blob, 'tobytes'): # Memoryview in Postgres
                    face_encoding_blob = face_encoding_blob.tobytes()
                    
                face_encoding = pickle.loads(face_encoding_blob)
                students.append({
                    'id': row['id'],
                    'name': row['name'],
                    'enrollment_no': row['enrollment_no'],
                    'face_encoding': face_encoding
                })
        return students
    
    def mark_attendance(self, student_id, session_date=None):
        """Mark attendance for a student"""
        if session_date is None:
            session_date = datetime.now().date()
        
        # Check if already marked today
        exists = self._execute('SELECT id FROM attendance WHERE student_id = %s AND session_date = %s', 
                             (student_id, session_date), fetchone=True)
        
        if exists:
            return False
        
        now = datetime.now()
        self._execute('INSERT INTO attendance (student_id, session_date, status, marked_at) VALUES (%s, %s, %s, %s)',
                     (student_id, session_date, 'present', now))
        return True
    
    def log_stranger(self, snapshot_base64, confidence):
        """Log unknown face detection"""
        self._execute('INSERT INTO stranger_logs (snapshot_base64, confidence) VALUES (%s, %s)',
                     (snapshot_base64, confidence))

    def get_attendance_today(self):
        """Get list of students who attended today"""
        today = datetime.now().date()
        query = '''
            SELECT s.name, s.enrollment_no, a.marked_at
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE a.session_date = %s
            ORDER BY a.marked_at DESC
        '''
        rows = self._execute(query, (today,), fetchall=True)
        
        formatted_rows = []
        if rows:
            for row in rows:
                ts = row['marked_at']
                if isinstance(ts, str):
                    try:
                        if '.' in ts:
                            ts = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f')
                        else:
                            ts = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                    except:
                        pass
                
                time_str = ts.strftime('%I:%M %p') if isinstance(ts, datetime) else str(ts)
                iso_time = ts.isoformat() if isinstance(ts, datetime) else str(ts)
                
                formatted_rows.append({
                    'name': row['name'], 
                    'enrollment_no': row['enrollment_no'], 
                    'time': time_str,
                    'iso_time': iso_time
                })
        return formatted_rows
    
    def delete_student(self, student_id):
        """Delete student and their records"""
        # Delete attendance first
        self._execute('DELETE FROM attendance WHERE student_id = %s', (student_id,))
        # Delete student
        self._execute('DELETE FROM students WHERE id = %s', (student_id,))
        return True

    def get_attendance_report(self):
        """Get full attendance history with student details"""
        query = '''
            SELECT s.name, s.enrollment_no, a.session_date, a.marked_at
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            ORDER BY a.marked_at DESC
        '''
        rows = self._execute(query, fetchall=True)
        
        report = []
        if rows:
            for row in rows:
                ts = row['marked_at']
                if isinstance(ts, str):
                    try:
                        if '.' in ts:
                            ts = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f')
                        else:
                            ts = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                    except:
                        pass
                
                report.append({
                    'name': row['name'],
                    'enrollment_no': row['enrollment_no'],
                    'date': str(row['session_date']),
                    'time': ts.strftime('%I:%M %p') if isinstance(ts, datetime) else str(ts)
                })
        return report

    def sync_admin(self, username, password):
        """Create or update the primary admin account"""
        password_hash = generate_password_hash(password)
        
        # Check if any admin exists
        exists = self._execute('SELECT id FROM admins LIMIT 1', fetchone=True)
        
        if not exists:
            # Create first admin
            self._execute('INSERT INTO admins (username, password_hash) VALUES (%s, %s)', (username, password_hash))
            print(f"✨ Default admin created: {username}")
        else:
            # Update existing admin (sync with .env)
            self._execute('UPDATE admins SET username = %s, password_hash = %s WHERE id = (SELECT id FROM admins LIMIT 1)', 
                         (username, password_hash))
            print(f"🔄 Admin credentials synced with .env")

    def verify_admin(self, username, password):
        """Check if username and password match"""
        row = self._execute('SELECT password_hash FROM admins WHERE username = %s', (username,), fetchone=True)
        if row:
            return check_password_hash(row['password_hash'], password)
        return False
