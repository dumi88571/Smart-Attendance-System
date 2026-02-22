# Student Attendance System

Automatic attendance marking system using face recognition. Students are automatically marked present when detected by the camera.

## Features
- 📸 **Real-time Face Recognition** (95%+ confidence threshold)
- ✅ **Auto-Attendance Marking** (once per day per student)
- 👥 **Multi-student Detection** (classroom-ready)
- 💾 **SQLite Database** for students and attendance records
- 🌐 **Web Interface** for live monitoring and reports

## Quick Start

### 1. Install Dependencies
```powershell
cd attendance_system
py -3.11 -m pip install -r requirements.txt
```

### 2. Add Sample Students

Create student folders in `student_images/`:
```
student_images/
    student_1/
        name.txt          # Student name
        enrollment.txt    # Enrollment number
        face.jpg          # Face photo
    student_2/
        ...
```

Run import script:
```powershell
py -3.11 import_students.py
```

### 3. Run the Application
```powershell
py -3.11 app.py
```

**Open browser:** `http://127.0.0.1:5000`

## How It Works

1. Camera continuously scans for faces
2. When a known face is detected with >95% confidence:
   - ✅ **Automatically marks attendance** in database
   - 📝 Records timestamp
   - 🖥️ Shows name on live feed with green box
3. **Prevents duplicates**: Each student marked only once per day
4. Real-time attendance list updates automatically

## Usage

- **Live Attendance**: Homepage shows camera feed + today's attendance
- **Manage Students**: Add/view registered students
- **Reports**: View attendance history by date

## Technical Details

- **Face Detection**: OpenCV Haar Cascade (built-in)
- **Face Recognition**: LBPH (Local Binary Patterns Histograms)
- **Database**: SQLite with students and attendance tables
- **Framework**: Flask with live video streaming
- **Processing**: Every 3rd frame for optimal speed
- **Confidence Threshold**: 50% (LBPH inverse confidence scale)
- **No CMake Required**: Pure OpenCV solution
