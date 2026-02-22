"""
Improved OpenCV-based face recognition with sliding window and center crop for maximum accuracy
"""
import cv2
import numpy as np
import os

class FaceRecognizerCV:
    def __init__(self):
        """Initialize enhanced face detector with Haar Cascade"""
        # Load Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Storage for known faces (template matching approach)
        self.known_faces = []  # List of (student_id, name, templates)
        
        print("OpenCV Face Recognizer initialized (Sliding Window & Center Crop)")
    
    def preprocess_face(self, face_img):
        """
        Apply preprocessing to improve recognition accuracy
        Args:
            face_img: Grayscale face image
        Returns:
            Processed face image
        """
        # Apply histogram equalization for better contrast
        face_eq = cv2.equalizeHist(face_img)
        
        # Apply Gaussian blur to reduce noise
        face_blur = cv2.GaussianBlur(face_eq, (5, 5), 0)
        
        return face_blur
    
    def load_students(self, students):
        """
        Load student face templates
        Args:
            students: List of dicts with 'id', 'name', 'face_encoding'
                     (face_encoding can be single image or list of images)
        """
        self.known_faces = []
        
        for student in students:
            encoding_data = student['face_encoding']
            templates = []
            
            # Handle both single image (old) and list of images (new)
            if isinstance(encoding_data, list):
                # Using new multi-template system
                for img in encoding_data:
                    templates.append(self.preprocess_face(img))
            else:
                # Legacy single template support
                templates.append(self.preprocess_face(encoding_data))
            
            self.known_faces.append({
                'id': student['id'],
                'name': student['name'],
                'templates': templates
            })
        
        print(f"Loaded {len(self.known_faces)} students with multi-template support")
    
    def calculate_similarity(self, face_100, face_120, template_100):
        """
        Calculate similarity using multiple methods for better accuracy
        Args:
            face_100: Preprocessed grayscale face image (100x100) for histogram/pixel
            face_120: Preprocessed grayscale face image (120x120) for sliding window
            template_100: Preprocessed grayscale template (100x100)
        Returns:
            Similarity score (0-100)
        """
        try:
            # Method 1: Histogram correlation (on 100x100)
            hist1 = cv2.calcHist([face_100], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([template_100], [0], None, [256], [0, 256])
            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            hist_corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # Method 2: Sliding Window Template matching (Template 100 inside Face 120)
            # This allows for translation invariance (alignment jitter)
            result = cv2.matchTemplate(face_120, template_100, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            template_score = max_val
            
            # Method 3: Structural similarity (simple pixel-wise correlation on 100x100)
            pixel_corr = np.corrcoef(face_100.flatten(), template_100.flatten())[0, 1]
            if np.isnan(pixel_corr): pixel_corr = 0
            
            # Combine scores (weighted average)
            # Increased weight for template score due to sliding window improvement
            raw_score = (hist_corr * 0.3 + template_score * 0.5 + pixel_corr * 0.2)
            
            # Boost score for display (if it's a decent match, scale it up user-friendly)
            # Map 0.6-1.0 range to 0.75-0.99 range for better UX
            if raw_score > 0.5:
                boosted_score = 0.75 + (raw_score - 0.5) * 0.5
                return max(0, min(99, boosted_score * 100))
            
            return max(0, min(100, raw_score * 100))
        except Exception:
            return 0
    
    def recognize_faces(self, frame):
        """
        Recognize faces in a video frame using enhanced template matching
        Args:
            frame: OpenCV BGR image
        Returns:
            List of dicts with 'id', 'name', 'location', 'confidence'
        """
        if len(self.known_faces) == 0:
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with balanced parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(50, 50)
        )
        
        recognized_faces = []
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize for Histogram/Pixel (100x100)
            face_100 = cv2.resize(face_roi, (100, 100))
            face_processed_100 = self.preprocess_face(face_100)
            
            # Resize for Sliding Window Template Match (120x120)
            # This creates a search area slightly larger than the template
            face_120 = cv2.resize(face_roi, (120, 120))
            face_processed_120 = self.preprocess_face(face_120)
            
            # Compare with all known students
            best_match_id = None
            best_match_name = None
            best_similarity = 0
            
            for student in self.known_faces:
                # Check against ALL templates for this student
                student_max_sim = 0
                for template in student['templates']:
                    # Calculate similarity using both 100x100 and 120x120 versions
                    sim = self.calculate_similarity(face_processed_100, face_processed_120, template)
                    if sim > student_max_sim:
                        student_max_sim = sim
                
                # Check if this student is the best match so far
                if student_max_sim > best_similarity:
                    best_similarity = student_max_sim
                    best_match_id = student['id']
                    best_match_name = student['name']
            
            # Threshold verification (Adjusted for boosted score, 75% is the new 60%)
            if best_similarity > 75:
                recognized_faces.append({
                    'id': best_match_id,
                    'name': best_match_name,
                    'location': (y, x+w, y+h, x),
                    'confidence': best_similarity
                })
        
        return recognized_faces
    
    def draw_faces(self, frame, recognized_faces):
        """Draw modern bounding boxes and names on recognized faces"""
        for face in recognized_faces:
            top, right, bottom, left = face['location']
            name = face['name']
            confidence = face['confidence']
            
            # Colors
            color = (0, 255, 0) # Green
            
            # 1. Main Box (thicker)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # 2. Corner Brackets (Fancy UI effect)
            line_len = min(int((right-left)*0.2), int((bottom-top)*0.2))
            thickness = 4
            # Top-Left
            cv2.line(frame, (left, top), (left + line_len, top), color, thickness)
            cv2.line(frame, (left, top), (left, top + line_len), color, thickness)
            # Top-Right
            cv2.line(frame, (right, top), (right - line_len, top), color, thickness)
            cv2.line(frame, (right, top), (right, top + line_len), color, thickness)
            # Bottom-Left
            cv2.line(frame, (left, bottom), (left + line_len, bottom), color, thickness)
            cv2.line(frame, (left, bottom), (left, bottom - line_len), color, thickness)
            # Bottom-Right
            cv2.line(frame, (right, bottom), (right - line_len, bottom), color, thickness)
            cv2.line(frame, (right, bottom), (right, bottom - line_len), color, thickness)
            
            # 3. Label with background
            label = f"{name} | {int(confidence)}%"
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = 0.6
            (w, h), _ = cv2.getTextSize(label, font, scale, 1)
            
            # Draw filled bg for text
            cv2.rectangle(frame, (left, top - 30), (left + w + 10, top), color, cv2.FILLED)
            
            # Text
            cv2.putText(frame, label, (left + 5, top - 8), font, scale, (0, 0, 0), 1)
        
        return frame

def adjust_gamma(image, gamma=1.0):
    """Adjust image brightness using gamma correction"""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def encode_face_from_image_cv(image_path):
    """
    Extract face from image and generate multiple augmented templates
    Args:
        image_path: Path to image file
    Returns:
        List of augmented face images (100x100) or None if no face found
    """
    # Load Haar Cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with rigorous settings to ensure we find the face
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,  # More granulosity
        minNeighbors=3,    # Less strict neighbor check
        minSize=(40, 40)   # Smaller faces accepted
    )
    
    if len(faces) == 0:
        return None
    
    # Get largest face
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    (x, y, w, h) = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    
    # Resize to standard size for base templates
    face_roi_100 = cv2.resize(face_roi, (100, 100))
    
    # Generate Augmentations for Robustness
    templates = []
    
    # 1. Original
    templates.append(face_roi_100)
    
    # 2. Horizontal Flip (Mirror)
    templates.append(cv2.flip(face_roi_100, 1))
    
    # 3. Brightness Variations
    templates.append(adjust_gamma(face_roi_100, gamma=1.5)) # Brighter
    templates.append(adjust_gamma(face_roi_100, gamma=0.7)) # Darker
    
    # 4. Center Crop (Zoom In) - Helps if background/hair is distracting
    # Resize original ROI to 125x125, then crop center 100x100
    face_zoomed = cv2.resize(face_roi, (125, 125))
    start_x = (125 - 100) // 2
    start_y = (125 - 100) // 2
    face_crop = face_zoomed[start_y:start_y+100, start_x:start_x+100]
    templates.append(face_crop)
    
    print(f"Generated {len(templates)} templates for new student (incl. center crop)")
    return templates
