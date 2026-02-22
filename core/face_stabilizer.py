import time
from collections import Counter

class FaceStabilizer:
    def __init__(self, window_size=5, distance_threshold=80):
        """
        Initialize the Face Stabilizer
        Args:
            window_size: Number of frames to maintain for majority vote
            distance_threshold: Pixel distance to consider it the same "slot/track"
        """
        self.window_size = window_size
        self.distance_threshold = distance_threshold
        self.tracks = {} # {track_id: {'history': [], 'last_pos': (x,y,w,h), 'last_seen': time}}
        self.next_track_id = 0
        self.max_idle_time = 2.0 # Seconds before a track is removed
        
    def _calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between center points of two face rectangles"""
        c1 = (pos1[3] + (pos1[1]-pos1[3])/2, pos1[0] + (pos1[2]-pos1[0])/2) # Center of (top, right, bottom, left)
        c2 = (pos2[3] + (pos2[1]-pos2[3])/2, pos2[0] + (pos2[2]-pos2[0])/2)
        return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5

    def update(self, detected_faces):
        """
        Smooth and stabilize detected faces with name lock-in
        Args:
            detected_faces: List of dicts with 'name', 'location', 'confidence'
        Returns:
            Stabilized list of faces
        """
        now = time.time()
        stabilized_output = []
        
        # Clean up old tracks
        self.tracks = {tid: t for tid, t in self.tracks.items() if now - t['last_seen'] < self.max_idle_time}
        
        for face in detected_faces:
            loc = face['location'] # (top, right, bottom, left)
            name = face['name']
            conf = face['confidence']
            
            # Find best matching track
            best_track_id = None
            min_dist = float('inf')
            
            for tid, track in self.tracks.items():
                dist = self._calculate_distance(loc, track['last_pos'])
                if dist < self.distance_threshold and dist < min_dist:
                    min_dist = dist
                    best_track_id = tid
            
            if best_track_id is not None:
                # Update existing track
                # If name is 'Unknown', give it less weight in history if we already have a locked name
                track = self.tracks[best_track_id]
                
                # Logic: If we are already "locked" to a student, be very stubborn about changing it
                locked_name = track.get('locked_name')
                
                if locked_name and name == 'Unknown' and conf < 50:
                    # Ignore weak unknown detections if we have a lock
                    pass 
                else:
                    track['history'].append(name)
                    if len(track['history']) > self.window_size:
                        track['history'].pop(0)
                
                track['last_pos'] = loc
                track['last_seen'] = now
                
                # Check for Lock-in eligibility
                occurence_count = Counter(track['history'])
                most_common_name, count = occurence_count.most_common(1)[0]
                
                # Lock if the same name appears in 80% of the window
                if count >= (self.window_size * 0.8) and most_common_name != 'Unknown':
                    track['locked_name'] = most_common_name
                
                stable_name = track.get('locked_name', most_common_name)
            else:
                # Create new track
                best_track_id = self.next_track_id
                self.tracks[best_track_id] = {
                    'history': [name],
                    'last_pos': loc,
                    'last_seen': now,
                    'locked_name': None if name == 'Unknown' else name
                }
                self.next_track_id += 1
                stable_name = name
            
            # Copy face data and update with stable name
            stabilized_face = face.copy()
            stabilized_face['name'] = stable_name
            stabilized_output.append(stabilized_face)
            
        return stabilized_output
