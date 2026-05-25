from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import sys
sys.path.append('../')
from utils import get_centre_of_bbox, get_width_bbox

class Tracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detection = []
        for i in range(0, len(frames), batch_size):
            detection_batch = self.model.predict(frames[i:i+batch_size], conf=0.2)
            detection += detection_batch
            
        return detection

    def get_object_tracks(self, frames,read_from_stub = False,stub_path = None):
        


        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                track_objects = pickle.load(f)
            return track_objects
        detections = self.detect_frames(frames)

        track_objects = {
            'player': [],
            'referee': [],
            'ball': []
        }

        for frame_num,detection in enumerate(detections):
            cls_names = detection.names
            
            cls_names_inv = {v:k for k,v in cls_names.items()}
            # Convert to Supervision Detection Format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #Convert Goalkeeper to Player
            for object_ind , class_ind in enumerate(detection_supervision.class_id):
                if cls_names[class_ind] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']
            
            
            #Track Players
            
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            track_objects['player'].append({})
            track_objects['referee'].append({})
            track_objects['ball'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_ide = frame_detection[3]
                track_id = frame_detection[4]

                if cls_ide == cls_names_inv['player']:
                    track_objects['player'][frame_num][track_id] = {"bbox": bbox}
                elif cls_ide == cls_names_inv['referee']:
                    track_objects['referee'][frame_num][track_id] = {"bbox": bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_ide = frame_detection[3]
                track_id = frame_detection[4]

                if cls_ide == cls_names_inv['ball']:
                    track_objects['ball'][frame_num][track_id] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(track_objects, f)


        return track_objects
    

    def draw_ellipse(self, frame, bbox, color = (0, 0, 255), track_id = None):
        y2 = int(bbox[3])
        x_centre, _ = get_centre_of_bbox(bbox)
        width = get_width_bbox(bbox)

        cv2.ellipse(frame,
                    center = (x_centre, y2),
                    axes = (int(width),int(0.35*width)),
                    angle = 0,
                    startAngle = -45,
                    endAngle = 235,
                    color = color,
                    thickness = 2,
                    lineType = cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_centre - rectangle_width // 2
        x2_rect = x_centre + rectangle_width // 2
        y1_rect = (y2 - rectangle_height//2)+15
        y2_rect = (y2 + rectangle_height//2)+15


        if track_id is not None:
            cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)
            x1_text =  x1_rect + 12
            if track_id >= 99:
                x1_text -= 10
            cv2.putText(frame, str(track_id), (x1_text, y1_rect + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)


        return frame
    
    
    

    def draw_annotations(self, frames, track_objects):
        annotated_frames = []
        for frame_num, frame in enumerate(frames):
           frame = frame.copy()

           player_tracks = track_objects['player'][frame_num]
           referee_tracks = track_objects['referee'][frame_num]
           ball_tracks = track_objects['ball'][frame_num]

           # Draw Player Tracks
           for track_id, player_info in player_tracks.items():
               frame = self.draw_ellipse(frame, player_info['bbox'], (0, 0, 255),track_id)


           for track_id, referee_info in referee_tracks.items():
                frame = self.draw_ellipse(frame, referee_info['bbox'], (0, 255, 255))

            


           annotated_frames.append(frame)
        


        return annotated_frames
