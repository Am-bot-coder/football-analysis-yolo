from utils import read_video, save_video
from tracker import Tracker

def main():
    # Reading the video frames
    video_frames = read_video(r"D:\DL\YOLO\Football Goal Analysis\input_videos\08fd33_4.mp4")

    # Initializing the Tracker
    tracker = Tracker(r"D:\DL\YOLO\Football Goal Analysis\models\best.pt")
    tracks = tracker.get_object_tracks(video_frames,read_from_stub = True,stub_path = r"D:\DL\YOLO\Football Goal Analysis\stubs\tracks.pkl")
    
    # Draw Outputs
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Saving the video Frames 
    save_video(output_video_frames, r"D:\DL\YOLO\Football Goal Analysis\output_videos\output.avi")


if __name__ == '__main__':
    main()