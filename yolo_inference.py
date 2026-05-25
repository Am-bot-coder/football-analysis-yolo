from ultralytics import YOLO

model = YOLO(r"D:\DL\YOLO\Football Goal Analysis\models\best.pt")

results = model.predict(r"D:\DL\YOLO\Football Goal Analysis\input_videos\08fd33_4.mp4",save = True)
print(results[0])
print("="*50)
for box in results[0].boxes:
    print(box)