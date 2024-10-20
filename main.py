import cv2 as cv
from ultralytics import YOLO
import math
from sort import Sort
import numpy as np
import face_recognition as fr
import pickle
cap = cv.VideoCapture("summation/test4.mp4")
model = YOLO("yolov8n.pt")
tracker = Sort()  # Initialize the SORT tracker

classnames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
              'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
              'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
              'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
              'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
              'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
              'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
              'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
              'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']



output_file = 'output_video.avi'
fourcc = cv.VideoWriter_fourcc(*'XVID')
fps = 20.0
frame_width = int(cap.get(3))  #
frame_height = int(cap.get(4))  #
out = cv.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

file = open("encodefile.p", "rb")
knewandid = pickle.load(file)
file.close()
encodelistknew, clineIds = knewandid

while True:
    ret, frame = cap.read()
    if not ret:
        break
    face_locations = fr.face_locations(frame)
    encodecurframe=fr.face_encodings(frame,face_locations)
    for ecface , facelc in zip(encodecurframe,face_locations):
        matches=fr.compare_faces(encodelistknew,ecface)
        facedis = fr.face_distance(encodelistknew, ecface)
        indix=np.argmin(facedis)
        if matches[indix]:
            for (top, right, bottom, left) in face_locations:
                cv.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 2)
    results = model(frame)

    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            clss = classnames[cls]

            if clss == "person":
                detections.append([x1, y1, x2, y2, conf])  # Append detections for tracking

    if detections:
        detections = np.array(detections)  # Convert to NumPy array
        tracked_objects = tracker.update(detections)  # Update the tracker with detections
        for obj in tracked_objects:



            x1, y1, x2, y2, obj_id = map(int, obj)  # Extract coordinates and ID
            for (top, right, bottom, left) in face_locations:
                cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                if (x1 <= left and y1 <= top and x2 >= right and y2 >= bottom):
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  #
                    cv.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    out.write(frame)

    # cv.imshow("frame", frame)
    #
    # if cv.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
cv.destroyAllWindows()
