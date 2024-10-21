import cv2 as cv
from ultralytics import YOLO
import math
from sort import Sort
import numpy as np
import face_recognition as fr
import pickle
import mediapipe as mp

cap = cv.VideoCapture(0)
model = YOLO("models/yolov8n.pt")
tracker = Sort()  # Initialize the SORT tracker
tracker_phone = Sort()  # Initialize the SORT tracker
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
bowl_position = (300, 300, 700, 700)  # Example coordinates for the bowl

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

    results = model(frame)
    detections_phone = []
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            clss = classnames[cls]
            if clss=='cell phone':
                detections_phone.append([x1, y1, x2, y2, conf])
                if (x1 >= bowl_position[0] and y1 >= bowl_position[1] and
                        x2 <= bowl_position[2] and y2 <= bowl_position[3]):
                    cv.putText(frame, 'Phone inside bowl!', (50, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            if clss == "person":
                detections.append([x1, y1, x2, y2, conf])  # Append detections for tracking



    cv.rectangle(frame, (bowl_position[0], bowl_position[1]), (bowl_position[2], bowl_position[3]), (255, 0, 255), 2)
    cv.putText(frame, 'Bowl', (bowl_position[0], bowl_position[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255),
                2)
    if detections_phone:
        detections_phone = np.array(detections_phone)  # Convert to NumPy array
        tracked_phone = tracker_phone.update(detections_phone)
        for objf in tracked_phone:
            x1, y1, x2, y2, obj_id_f = map(int, objf)
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, f"ID: celll{obj_id_f}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results_hands = hands.process(frame_rgb)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            # Get the position of the wrist (0) and the index finger tip (8)
            wrist_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1])
            wrist_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame.shape[0])
            index_finger_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
            index_finger_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

            # Draw hand landmarks
            for landmark in hand_landmarks.landmark:
                cx = int(landmark.x * frame.shape[1])
                cy = int(landmark.y * frame.shape[0])
                cv.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # Check if hand is close to detected phones
            for (x1, y1, x2, y2, conf) in detections_phone:
                if (wrist_x > x1 and wrist_x < x2) and (wrist_y > y1 and wrist_y < y2):
                    cv.putText(frame, 'Hand close to phone!', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if (index_finger_x > x1 and index_finger_x < x2) and (index_finger_y > y1 and index_finger_y < y2):
                    cv.putText(frame, 'Taking the phone!', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if detections:
        detections = np.array(detections)  # Convert to NumPy array
        tracked_objects = tracker.update(detections)  # Update the tracker with detections
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)
            face_locations = fr.face_locations(frame)
            encodecurframe = fr.face_encodings(frame, face_locations)

            for ecface, facelc in zip(encodecurframe, face_locations):
                matches = fr.compare_faces(encodelistknew, ecface)
                facedis = fr.face_distance(encodelistknew, ecface)
                indix = np.argmin(facedis)

                # Unpacking the face location coordinates (top, right, bottom, left)
                top, right, bottom, left = facelc  # Make sure facelc is correctly unpacked here

                if matches[indix]:
                    # Ensure that left, top, right, bottom are defined within this scope
                    if (x1 <= left and y1 <= top and x2 >= right and y2 >= bottom):
                        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    if (x1 <= left and y1 <= top and x2 >= right and y2 >= bottom):
                        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    out.write(frame)

    cv.imshow("frame", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
