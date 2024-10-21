import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from sort import Sort

tracker = Sort()  # Initialize the SORT tracker

# Load the YOLOv5 model (ultralytics version)
model = YOLO('models/yolov8n.pt')  # You can specify a different model if needed

# Set up MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Define the bowl's position (x1, y1, x2, y2)
bowl_position = (300, 300, 700, 700)  # Example coordinates for the bowl

# Open video capture
cap = cv2.VideoCapture("summation/phone.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5 object detection
    results = model(frame)

    # Get results
    detections = results[0]

    # Prepare detection data for tracking
    detection_data = []
    phones = []
    for detection in detections.boxes.data.numpy():
        # Check if the detected object is a phone (class id for 'cell phone' is 67 in COCO)
        if int(detection[5]) == 67:
            x1, y1, x2, y2, conf = map(int, detection[:5])
            phones.append((x1, y1, x2, y2))

            # Append detection for SORT tracker (format: [x1, y1, x2, y2, confidence])
            detection_data.append([x1, y1, x2, y2, conf])

            # Draw bounding box around the phone
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(frame, 'Phone', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Check if the phone is inside the bowl
            if (x1 >= bowl_position[0] and y1 >= bowl_position[1] and
                x2 <= bowl_position[2] and y2 <= bowl_position[3]):
                cv2.putText(frame, 'Phone inside bowl!', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Draw the bowl position
    cv2.rectangle(frame, (bowl_position[0], bowl_position[1]), (bowl_position[2], bowl_position[3]), (255, 0, 255), 2)
    cv2.putText(frame, 'Bowl', (bowl_position[0], bowl_position[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # Update the tracker with the new detections
    if detection_data:
        detections_tracked = tracker.update(np.array(detection_data))

        for d in detections_tracked:
            x1, y1, x2, y2, track_id = map(int, d)  # Extract the tracking ID
            # Draw the tracking ID on the frame
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Process hand detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # Check if hand is close to detected phones
            for (x1, y1, x2, y2) in phones:
                if (wrist_x > x1 and wrist_x < x2) and (wrist_y > y1 and wrist_y < y2):
                    cv2.putText(frame, 'Hand close to phone!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if (index_finger_x > x1 and index_finger_x < x2) and (index_finger_y > y1 and index_finger_y < y2):
                    cv2.putText(frame, 'Taking the phone!', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the final frame
    cv2.imshow('YOLOv5 and MediaPipe Hand Detection', frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
