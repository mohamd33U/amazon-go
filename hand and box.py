import cv2
import numpy as np
from ultralytics import YOLO
import math
from sort import Sort
import mediapipe as mp

tracker = Sort( max_age=5, min_hits=3, iou_threshold=0.5)
# تحميل نموذج YOLOv8
model = YOLO('models/yolov8n.pt')
# path="summation/sb.mp4"
patho=0
cap = cv2.VideoCapture(patho)  # التقاط الفيديو من الكاميرا
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

# قائمة المناطق المتعددة (10 مناطق)
shelf_zones = [
    [(50, 50), (300, 300)],
    # [(75, 77),(229, 217)],  # المنطقة الأولى
    # [(312, 76),(450, 227)], # المنطقة الثانية
    # [(76, 307),(222, 451)], # المنطقة الرابعة
    # [(307, 307),(450, 451)], # المنطقة الخامسة
]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
# الدالة للتحقق ما إذا كان المنتج داخل أي منطقة من مناطق الرفوف
def is_in_shelf_zone(x, y, zones):
    for i, zone in enumerate(zones):
        if zone[0][0] <= x <= zone[1][0] and zone[0][1] <= y <= zone[1][1]:
            return i + 1  # ترجع رقم المنطقة (1, 2, 3, ...)
    return None
product_taken = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # اكتشاف المنتجات باستخدام YOLO
    results = model(frame)
    detections=[]
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            clss = classnames[cls]

            if clss == "cell phone":
                detections.append([x1, y1, x2, y2, conf])
    if detections:
        detections = np.array(detections)  # Convert to NumPy array
        tracked_objects = tracker.update(detections)  # Update the tracker with detections
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)
            # حساب مركز التفاحة
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # التحقق مما إذا كانت التفاحة داخل أي منطقة من مناطق الرفوف
            shelf_zone_number = is_in_shelf_zone(center_x, center_y, shelf_zones)

            # if shelf_zone_number:
            #     cv2.putText(frame, f" id {obj_id}:Apple in shelf zone {shelf_zone_number}", (x1, y1 - 30),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # else:
            #     cv2.putText(frame, f"id {obj_id}:Apple taken ", (x1, y1 - 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_hands = hands.process(frame_rgb)
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    # Get the position of the wrist (0) and the index finger tip (8)
                    wrist_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1])
                    wrist_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame.shape[0])
                    index_finger_x = int(
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                    index_finger_y = int(
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

                    # Draw hand landmarks
                    for landmark in hand_landmarks.landmark:
                        cx = int(landmark.x * frame.shape[1])
                        cy = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                    if (wrist_x > x1 and wrist_x < x2) and (wrist_y > y1 and wrist_y < y2) :
                        cv2.putText(frame, 'Hand close to banana!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                   (0, 0, 255), 2)
                    if (index_finger_x > x1 and index_finger_x < x2) and (
                            index_finger_y > y1 and index_finger_y < y2) :
                        if shelf_zone_number is None:
                            cv2.putText(frame, f"id {obj_id}:Taking the banana!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
                            product_taken = True


                    # if product_taken and (wrist_x < x1 or wrist_x > x2 or wrist_y < y1 or wrist_y > y2):
                    #     cv2.putText(frame, f"id {obj_id}:Returning the banana!", (50, 150),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    #     product_taken = False

            # رسم كل مناطق الرفوف على الفيديو
    for zone in shelf_zones:
        cv2.rectangle(frame, zone[0], zone[1], (0,175, 255), 2)
        # cv2.imwrite("img.jpg", frame)
            # عرض الفيديو
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




































