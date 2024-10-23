import cv2 as cv
from ultralytics import YOLO
import math
from sort import Sort
import numpy as np
import face_recognition as fr
import pickle
import mediapipe as mp
import csv
import os
#####
# File name
filename = 'transactions.csv'
# Create CSV file if it doesn't exist
if not os.path.isfile(filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the headers
        writer.writerow(["CustomerID", "ProductIDs", "TotalPrice"])
import csv

def add_transaction(customer_id, product_ids):
    product_prices = {
        'cell phone': 1.00,
        'banana': 0.50,
        'apple': 0.75,
        'pizza': 1.25,
        'hot dog': 1.50
    }

    existing_transactions = []

    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # قراءة العنوان
        for row in reader:
            existing_transactions.append(row)

    found = False
    for i, row in enumerate(existing_transactions):
        if row[0] == customer_id:
            found = True
            current_product_ids = row[1].split(', ')

            current_product_ids.extend(product_ids)

            total_price = sum(product_prices[product] for product in current_product_ids if product in product_prices)

            existing_transactions[i][1] = ', '.join(current_product_ids)  # تحديث قائمة المنتجات
            existing_transactions[i][2] = total_price  # تحديث السعر الإجمالي
            break

    if not found:
        total_price = sum(product_prices[product] for product in product_ids if product in product_prices)  # حساب السعر الإجمالي للسجل الجديد
        existing_transactions.append([customer_id, ', '.join(product_ids), total_price])

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(existing_transactions)

    print(f"Transaction added/updated: {customer_id}, Products: {', '.join(product_ids)}, Total Price: {total_price:.2f}")

###
cap = cv.VideoCapture(0)
model = YOLO("models/yolov8n.pt")
tracker = Sort()  # Initialize the SORT tracker
tracker_phone = Sort()
tracker_banana = Sort()
tracker_apple = Sort()
tracker_pizza = Sort()
tracker_hotdog = Sort()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
bowl_position = (100,200, 700, 700)  # Example coordinates for the bowl

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

output_file = 'output_video/output_video.avi'
fourcc = cv.VideoWriter_fourcc(*'XVID')
fps = 20.0
frame_width = int(cap.get(3))  #
frame_height = int(cap.get(4))  #
out = cv.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

file = open("encoding/encodefile.p", "rb")
knewandid = pickle.load(file)
file.close()
encodelistknew, clineIds = knewandid
was_detected_ids = []
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections_phone = []
    detections_banana = []
    detections_apple = []
    detections_pizza = []
    detections_hotdog = []
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            clss = classnames[cls]
            ##############################
            if clss == 'cell phone' and conf >= 0.30:
                detections_phone.append([x1, y1, x2, y2, conf])
            if clss == 'banana' and conf >= 0.30:
                detections_banana.append([x1, y1, x2, y2, conf])
            if clss == 'apple' and conf >= 0.30:
                detections_apple.append([x1, y1, x2, y2, conf])
            if clss == 'pizza' and conf >= 0.30:
                detections_pizza.append([x1, y1, x2, y2, conf])
            if clss == 'hot dog' and conf >= 0.30:
                detections_hotdog.append([x1, y1, x2, y2, conf])
            ##############################

            if clss == "person" and conf>=0.60:
                detections.append([x1, y1, x2, y2, conf])  # Append detections for tracking



    cv.rectangle(frame, (bowl_position[0], bowl_position[1]), (bowl_position[2], bowl_position[3]), (255, 0, 255), 2)
    cv.putText(frame, 'shopping cart', (bowl_position[0], bowl_position[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255),
                2)



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
                    # print(clineIds[indix])
                    # Ensure that left, top, right, bottom are defined within this scope
                    if (x1 <= left and y1 <= top and x2 >= right and y2 >= bottom):
                        cv.rectangle(frame, (left, top), (right, bottom), (0, 255,0), 2)
                        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255,0), 2)
                        cv.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        ##############################

                        if detections_phone:
                            detections_phone = np.array(detections_phone)  # Convert to NumPy array
                            tracked_phone = tracker_phone.update(detections_phone)
                            for objf in tracked_phone:
                                x1, y1, x2, y2, obj_id_f = map(int, objf)
                                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                                cv.putText(frame, f"ID: {obj_id_f}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                                           (0, 165, 255), 2)
                                if (x1 >= bowl_position[0] and y1 >= bowl_position[1] and x2 <= bowl_position[
                                    2] and y2 <= bowl_position[3]):
                                    cv.putText(frame, 'Phone inside shopping cart!', (50, 150), cv.FONT_HERSHEY_SIMPLEX,
                                               1, (255, 0, 0), 2)
                                    if obj_id_f not in was_detected_ids:
                                        add_transaction(clineIds[indix], ['cell phone'])
                                        was_detected_ids.append(obj_id_f)

                        if detections_banana:
                            detections_banana = np.array(detections_banana)
                            tracked_banana = tracker_banana.update(detections_banana)
                            for objb in tracked_banana:
                                x1, y1, x2, y2, obj_id_b = map(int, objb)
                                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                                cv.putText(frame, f"ID: {obj_id_b}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX,0.5,(0, 165, 255), 2)
                                if (x1 >= bowl_position[0] and y1 >= bowl_position[1] and x2 <=bowl_position[2] and y2 <= bowl_position[3]):
                                    cv.putText(frame, 'Banana inside shopping cart!', (50, 150),
                                    cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                                    if obj_id_b not in was_detected_ids:
                                        add_transaction(clineIds[indix], ['banana'])
                                        was_detected_ids.append(obj_id_b)

                        if detections_apple:
                            detections_apple = np.array(detections_apple)
                            tracked_apple = tracker_apple.update(detections_apple)
                            for obja in tracked_apple:
                                x1, y1, x2, y2, obj_id_a = map(int, obja)
                                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                                cv.putText(frame, f"ID: {obj_id_a}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                                if (x1 >= bowl_position[0] and y1 >= bowl_position[1] and x2 <=bowl_position[2] and y2 <= bowl_position[3]):
                                    cv.putText(frame, 'Apple inside shopping cart!', (50, 150),cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                                    if obj_id_a not in was_detected_ids:
                                        add_transaction(clineIds[indix], ['apple'])
                                        was_detected_ids.append(obj_id_a)

                        if detections_pizza:
                            detections_pizza = np.array(detections_pizza)
                            tracked_pizza = tracker_pizza.update(detections_pizza)
                            for objp in tracked_pizza:
                                x1, y1, x2, y2, obj_id_p = map(int, objp)
                                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                                cv.putText(frame, f"ID: {obj_id_p}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0, 165, 255), 2)
                                if (x1 >= bowl_position[0] and y1 >= bowl_position[1] and x2 <=bowl_position[2] and y2 <= bowl_position[3]):
                                    cv.putText(frame, 'Pizza inside shopping cart!', (50, 150),cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                                    if obj_id_p not in was_detected_ids:
                                        add_transaction(clineIds[indix], ['pizza'])
                                        was_detected_ids.append(obj_id_p)

                        if detections_hotdog:
                            detections_hotdog = np.array(detections_hotdog)
                            tracked_hotdog = tracker_hotdog.update(detections_hotdog)
                            for objh in tracked_hotdog:
                                x1, y1, x2, y2, obj_id_h = map(int, objh)
                                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                                cv.putText(frame, f"ID: {obj_id_h}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX,0.5,(0, 165, 255), 2)
                                if (x1 >= bowl_position[0] and y1 >= bowl_position[1] and x2 <=bowl_position[2] and y2 <= bowl_position[3]):
                                    cv.putText(frame, 'Hot Dog inside shopping cart!', (50, 150),cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                                    if obj_id_h not in was_detected_ids:
                                        add_transaction(clineIds[indix], ['hot dog'])
                                        was_detected_ids.append(obj_id_h)


                                    ##############################

                        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
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
                                    cv.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                                ##############################

                                # Check if hand is close to detected phones
                                for (x1, y1, x2, y2, conf) in detections_phone:
                                    if (wrist_x > x1 and wrist_x < x2) and (wrist_y > y1 and wrist_y < y2):
                                        cv.putText(frame, 'Hand close to cell phone!', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1,
                                                   (0, 0, 255), 2)
                                    if (index_finger_x > x1 and index_finger_x < x2) and (
                                            index_finger_y > y1 and index_finger_y < y2):
                                        cv.putText(frame, 'Taking the cell phone!', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1,
                                                   (0, 0, 255), 2)


                                for (x1, y1, x2, y2, conf) in detections_banana:
                                    if (wrist_x > x1 and wrist_x < x2) and (wrist_y > y1 and wrist_y < y2):
                                        cv.putText(frame, 'Hand close to banana!', (50, 50),cv.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
                                    if (index_finger_x > x1 and index_finger_x < x2) and (index_finger_y > y1 and index_finger_y < y2):
                                            cv.putText(frame, 'Taking the banana!', (50, 100), cv.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255), 2)


                                for (x1, y1, x2, y2, conf) in detections_apple:
                                    if (wrist_x > x1 and wrist_x < x2) and (wrist_y > y1 and wrist_y < y2):
                                        cv.putText(frame, 'Hand close to apple!', (50, 50),
                                                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                    if (index_finger_x > x1 and index_finger_x < x2) and (
                                            index_finger_y > y1 and index_finger_y < y2):
                                        cv.putText(frame, 'Taking the apple!', (50, 100), cv.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2)

                                for (x1, y1, x2, y2, conf) in detections_pizza:
                                    if (wrist_x > x1 and wrist_x < x2) and (wrist_y > y1 and wrist_y < y2):
                                        cv.putText(frame, 'Hand close to pizza!', (50, 50),
                                                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                    if (index_finger_x > x1 and index_finger_x < x2) and (
                                            index_finger_y > y1 and index_finger_y < y2):
                                        cv.putText(frame, 'Taking the pizza!', (50, 100),
                                                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                for (x1, y1, x2, y2, conf) in detections_hotdog:
                                    if (wrist_x > x1 and wrist_x < x2) and (wrist_y > y1 and wrist_y < y2):
                                        cv.putText(frame, 'Hand close to hotdog!', (50, 50),
                                                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                    if (index_finger_x > x1 and index_finger_x < x2) and (
                                            index_finger_y > y1 and index_finger_y < y2):
                                        cv.putText(frame, 'Taking the hotdog!', (50, 100),
                                                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                                        ##############################

                else:
                    if (x1 <= left and y1 <= top and x2 >= right and y2 >= bottom):
                        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        print('send a message to security for going to this person')




    # out.write(frame)

    cv.imshow("frame", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
print(was_detected_ids)
