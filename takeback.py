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


def add_transaction(customer_id, product_take_ids=None, product_returning_ids=None):
    if product_take_ids is None:
        product_take_ids = []
    if product_returning_ids is None:
        product_returning_ids = []

    # Example product prices
    product_prices = {
        'cell phone': 1.00,
        'banana': 0.50,
        'apple': 0.75,
        'pizza': 1.25,
        'hot dog': 1.50
    }

    # Create a dictionary to count final products
    final_product_counts = {}

    # Count taken products
    for product in product_take_ids:
        final_product_counts[product] = final_product_counts.get(product, 0) + 1

    # Subtract returned products
    for product in product_returning_ids:
        if product in final_product_counts:
            final_product_counts[product] -= 1
            if final_product_counts[product] <= 0:
                del final_product_counts[product]

    # Prepare the final products and total price calculation
    endproud_ids = list(final_product_counts.keys())
    total_price = sum(product_prices.get(product, 0) * final_product_counts[product] for product in endproud_ids)

    existing_transactions = []

    # Check if the file exists and is not empty
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, mode='r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)  # Read header
            for row in reader:
                existing_transactions.append(row)
    else:
        # If the file does not exist or is empty, create the header
        header = ['Customer ID', 'Products Taken', 'Products Returned', 'Final Products', 'Total Price']

    found = False
    for i, row in enumerate(existing_transactions):
        if row[0] == customer_id:
            found = True
            current_take_ids = row[1].split(', ') if row[1] else []
            current_return_ids = row[2].split(', ') if row[2] else []
            current_end_ids = row[3].split(', ') if row[3] else []

            # Update taken, returned, and final product counts
            for product in product_take_ids:
                current_take_ids.append(product)
                final_product_counts[product] = final_product_counts.get(product, 0) + 1

            for product in product_returning_ids:
                current_return_ids.append(product)
                if product in final_product_counts:
                    final_product_counts[product] -= 1
                    if final_product_counts[product] <= 0:
                        del final_product_counts[product]

            current_end_ids = list(final_product_counts.keys())

            # Calculate total price based on the updated current_end_ids
            total_price = sum(
                product_prices.get(product, 0) * final_product_counts[product] for product in current_end_ids)

            # Update the row with new data
            existing_transactions[i][1] = ', '.join(current_take_ids)
            existing_transactions[i][2] = ', '.join(current_return_ids)
            existing_transactions[i][3] = ', '.join(current_end_ids)
            existing_transactions[i][4] = f"{total_price:.2f}"  # Update total price
            break

    if not found:
        # If no existing transaction, create a new one
        existing_transactions.append([
            customer_id,
            ', '.join(product_take_ids),
            ', '.join(product_returning_ids),
            ', '.join(endproud_ids),
            f"{total_price:.2f}"
        ])

    # Write the updated transactions back to the CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(existing_transactions)

    print(f"Transaction added/updated: {customer_id}, Products Taken: {', '.join(product_take_ids)}, "
          f"Products Returned: {', '.join(product_returning_ids)}, Final Products: {', '.join(endproud_ids)}, "
          f"Total Price: {total_price:.2f}")
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
shelf_zones = [
    [(50, 50), (300, 300)],
    # [(75, 77),(229, 217)],  # المنطقة الأولى
    # [(312, 76),(450, 227)], # المنطقة الثانية
    # [(76, 307),(222, 451)], # المنطقة الرابعة
    # [(307, 307),(450, 451)], # المنطقة الخامسة
]

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
tack_phone_detected_ids = []
back_phone_detected_ids = []
tack_banana_detected_ids = []
back_banana_detected_ids = []
tack_apple_detected_ids = []
back_apple_detected_ids = []
tack_pizza_detected_ids = []
back_pizza_detected_ids = []
tack_hotdog_detected_ids = []
back_hotdog_detected_ids = []
def is_in_shelf_zone(x, y, zones):
    for i, zone in enumerate(zones):
        if zone[0][0] <= x <= zone[1][0] and zone[0][1] <= y <= zone[1][1]:
            return i + 1  # ترجع رقم المنطقة (1, 2, 3, ...)
    return None
phone_taken = False
banana_taken = False
apple_taken = False
pizza_taken = False
hotdog_taken = False
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
                                if len(detections_phone)>0 :
                                    detections_phone = np.array(detections_phone)  # Convert to NumPy array
                                    tracked_phone = tracker_phone.update(detections_phone)
                                    for objf in tracked_phone:
                                        x1, y1, x2, y2, obj_id_f = map(int, objf)
                                        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                                        # cv.putText(frame, f"ID: {obj_id_f}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX,0.5,(0, 165, 255), 2)
                                        center_x = (x1 + x2) // 2
                                        center_y = (y1 + y2) // 2
                                        shelf_zone_number = is_in_shelf_zone(center_x, center_y, shelf_zones)
                                        if (wrist_x > x1 and wrist_x < x2) and (wrist_y > y1 and wrist_y < y2):
                                            cv.putText(frame, 'Hand close to phone!', (50, 50),cv.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
                                        if (index_finger_x > x1 and index_finger_x < x2) and (
                                                index_finger_y > y1 and index_finger_y < y2):
                                            if shelf_zone_number is None:
                                                cv.putText(frame, f"id {obj_id}:Taking the phone!", (50, 100),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                                phone_taken = True
                                                if obj_id_f not in tack_phone_detected_ids:
                                                    add_transaction(clineIds[indix], ['cell phone'])
                                                    tack_phone_detected_ids.append(obj_id_f)
                                        if phone_taken and (wrist_x < x1 or wrist_x > x2 or wrist_y < y1 or wrist_y > y2):
                                            if shelf_zone_number:
                                                cv.putText(frame, f"id {obj_id}:Returning the banana!", (50, 150),
                                                            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                                phone_taken = False
                                                if obj_id_f not in back_phone_detected_ids:
                                                    add_transaction(clineIds[indix], product_returning_ids=['cell phone'])
                                                    back_phone_detected_ids.append(obj_id_f)

                                        # if obj_id_f not in was_detected_ids:
                                        #     add_transaction(clineIds[indix], ['cell phone'])
                                        #     was_detected_ids.append(obj_id_f)

                                if len(detections_banana)>0:
                                    detections_banana = np.array(detections_banana)
                                    tracked_banana = tracker_banana.update(detections_banana)
                                    for objb in tracked_banana:
                                        x1, y1, x2, y2, obj_id_b = map(int, objb)
                                        center_x = (x1 + x2) // 2
                                        center_y = (y1 + y2) // 2
                                        shelf_zone_number = is_in_shelf_zone(center_x, center_y, shelf_zones)
                                        if (wrist_x > x1 and wrist_x < x2) and (wrist_y > y1 and wrist_y < y2):
                                            cv.putText(frame, 'Hand close to banana!', (50, 50), cv.FONT_HERSHEY_SIMPLEX,
                                                       1, (0, 0, 255), 2)
                                        if (index_finger_x > x1 and index_finger_x < x2) and (
                                                index_finger_y > y1 and index_finger_y < y2):
                                            if shelf_zone_number is None:
                                                cv.putText(frame, f"id {obj_id}:Taking the banana!", (50, 100),
                                                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                                banana_taken= True
                                                if obj_id_f not in tack_banana_detected_ids:
                                                    add_transaction(clineIds[indix], ['banana'])
                                                    tack_banana_detected_ids.append(obj_id_f)
                                        if banana_taken and (
                                                wrist_x < x1 or wrist_x > x2 or wrist_y < y1 or wrist_y > y2):
                                            if shelf_zone_number:
                                                cv.putText(frame, f"id {obj_id}:Returning the banana!", (50, 150),
                                                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                                banana_taken = False
                                                if obj_id_f not in back_banana_detected_ids:
                                                    add_transaction(clineIds[indix],product_returning_ids=['banana'])
                                                    back_banana_detected_ids.append(obj_id_f)

                                        # if obj_id_b not in was_detected_ids:
                                        #     add_transaction(clineIds[indix], ['banana'])
                                        #     was_detected_ids.append(obj_id_b)

                                if len(detections_apple)>0:
                                    detections_apple = np.array(detections_apple)
                                    tracked_apple = tracker_apple.update(detections_apple)
                                    for obja in tracked_apple:
                                        x1, y1, x2, y2, obj_id_a = map(int, obja)
                                        center_x = (x1 + x2) // 2
                                        center_y = (y1 + y2) // 2
                                        shelf_zone_number = is_in_shelf_zone(center_x, center_y, shelf_zones)
                                        if (wrist_x > x1 and wrist_x < x2) and (wrist_y > y1 and wrist_y < y2):
                                            cv.putText(frame, 'Hand close to apple!', (50, 50), cv.FONT_HERSHEY_SIMPLEX,
                                                       1, (0, 0, 255), 2)
                                        if (index_finger_x > x1 and index_finger_x < x2) and (
                                                index_finger_y > y1 and index_finger_y < y2):
                                            if shelf_zone_number is None:
                                                cv.putText(frame, f"id {obj_id}:Taking the apple!", (50, 100),
                                                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                                apple_taken= True
                                                if obj_id_f not in tack_apple_detected_ids:
                                                    add_transaction(clineIds[indix], ['apple'])
                                                    tack_apple_detected_ids.append(obj_id_f)
                                        if apple_taken and (
                                                wrist_x < x1 or wrist_x > x2 or wrist_y < y1 or wrist_y > y2):
                                            if shelf_zone_number:
                                                cv.putText(frame, f"id {obj_id}:Returning the apple!", (50, 150),
                                                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                                apple_taken = False
                                                if obj_id_f not in back_apple_detected_ids:
                                                    add_transaction(clineIds[indix],
                                                                    product_returning_ids=['apple'])
                                                    back_apple_detected_ids.append(obj_id_f)

                                        # if obj_id_a not in was_detected_ids:
                                        #     add_transaction(clineIds[indix], ['apple'])
                                        #     was_detected_ids.append(obj_id_a)

                                if len(detections_pizza)>0:
                                    detections_pizza = np.array(detections_pizza)
                                    tracked_pizza = tracker_pizza.update(detections_pizza)
                                    for objp in tracked_pizza:
                                        x1, y1, x2, y2, obj_id_p = map(int, objp)
                                        center_x = (x1 + x2) // 2
                                        center_y = (y1 + y2) // 2
                                        shelf_zone_number = is_in_shelf_zone(center_x, center_y, shelf_zones)
                                        if (wrist_x > x1 and wrist_x < x2) and (wrist_y > y1 and wrist_y < y2):
                                            cv.putText(frame, 'Hand close to pizza!', (50, 50), cv.FONT_HERSHEY_SIMPLEX,
                                                       1, (0, 0, 255), 2)
                                        if (index_finger_x > x1 and index_finger_x < x2) and (
                                                index_finger_y > y1 and index_finger_y < y2):
                                            if shelf_zone_number is None:
                                                cv.putText(frame, f"id {obj_id}:Taking the pizza!", (50, 100),
                                                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                                pizza_taken= True
                                                if obj_id_f not in tack_pizza_detected_ids:
                                                    add_transaction(clineIds[indix], ['pizza'])
                                                    tack_pizza_detected_ids.append(obj_id_f)
                                        if pizza_taken and (
                                                wrist_x < x1 or wrist_x > x2 or wrist_y < y1 or wrist_y > y2):
                                            if shelf_zone_number:
                                                cv.putText(frame, f"id {obj_id}:Returning the pizza", (50, 150),
                                                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                                pizza_taken = False
                                                if obj_id_f not in back_pizza_detected_ids:
                                                    add_transaction(clineIds[indix],
                                                                    product_returning_ids=['pizza'])
                                                    back_pizza_detected_ids.append(obj_id_f)

                                        # if obj_id_p not in was_detected_ids:
                                        #     add_transaction(clineIds[indix], ['pizza'])
                                        #     was_detected_ids.append(obj_id_p)

                                if len(detections_hotdog)>0:
                                    detections_hotdog = np.array(detections_hotdog)
                                    tracked_hotdog = tracker_hotdog.update(detections_hotdog)
                                    for objh in tracked_hotdog:
                                        x1, y1, x2, y2, obj_id_h = map(int, objh)
                                        center_x = (x1 + x2) // 2
                                        center_y = (y1 + y2) // 2
                                        shelf_zone_number = is_in_shelf_zone(center_x, center_y, shelf_zones)
                                        if (wrist_x > x1 and wrist_x < x2) and (wrist_y > y1 and wrist_y < y2):
                                            cv.putText(frame, 'Hand close to hotdog!', (50, 50), cv.FONT_HERSHEY_SIMPLEX,
                                                       1, (0, 0, 255), 2)
                                        if (index_finger_x > x1 and index_finger_x < x2) and (
                                                index_finger_y > y1 and index_finger_y < y2):
                                            if shelf_zone_number is None:
                                                cv.putText(frame, f"id {obj_id}:Taking the hotdog!", (50, 100),
                                                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                                hotdog_taken= True
                                                if obj_id_f not in tack_hotdog_detected_ids:
                                                    add_transaction(clineIds[indix], ['hotdog'])
                                                    tack_hotdog_detected_ids.append(obj_id_f)
                                        if hotdog_taken and (
                                                wrist_x < x1 or wrist_x > x2 or wrist_y < y1 or wrist_y > y2):
                                            if shelf_zone_number:
                                                cv.putText(frame, f"id {obj_id}:Returning the hotdog!", (50, 150),
                                                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                                hotdog_taken = False
                                                if obj_id_f not in back_hotdog_detected_ids:
                                                    add_transaction(clineIds[indix],
                                                                    product_returning_ids=['hotdog'])
                                                    back_hotdog_detected_ids.append(obj_id_f)

                                        # if obj_id_h not in was_detected_ids:
                                        #     add_transaction(clineIds[indix], ['hot dog'])
                                        #     was_detected_ids.append(obj_id_h)

                                            ##############################



                else:
                    if (x1 <= left and y1 <= top and x2 >= right and y2 >= bottom):
                        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        print('send a message to security for going to this person')





    for zone in shelf_zones:
        cv.rectangle(frame, zone[0], zone[1], (0,175, 255), 2)
    # out.write(frame)
    cv.imshow("frame", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
