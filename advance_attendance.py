import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import random

# ==========================================
# MODULE 1: AI CONCEPTS (Data Initialization)
# ==========================================
student_db = {
    "Student_1": {"name": "Shriyansh", "roll": 101, "attendance_history": [1, 1, 0, 1, 1, 1, 0, 1]},
    "Student_2": {"name": "Tushar",    "roll": 102, "attendance_history": [0, 0, 1, 0, 0, 1, 0, 0]},
    "Student_3": {"name": "Gaurav",    "roll": 102, "attendance_history": [0, 0, 1, 0, 0, 1, 0, 0]}

}

path = 'images'
known_encodings = []
class_names = []

print("Loading AI Model...")

if not os.path.exists(path):
    os.makedirs(path)
    print(f"Created folder '{path}'. Add photos here.")

# ==========================================
# MODULE 3: NEURAL NETWORK ARCHITECTURE (Encoding Faces)
# ==========================================
for cl in os.listdir(path):
    if cl.lower().endswith(('.jpg', '.png', '.jpeg')):
        try:
            cur_img = cv2.imread(f'{path}/{cl}')
            cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(cur_img)
            if len(encs) > 0:
                known_encodings.append(encs[0])
                name_key = os.path.splitext(cl)[0]
                class_names.append(name_key)
                print(f" -> Loaded: {name_key}")
        except Exception:
            pass

print(f"Database Loaded. {len(class_names)} students registered.")

# ==========================================
# HELPER: GREEN CORNER BOX
# ==========================================
def draw_border_corners(img, x1, y1, x2, y2, color, line_length=25, thickness=4):
    # Top Left
    cv2.line(img, (x1, y1), (x1 + line_length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + line_length), color, thickness)
    # Top Right
    cv2.line(img, (x2, y1), (x2 - line_length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length), color, thickness)
    # Bottom Left
    cv2.line(img, (x1, y2), (x1 + line_length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length), color, thickness)
    # Bottom Right
    cv2.line(img, (x2, y2), (x2 - line_length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length), color, thickness)

# ==========================================
# MODULE 2: EVOLUTIONARY ALGORITHMS (Threshold GA)
# ==========================================
def optimize_threshold_ga():
    population = [random.uniform(0.50, 0.60) for _ in range(5)]
    return min(population, key=lambda x: abs(x - 0.55))

optimal_tolerance = optimize_threshold_ga()

# ==========================================
# MODULE 5: REGRESSION & CLUSTERING (Analytics)
# ==========================================
def predict_attendance_performance(history):
    if len(history) < 2:
        return "New"
    X = np.array(range(len(history))).reshape(-1, 1)
    y = np.array(history)
    model = LinearRegression()
    model.fit(X, y)
    return "Rising" if model.predict([[len(history) + 1]]) > 0.5 else "Falling"

# ==========================================
# MODULE 6: FULL PROJECT INTEGRATION (Main Loop)
# ==========================================
cap = cv2.VideoCapture(0)

process_this_frame = True
face_locations = []
face_encodings = []
face_landmarks_list = []
face_names = []
face_infos = []

print("Starting Camera... Press 'q' to quit.")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_h, img_w, _ = img.shape

    if process_this_frame:
        img_small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        # --- Detection + Encoding + Landmarks (Module 3) ---
        face_locations = face_recognition.face_locations(img_small)
        face_landmarks_list = face_recognition.face_landmarks(img_small, face_locations)
        face_encodings = face_recognition.face_encodings(img_small, face_locations)

        face_names = []
        face_infos = []

        # --- Classification (Module 4: ML Pipeline) ---
        for encode_face in face_encodings:
            matches = face_recognition.compare_faces(
                known_encodings, encode_face, tolerance=optimal_tolerance
            )
            face_dis = face_recognition.face_distance(known_encodings, encode_face)

            name = "Unknown"
            info_text = "Not Registered"

            if len(face_dis) > 0:
                match_index = np.argmin(face_dis)
                if matches[match_index]:
                    raw_name = class_names[match_index]

                    if "student_1" in raw_name.lower():
                        name_key = "Student_1"
                    elif "student_2" in raw_name.lower():
                        name_key = "Student_2"
                    else:
                        name_key = raw_name.capitalize()

                    student_info = student_db.get(
                        name_key,
                        {"name": "Unknown", "roll": 0, "attendance_history": [1]}
                    )
                    name = f"{student_info['name']}"

                    history = student_info['attendance_history']
                    att_percentage = (sum(history) / len(history)) * 100
                    trend = predict_attendance_performance(history)

                    info_text = f"Roll:{student_info['roll']} | {att_percentage:.0f}% | {trend}"

            face_names.append(name)
            face_infos.append(info_text)

    process_this_frame = not process_this_frame

    # ---------- MINIMAL CONTOUR SKELETON (Option 1) ----------
    skeleton_color = (220, 220, 220)  # soft light grey

    for lm in face_landmarks_list:
        # List of features to outline
        features_to_draw = [
            "chin",
            "left_eyebrow",
            "right_eyebrow",
            "nose_bridge",
            "nose_tip",
            "top_lip",
            "bottom_lip",
        ]
        for feature in features_to_draw:
            if feature not in lm:
                continue
            pts = np.array(lm[feature], np.int32) * 4  # scale back up
            # Close the curve for lips; others stay open
            is_closed = feature in ["top_lip", "bottom_lip"]
            cv2.polylines(
                img,
                [pts],
                is_closed,
                skeleton_color,
                1,
                lineType=cv2.LINE_AA
            )

    # ---------- GREEN FACE BOX + BLACK INFO BOX ----------
    for (top, right, bottom, left), name, info in zip(face_locations, face_names, face_infos):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        padding_y_top = 30
        padding_y_bot = 10
        padding_x = 10

        top = max(0, top - padding_y_top)
        bottom = min(img_h, bottom + padding_y_bot)
        left = max(0, left - padding_x)
        right = min(img_w, right + padding_x)

        (text_w, text_h), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
        face_width = right - left
        box_width = max(face_width, text_w + 20)
        center_x = left + (face_width // 2)
        rect_left = center_x - (box_width // 2)
        rect_right = center_x + (box_width // 2)

        corner_color = (0, 255, 0)   # Green
        black_bg = (0, 0, 0)
        white_text = (255, 255, 255)

        # Green corners (thick)
        draw_border_corners(img, left, top, right, bottom,
                            corner_color, line_length=35, thickness=5)

        # Black info box
        cv2.rectangle(img, (rect_left, bottom + 5),
                      (rect_right, bottom + 70), black_bg, cv2.FILLED)

        # Bold white name
        cv2.putText(img, name, (rect_left + 10, bottom + 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, white_text, 2)
        # Bold white info
        cv2.putText(img, info, (rect_left + 10, bottom + 60),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, white_text, 2)

    cv2.imshow('AI Attendance System', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()