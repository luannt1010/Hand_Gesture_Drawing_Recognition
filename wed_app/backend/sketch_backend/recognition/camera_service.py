import cv2
import mediapipe as mp
import numpy as np
import base64
import threading

camera_running = False
latest_frame_b64 = None  # hiển thị cho frontend
latest_canvas_b64 = None  # ảnh trắng nét đen để predict
cap = None
canvas = None
prev_x, prev_y = 0, 0

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

def fingers_up(hand_landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []
    # Ngón cái
    fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x else 0)
    # 4 ngón còn lại
    for id in range(1, 5):
        fingers.append(1 if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y else 0)
    return fingers

def camera_loop():
    global cap, camera_running, canvas, latest_frame_b64, latest_canvas_b64, prev_x, prev_y

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠️ Không mở được camera.")
        return

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if canvas is None:
            canvas = np.zeros((h, w, 3), np.uint8)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers = fingers_up(hand_landmarks)
                index_finger = hand_landmarks.landmark[8]
                x, y = int(index_finger.x * w), int(index_finger.y * h)

                # ✏️ Vẽ khi chỉ giơ ngón trỏ
                if fingers[1] == 1 and all(f == 0 for i, f in enumerate(fingers) if i != 1):
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = x, y
                    cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 255, 255), 20)
                    prev_x, prev_y = x, y
                else:
                    prev_x, prev_y = 0, 0

                # # 🧹 Xóa nét khi giơ 2 ngón
                # if fingers[1] == 1 and fingers[2] == 1:
                #     canvas[:] = 0

        # ✅ HIỂN THỊ CHO NGƯỜI DÙNG: camera + nét
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
        combined = cv2.add(frame_bg, canvas_fg)

        # Encode frame (camera + nét) để hiển thị
        _, buffer = cv2.imencode(".jpg", combined)
        latest_frame_b64 = base64.b64encode(buffer).decode("utf-8")

        # ✅ ẢNH NỀN TRẮNG NÉT ĐEN — DÙNG ĐỂ PREDICT
        drawing_white = np.ones_like(gray_canvas) * 255
        drawing_white[gray_canvas > 20] = 0
        drawing_white = cv2.cvtColor(drawing_white, cv2.COLOR_GRAY2BGR)
        _, buffer2 = cv2.imencode(".jpg", drawing_white)
        latest_canvas_b64 = base64.b64encode(buffer2).decode("utf-8")

    cap.release()
    print("🛑 Camera stopped.")

def start_camera():
    global camera_running
    if not camera_running:
        camera_running = True
        threading.Thread(target=camera_loop, daemon=True).start()
        print("🎥 Camera started.")

def stop_camera():
    global camera_running, cap
    camera_running = False
    if cap:
        cap.release()
    print("🛑 Camera stopped.")

def get_latest_frame_base64():
    global latest_frame_b64
    return latest_frame_b64

def get_latest_canvas_base64():
    global latest_canvas_b64
    return latest_canvas_b64

def clear_canvas():
    global canvas, latest_canvas_b64
    if canvas is not None:
        canvas[:] = 0
    latest_canvas_b64 = None
    print("🧹 Canvas cleared.")
