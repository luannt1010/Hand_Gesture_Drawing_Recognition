import cv2
import mediapipe as mp
import numpy as np
import time
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
canvas = None
prev_x, prev_y = 0, 0

def fingers_up(hand_landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
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

            # --- Vẽ khi chỉ giơ ngón trỏ ---
            if fingers[1] == 1 and all(f == 0 for i, f in enumerate(fingers) if i != 1):
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y
                cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 255, 255), 5)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = 0, 0

            # # --- Clear canvas ---
            # if fingers[1] == 1 and fingers[2] == 1:
            #     canvas = np.zeros((h, w, 3), np.uint8)

    # Gộp camera và canvas để hiển thị
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    combined = cv2.add(frame_bg, canvas_fg)

    cv2.imshow("Air Drawing", combined)

    key = cv2.waitKey(1) & 0xFF

    # Lưu ảnh với nền trắng, nét đen
    if key == ord('s'):
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        # ảnh trắng nét đen
        drawing_white = np.ones_like(gray) * 255
        drawing_white[gray > 20] = 0  # nơi có nét vẽ (màu trắng trên canvas đen) → chuyển thành nét đen
        drawing_white = cv2.cvtColor(drawing_white, cv2.COLOR_GRAY2BGR)
        filename = f"drawing_white_{int(time.time())}.png"
        cv2.imwrite(filename, drawing_white)
        print(f"Ảnh nền trắng, nét đen đã lưu: {os.path.abspath(filename)}")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
