import cv2
import mediapipe as mp
import numpy as np

# -------------------------------
# Initialize MediaPipe Hands
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# -------------------------------
# Start Webcam
# -------------------------------
cap = cv2.VideoCapture(0)

# -------------------------------
# Create Canvas
# -------------------------------
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Previous finger position
prev_x, prev_y = 0, 0

# -------------------------------
# Function to detect fingers
# -------------------------------
def fingers_up(hand_landmarks):
    fingers = []

    # Index finger
    fingers.append(
        hand_landmarks.landmark[8].y <
        hand_landmarks.landmark[6].y
    )

    # Middle finger
    fingers.append(
        hand_landmarks.landmark[12].y <
        hand_landmarks.landmark[10].y
    )

    return fingers

# -------------------------------
# Main Loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            fingers = fingers_up(hand)

            # Index finger tip position
            x = int(hand.landmark[8].x * w)
            y = int(hand.landmark[8].y * h)

            # -------------------------------
            # Write Mode
            # -------------------------------
            if fingers == [True, False]:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(
                    canvas,
                    (prev_x, prev_y),
                    (x, y),
                    (0, 0, 255),
                    5
                )
                prev_x, prev_y = x, y

            # -------------------------------
            # Erase Mode
            # -------------------------------
            elif fingers == [True, True]:
                cv2.circle(
                    canvas,
                    (x, y),
                    30,
                    (0, 0, 0),
                    -1
                )
                prev_x, prev_y = 0, 0

            else:
                prev_x, prev_y = 0, 0

    # Merge canvas with video frame
    frame = cv2.add(frame, canvas)

    # Display text instructions
    cv2.putText(
        frame,
        "Index Finger = Write | Two Fingers = Erase | Q = Exit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Air Writing System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()