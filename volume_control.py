import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Set up hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

# Set up volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip video to act like a mirror
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Look for a hand
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw dots and lines on hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Find thumb and index finger
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            h, w, c = frame.shape
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

            # Measure distance between thumb and index
            distance = math.hypot(index_x - thumb_x, index_y - thumb_y)

            # Change volume based on distance
            vol = np.interp(distance, [30, 300], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(vol, None)

            # Draw line and dots for thumb and index
            cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 2)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 0, 255), -1)
            cv2.circle(frame, (index_x, index_y), 10, (0, 0, 255), -1)

            # Show volume percentage
            vol_percent = np.interp(vol, [min_vol, max_vol], [0, 100])
            cv2.putText(frame, f"Volume: {int(vol_percent)}%", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the video
    cv2.imshow("Volume Control", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()