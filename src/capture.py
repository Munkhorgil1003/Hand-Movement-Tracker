import cv2
import numpy as np
import mediapipe as mp
from screen_cleaner import clear

def run():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise RuntimeError("Could not open web cam")
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils
    
    canvas = None #for holding the persistent drwaing
    prev_x, prev_y = None, None #Last finger tip position
    drawing_enabled = False 
    was_drawing_enabled = False
    effect_frames_left = 0
    effect_center = None

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        
        if canvas is None:
            canvas = np.zeros_like(frame)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                lm = hand_landmarks.landmark
                
                index_up = lm[8].y < lm[6].y
                middle_up = lm[12].y < lm[10].y
                ring_up = lm[16].y < lm[14].y
                pinky_up = lm[20].y < lm[18].y
                
                pointing = index_up and not middle_up and not ring_up and not pinky_up
                showed_peace = index_up and middle_up and not ring_up and not pinky_up
                
                index_tip = lm[8]
                h, w, _ = frame.shape
                x = int(index_tip.x * w)
                y = int(index_tip.y * h)
                
                if showed_peace:
                    drawing_enabled = True
                    
                if drawing_enabled and not was_drawing_enabled:
                    effect_frames_left = 20
                    effect_center = (x, y)
                
                was_drawing_enabled = drawing_enabled                
                
                if drawing_enabled and pointing:
                    cv2.circle(frame, (x, y), 10, (0, 0, 255), cv2.FILLED)
                    if prev_x is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), 5)
                    prev_x, prev_y = x, y
                else:
                    prev_x, prev_y = None, None
        
        else:
            prev_x, prev_y = None, None
            
        if effect_frames_left > 0:
            progress = 1 - (effect_frames_left / 20)
            radius = int(20 * progress * 150)
            thickness = max(1, int(8 * (1 - progress)))
            cv2.circle(frame, effect_center, radius, (0, 255, 255), thickness)
            effect_frames_left -= 1
            
        combined = cv2.add(frame, canvas)
        cv2.imshow("Hand Tracker", combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            canvas = clear(canvas)
    
    cap.release()
    cv2.destroyAllWindows()