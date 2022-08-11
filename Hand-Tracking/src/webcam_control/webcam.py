import cv2
import mediapipe as mp
from src.hand_detection.detector import detect_hand,draw_landmarks
import time

cap = cv2.VideoCapture(0)

pTime = 0
cTime = 0

def get_fps()->int:
    global pTime
    global cTime
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    return int(fps)

def start_video_processing(): 
    while True:
        success, img = cap.read()
        
    
        if success:
            handsMarks = detect_hand(img)
            if cv2.waitKey(1) == ord('q'):
                break 
        
            if handsMarks:
                draw_landmarks(handsMarks,img)
            
            
                
            cv2.putText(img,str(get_fps()),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)            
            cv2.imshow("Capture",img)
            
        

    
    cap.release()
    cv2.destroyAllWindows()