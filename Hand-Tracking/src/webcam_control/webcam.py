import cv2
import mediapipe as mp
from src.hand_detection.detector import HandDetector
import time


class WebCamController(object):
    
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.pTime = 0
        self.cTime = 0
        self.hand_detector = HandDetector()

    def get_fps(self)->int:
        """ Compute the current frame rate per second

        Returns:
            int: the frame rate per second  
        """        
        self.cTime = time.time()
        fps = 1/(self.cTime-self.pTime)
        self.pTime = self.cTime
        return int(fps)

    def start_video_processing(self)->None:
        """Initialize the video processing pipeline
        """         
        while True:
            success, img = self.cap.read()
            if success:
                handsMarks,_,_ = self.hand_detector.detect_hand(img)
                if cv2.waitKey(1) == ord('q'):
                    break 
                if handsMarks:
                    self.hand_detector.draw_landmarks(handsMarks,img)
                    
                cv2.putText(img,str(self.get_fps()),(10,70),
                            cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)            
                cv2.imshow("Capture",img)
                
        

    
        self.cap.release()
        cv2.destroyAllWindows()