from typing import List, Tuple
import mediapipe as mp
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
import numpy as np
import cv2

class HandDetector(object):
    
    def __init__(self,STATIC_IMAGE_MODE:bool=False,MAX_NUM_HANDS:int=2,
                 MODEL_COMPLEXITY:int=1,MIN_DETECTION_CONFIDENCE:float=0.5,
                 MIN_TRACKING_CONFIDENCE:float=0.5):
        """ A class representing the Hands Detector using
           mediapipe from google 

        Args:
            STATIC_IMAGE_MODE (bool): If input is single image we set it
                                to true, otherwise set false to track frames
            MAX_NUM_HANDS (int): Maximum number of hands in frame, default 2
            MODEL_COMPLEXITY (int): Two models 0 or 1 where 1 provides better
                                    results than 0
            MIN_DETECTION_CONFIDENCE (float): Detections confidence
            MIN_TRACKING_CONFIDENCE (float): If tracking frames, then tracking 
                                        confidence
        """        
        self.mode = STATIC_IMAGE_MODE
        self.num_hands = MAX_NUM_HANDS
        self.model = MODEL_COMPLEXITY
        self.min_detection_confidence = MIN_DETECTION_CONFIDENCE
        self.min_tracking_confidence = MIN_TRACKING_CONFIDENCE
        
        self.mpHands = mp.solutions.hands
        self.hands =  self.mpHands.Hands(static_image_mode=STATIC_IMAGE_MODE,
                                         max_num_hands=MAX_NUM_HANDS,
                                         model_complexity = MODEL_COMPLEXITY,
                                         min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                                         min_tracking_confidence = MIN_TRACKING_CONFIDENCE)
        
        self.mpDraw = mp.solutions.drawing_utils


    def detect_hand(self,img:np.array)->Tuple[list,list,list]:
        """Detects up until two hands on image

            this method uses the mediapipe library to detect hand
        Args:
            img (np.array): the image to be detected

        Returns:
            MULTI_HAND_LANDMARKS (list): Detection or tracked landmarks 
                                        as a list
            MULTI_HAND_WORLD_LANDMARKS (list): Real world 3D coordinates
            MULTI_HANDEDNESS (list): Handness detection as left or right 
                                    hand with score
        """        
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        hand = self.hands.process(imgRGB)    
        
        return hand.multi_hand_landmarks,\
               hand.multi_hand_world_landmarks,\
               hand.multi_handedness 

    def draw_landmarks(self,handsLms:list,img:np.array)->None:
        """
            Draw the handmarks on the specified image    

        Args:
            handsLms (list): List of hands that was be detected
            img (np.array): image to draw the handmarks on
        """        
        for handMark in handsLms:
            self.mpDraw.draw_landmarks(img,handMark,self.mpHands.HAND_CONNECTIONS)
            
    def get_positions_marks(self,imgShape:tuple,handMark:NormalizedLandmarkList)->list:
        """
           Returns a list of positions for a given hand marker

        Args:
            imgShape (tuple): image shape to calculate positions
            handMark (NormalizedLandmarkList): the handmark to get positions for

        Returns:
            list: all position off handmark
        """        
    
        height,width,chanels = imgShape
        listPositions = {}
        for id,lm in enumerate(handMark.landmark):
            cx,cy = int(lm.x*width),int(lm.y*height)
            listPositions[id] = {"cx":cx, "cy":cy}
        
        return listPositions
                        