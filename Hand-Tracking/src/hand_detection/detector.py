import mediapipe as mp
import numpy as np
import cv2

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


def detect_hand(img:np.array)->list:
    
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    hand = hands.process(imgRGB)    
    
    return hand.multi_hand_landmarks

def draw_landmarks(handsLms:list,img:np.array)->None:
    
    for handMark in handsLms:
        for id,lm in enumerate(handMark.landmark):
            height,width,chanels = img.shape
            cx,cy = int(lm.x*width),int(lm.y*height)
            
        mpDraw.draw_landmarks(img,handMark,mpHands.HAND_CONNECTIONS)