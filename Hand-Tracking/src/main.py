
from src.webcam_control.webcam import WebCamController 

def main():
    webcam = WebCamController()
    webcam.start_video_processing()

if __name__ == '__main__': 
    main()