'''
Testing program written by myself to apply a pose detection solution 
offered by MediaPipe to a stock video of a person running found on 
the StoryBooks database.
'''

import cv2 as cv
import mediapipe as mp
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp.solutions.pose

saved_data = {}
num = 1

cap = cv.VideoCapture('./trainingvidsrefined/Bad_20_edited.webm')
with pose.Pose(model_complexity=2, smooth_landmarks=True,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.8) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      break

    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    results = pose.process(image)
    
    mp_drawing.draw_landmarks(image, results.pose_landmarks, POSE_CONNECTIONS,
                               mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                               mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                )   
    cv.imshow('Mediapipe Feed', image)

    
    if(results.pose_landmarks != None):
        keypoints = []
        for data_point in results.pose_landmarks.landmark:
            keypoints.append({
                             'X': data_point.x,
                             'Y': data_point.y,
                             'Z': data_point.z
                             })
        saved_data[num] = keypoints
        num = num+1
        

    if cv.waitKey(10) & 0xFF == ord('q'):
           break
       
cap.release()
