'''Extracts the motion-based data from the split videos and saves that
data into a csv file for ML training later'''

import numpy as np
import pandas as pd
import os
import cv2 as cv
import mediapipe as mp
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp.solutions.pose

final_data = []
saved_data = {}
count = 0

'''directory = 'trainingvidssplit/Bad'

for current_vid in os.listdir(directory):'''

current_vid = "./trainingvidsrefined/Bad/Bad_13_edited.webm"


pose = mp.solutions.pose
cap = cv.VideoCapture(current_vid)
num = 1
with pose.Pose(model_complexity=2, smooth_landmarks=True,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.8) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      break

    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    alpha = 0
    beta = 255
    brightness = 0
    cv.normalize(image, image, alpha+brightness, beta+brightness, cv.NORM_MINMAX)
    
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
'''
temp_list = ["Bad"]
for frame in saved_data:
    for joint in saved_data[frame]:
        temp_list.append(joint['X'])
        temp_list.append(joint['Y'])
        temp_list.append(joint['Z'])
final_data.append(temp_list)
count = count + 1

cap.release()
df = pd.DataFrame(final_data)
'''