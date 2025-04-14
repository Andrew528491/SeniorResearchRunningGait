import numpy as np
from moviepy.editor import VideoFileClip

fileName = "tagged_videos.csv"
raw_data = open(fileName, "rt")
refined = np.loadtxt(raw_data, usecols=(0,1,2,3),skiprows=1, delimiter=",", dtype=str)
count = 62

for x in range(40):
    video = VideoFileClip("./trainingvidsraw/Good_"+str(count+1)+".mp4").subclip(refined[count, 1], refined[count,2])
    video.write_videofile("Good_"+str(count+1)+"_edited.webm", fps = 30)
    count = count + 1
