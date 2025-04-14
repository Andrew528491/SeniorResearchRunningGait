import numpy as np
import os
from moviepy.editor import VideoFileClip

directory = 'trainingvidsrefined/Good'

def drange(start, stop, step):
    while start < stop:
            yield start
            start += step

# This pulls up the csv of tagged videos
# It's not necessary for this code, but I wasn't sure of that when I started writing
# So I pulled it in anyway
fileName = "tagged_videos.csv"
raw_data = open(fileName, "rt")
refined = np.loadtxt(raw_data, usecols=(0,1,2,3),skiprows=1, delimiter=",", dtype=str)
count = 1

# Goes through each video in the directory of edited 10 second clips
for current_vid in os.listdir(directory):
    print(current_vid)
    video = VideoFileClip("./trainingvidsrefined/Good/"+str(current_vid))
    
    # This loop goes through each second of the 10 second clip
    for x in drange(0, int(video.duration), 0.75):
        # This if statement discards the first and last second of each clip
        if(x != 0 and (x <= (video.duration-0.75))):
            # Then this code creates a new subclip, of 0.5 second length
            video_temp = video.subclip(x, x+0.75)
            video_temp.write_videofile("Good_"+str(count)+"_split.webm", fps = 30)
            # Finally this code moves the newly created file into another folder
            os.rename("Good_"+str(count)+"_split.webm", "trainingvidssplit/Good/Good_"+str(count)+"_split.webm")
            count = count + 1
            
