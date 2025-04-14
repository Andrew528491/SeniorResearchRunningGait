# Program to download all of the training videos 
# And save them as mp4 files in 

from pytube import YouTube
import numpy as np
from moviepy import *

fileName = "tagged_videos.csv"
raw_data = open(fileName, "rt")
refined = np.loadtxt(raw_data, usecols=(0,1,2,3),skiprows=1, delimiter=",", dtype=str)
count = 0

for video in refined[:,0]:
    yt = YouTube('http://youtube.com/watch?v=' + video)
    yt.streams.filter(file_extension="mp4").get_highest_resolution().download(filename=refined[count, 3]+"_"+str(count+1)+".mp4",output_path="./trainingvidsraw")
    print(yt.title + " downloaded")
    count = count + 1