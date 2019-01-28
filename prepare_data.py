from pytube import YouTube
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import numpy as np
import csv

raw_data = 'data/raw/'
detected_data = 'data/detected/'

def get_video(youtube_id):
    yt = YouTube('https://www.youtube.com/watch?v=' + youtube_id)
    name = yt.title
    yt.streams.first().download(filename='video')
    return cv2.VideoCapture('video.webm')

def get_frame_at(video, ms):
    video.set(cv2.CAP_PROP_POS_MSEC,(ms))     
    success, image = video.read()
    if success:
        cv2.imwrite('image.jpg', image)
        return True 
    return False

def save_image(name, rect_coords):
    raw_img = Image.open('image.jpg')
    raw_img.save(raw_data + name)

    det_img = Image.open('image.jpg')
    width, height = det_img.size

    draw = ImageDraw.Draw(det_img)
    draw.rectangle(((width*rect_coords[0], height*rect_coords[2]), (width*rect_coords[1], height*rect_coords[3])))
    det_img.save(detected_data + name)


dogs = [line.split(',') for line in open('dogs').read().splitlines()]


curr_id = ''
curr_ms = 0
video = None
images_downloaded = 0
metadata = []

failed_videos = set()
successful_video_download = False

for dog_data in dogs:
    if dog_data[0] != curr_id and (dog_data[0] not in failed_videos):
        try:
            print "Downloading video ", dog_data[0] 
            video = get_video(dog_data[0])
            curr_id = dog_data[0]
            successful_video_download = True
        except:
            print "Downloaling video failed"
            failed_videos.add(dog_data[0])
            successful_video_download = False
    elif int(dog_data[1]) - curr_ms <= 2000:
        continue
    
    if successful_video_download and dog_data[5] == 'present':
        curr_ms = int(dog_data[1])
        success = get_frame_at(video, int(dog_data[1]))
        if success:
            print "Saving image ms =", dog_data[1]
            coords = map(float, dog_data[6:])
            
            save_image(str(images_downloaded) + '.jpg', coords)
            metadata.append([images_downloaded] + coords)
            images_downloaded += 1

        if images_downloaded % 1 == 0:
            with open('metadata.csv', 'w+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerows(metadata)
