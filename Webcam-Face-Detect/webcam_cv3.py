import numpy as np
import cv2
import sys
import logging as log
import datetime as dt
import tkinter as tk
from time import sleep
from PIL import Image, ImageTk
import os


#Set up GUI
window = tk.Tk()  #Makes main window
window.wm_title("Digital Microscope")
window.config(background="#FFFFFF")

#Graphics window
imageFrame = tk.Frame(window, width=600, height=500)
imageFrame.grid(row=0, column=0, padx=10, pady=2)

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0


while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    if cv2.waitKey(1) & 0xFF == ord('w'):
        for (x,y,w,h) in faces :
            crop_img = frame[y: y + h, x: x + w] # Crop from x, y, w, h -> 100, 200, 300, 400
            cv2.imwrite("face.jpg", crop_img)
            print("hello")
    
    # Display the resulting frame
    cv2.imshow('Video', frame)
    #print("hejsa")
    key = cv2.waitKey(1)
    
    if key == ord('c'):
        for (x,y,w,h) in faces :
            path = os.path.sep.join([output_dir, "{}.jpg".format(str(total).zfill(8))])
            crop_face = frame[y:y+h, x:x+w]
            cv2.imwrite(crop_face, )
            cv2.imwrite(r'C:\Users\Mathi\Documents\GitHub\CNN\Webcam-Face-Detect', crop_face)
            print("hello")
            #python webcam_cv3.py haarcascade_frontalface_default.xml

# When everything is done, release the capture
#show_frame() #Display
window.mainloop()  #Starts GUI

