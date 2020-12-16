import numpy as np
import cv2
import matplotlib
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing import image
import tkinter as tk
from time import sleep
from PIL import Image, ImageTk, ImageOps
import os

result_emotion = "ingenting"

#Set up GUI
window = tk.Tk()  #Makes main window
window.wm_title("Video")
window.config(background="#FFFFFF")

#Graphics window
imageFrame = tk.Frame(window, width=1400, height=1800)
imageFrame.grid(row=2, column=2, padx=50, pady=20)

window.geometry("1300x1150+300+300")

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

matplotlib.use('Agg')

v = tk.StringVar()


def predict_and_save_graph():
    #imports skal måske være i GUI?
    class_names = ['Angry',
                   'Disgust',
                   'Fear',
                   'Happy',
                   'Neutral',
                   'Sad',
                   'Surprise']

    #load den trænede model fra filsti
    loaded_model = tf.keras.models.load_model('save models')

    #load face.jpg
    test_image = image.load_img('webcam images/face.jpg', target_size = (48, 48))
    test_image = ImageOps.grayscale(test_image)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)

    #predict med loaded_model
    result = loaded_model.predict(test_image, steps=1)
    #result_emotion = kategori med højst prob.
    result_emotion = result.argmax(axis=-1)
    #result_emotion = class_names[result_emotion]
    result = result.tolist()
    result = result[0]
    tal = result_emotion
    #result_emotion = class_names[tal]
    class_names = np.array(['Angry',
                   'Disgust',
                   'Fear',
                   'Happy',
                   'Neutral',
                   'Sad',
                   'Surprise'])
    print(class_names[result_emotion])
    #lav (og gem) søjlediagram med predict results
    plt.title('Mood')
    plt.ylabel('Accuracy')
    fig = plt.bar(class_names, result)
    plt.savefig('webcam images/predictedGraph.png')
    plt.clf()
    v.set(class_names[result_emotion])

#Capture video frames

cap = cv2.VideoCapture(0)
mood = result_emotion
while not cap.isOpened():
    print('Unable to load camera.')
    sleep(5)
    cap = cv2.VideoCapture(0)
    pass
def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
            
    faces = faceCascade.detectMultiScale(
        cv2image,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(64, 64)
    )
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    
    if cv2.waitKey(1) & 0xFF == ord('w'):
        for (x,y,w,h) in faces :
            crop_img = frame[y: y + h, x: x + w] # Crop from x, y, w, h -> 100, 200, 300, 400
            
            
            cv2.imwrite("webcam images/face.jpg", crop_img)
            faceRoute = "webcam images/face.jpg"
            load = Image.open(faceRoute)
            load = load.resize((180, 180), Image.ANTIALIAS)
            render = ImageTk.PhotoImage(load)
            img = tk.Label(yellowFrame, image = render)
            img.image = render
            img.place(x=1, y=1)
            
            predict_and_save_graph()
            #cv2.imwrite("graph.jpg", crop_img)  # skal slettes når vi bliver trætte af den
            graphRoute = "webcam images/predictedGraph.png"
            load = Image.open(graphRoute)
            load = load.resize((420, 360), Image.ANTIALIAS)
            render = ImageTk.PhotoImage(load)
            img = tk.Label(redFrame, image = render)
            img.image = render
            img.place(x=1, y=1)
           
                      
    
    # Display the resulting frame
    cv2.imshow('window', frame)
    display2.imgtk = imgtk #Shows frame for display 2
    display2.configure(image=imgtk)
    window.after(10, show_frame)
    
    

    
display1 = tk.Label(imageFrame)
display1.grid(row=1, column=0, padx=10, pady=2)  #Display 1
 
blueFrame = tk.Frame(window, bg = "blue", borderwidth=3, relief="solid")
blueFrame.place(rely = 0.1, relwidth = 0.5, relheight = 0.8)

yellowFrame = tk.Frame(window, bg = "yellow", borderwidth=3, relief="solid")
yellowFrame.place(relx = 0.5, rely = 0.1, relwidth = 0.5, relheight = 0.4)

moodLabel = tk.Label(window, textvariable=v, bg = "yellow")
moodLabel.config(font=("Courier", 44))
moodLabel.place(relx = 0.84, rely = 0.2, anchor = "center")

redFrame = tk.Frame(window, bg = "red", borderwidth=3, relief="solid")
redFrame.place(relx = 0.5, rely = 0.5, relwidth = 0.5, relheight = 0.4)




display2 = tk.Label(blueFrame)
display2.grid(row=3, column=3) #Display 2

# Pauseknap virker ikke på nuværende tidspunkt, men var noget der skulle være en del af programet 
'''def pauseActivate(activated):
    if activated == True:
        activated = False
        print("falsk")
        return activated
    else:
        activated = True
        print("sandt")
        return activated
        

var = tk.IntVar()

Start=tk.Radiobutton(window, text="Start", variable = var, value = 1)
Start.grid(row=5,column=5)
Sluk=tk.Radiobutton(window, text="Sluk", variable = var, value = 2)
Sluk.grid(row=6,column=5)
'''
show_frame() #Display    
window.mainloop()  #Starts GUI