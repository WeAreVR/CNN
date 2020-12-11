import numpy as np
import cv2
import Tkinter as tk
import Image, ImageTk

#Set up GUI
root = tk.Tk()  #Makes main window
root.wm_title("Digital Microscope")
root.config(background="#FFFFFF")

#Graphics window
imageFrame = tk.Frame(root, width=600, height=500)
imageFrame.grid(row=0, column=0, padx=10, pady=2)

#Capture video frames

cap = cv2.VideoCapture(0)

def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    display1.imgtk = imgtk #Shows frame for display 1
    display1.configure(image=imgtk)
    display2.imgtk = imgtk #Shows frame for display 2
    display2.configure(image=imgtk)
    root.after(10, show_frame)

display1 = tk.Label(imageFrame)
display1.grid(row=1, column=0, padx=10, pady=2)  #Display 1
display2 = tk.Label(imageFrame)
display2.grid(row=0, column=0) #Display 2

#Slider window (slider controls stage position)
sliderFrame = tk.Frame(root, width=600, height=100)
sliderFrame.grid(row = 600, column=0, padx=10, pady=2)


def autoActivated():
    buttonText = "nogetAndet"
    #messagebox.showinfo( "Hello Python", "Hello World")
    showStats()
    showWebCam()
    takePicture()

mood = "test1"
canvas = tk.Canvas(root, height = HEIGHT, width = WIDTH)
canvas.pack()


button = tk.Button(root, text = buttonText, command = autoActivated)
button.pack()

blueFrame = tk.Frame(root, bg = "blue")
blueFrame.place(rely = 0.1, relwidth = 0.5, relheight = 0.8)

lmain = tk.Label(blueFrame)
cap = cv2.VideoCapture(0)
def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


webCamLabel = tk.Label(blueFrame, text="WebCam her", bg="gray")
webCamLabel.place(relx = 0.5, rely = 0.5, anchor = "center")
webCamLabel.pack()

redFrame = tk.Frame(root, bg = "red")
redFrame.place(relx = 0.5, rely = 0.1, relwidth = 0.5, relheight = 0.4)


moodLabel = tk.Label(redFrame, text = mood, bg = "red")
moodLabel.place(relx = 0.5, rely = 0.5, anchor = "center")

yellowFrame = tk.Frame(root, bg = "yellow")
yellowFrame.place(relx = 0.5, rely = 0.43, relwidth = 0.5, relheight = 0.47)

statsLabel = tk.Label(yellowFrame, text = "stats her")
statsLabel.pack()
show_frame() #Display
root.mainloop()  #Starts GUI
