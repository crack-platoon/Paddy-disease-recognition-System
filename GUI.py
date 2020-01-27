from Tkinter import *
import tkFileDialog as dfl
import os
import sys

import cv2
import numpy as np
import random
import math
import scipy as sp
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from subprocess import call


window = Tk()
window.title('Paddy Desease Detection')
window.geometry("800x600")
window.resizable(0,0)
canvas = Canvas(window, width = 800, height = 600)
canvas.pack()

def update():
    with open("output.txt","r") as f:
        data = f.read()
        T.delete("1.0", "end")  # if you want to remove the old data
        T.insert(END,data)
    T.after(1000, update)
  
def fileOpen():
    file1 = dfl.askopenfilename()
    print file1
    input_file = file1
    file = open("file_name.txt","w+")
    file.write(file1)
    file.close()

def prediction():
    #os.system('python checker_Svm.py')
    #execfile('checker_Svm.py')
    call(["python", "checker_Svm.py"])

def train():
    call(["python", "test.py"])

f = open("output.txt","r+")
li = f.readlines(1)
li1 = f.readlines(2)
data = f.read()

print li
#print li1
    
#Label(canvas, text=li[0],fg="red",bg="yellow",font=("Helvetica",16)).pack()
#Label(canvas, text=li[1],fg="red",bg="yellow",font=("Helvetica",16)).pack()
T = Text(canvas, height=5, width=30,bg='yellow',fg='red',font=('Helvetica',16))
T.insert(END,data)
T.pack()

#update()
    

button = Button(canvas,text = "Choose Image", width=50,height=5,bg='black',fg='yellow', command= fileOpen).pack()
button1 = Button(canvas,text = "Prediction", width=50,height=5,bg='black',fg='yellow', command= prediction).pack()
button2 = Button(canvas,text = "Train", width=50,height=5,bg='black',fg='yellow', command= train).pack()
button3 = Button(canvas,text = "Show Result", width=50,height=5,bg='black',fg='yellow', command= update).pack()
canvas.mainloop()
