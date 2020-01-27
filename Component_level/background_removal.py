import cv2
import numpy as np
import math
import scipy as sp
import random
import glob
import os
import pandas as pd
for file in glob.glob("*.jpg"):
    image = cv2.imread(file)
    image = cv2.resize(image,(120,120))
    image_list.append(image)
    print file
    name = str(file)
    n1,n2 = name.split("_")
    print n1
    if(n1=='blast'):
        image_name.append(1)
    elif(n1=='blight'):
        image_name.append(2)
    elif(n1=='brown'):
        image_name.append(3)

    
    


    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h,s,v = cv2.split(hsv)

    #cv2.imshow("Saturation",hsv)

    equ = cv2.equalizeHist(s)

    ret, thresh = cv2.threshold(equ,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#cv2.imshow('ThreshHold',thresh)

    arr = np.asarray(gray)


    kernel = np.ones((3,3), np.uint8)
#opening = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 3)

    cv2.waitKey(0)

    cv2.destroyAllWindows

#cv2.imshow('Opening',opening)

