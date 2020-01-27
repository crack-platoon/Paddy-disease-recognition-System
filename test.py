import cv2
import numpy as np
import math
import scipy as sp
import random
import glob
import os
import pandas as pd

def kmeans(histogram):
    for k in range(0,21):
        #print '\niteration',k
        ''' First iteration assign random centroid points '''
        if k == 0:
            cent1 = rand_points[0]
            cent2 = rand_points[1]
        else:
            #print '\n selecting centroid values'
            cent1 = centroid1_avg
            cent2 = centroid2_avg

        #print histogram
        point1_centroid = []
        point2_centroid = []
        w1_centroid = []
        w2_centroid = []
        sum1 = 0
        sum2 = 0
        for i,val in enumerate(histogram):
            ''' computing absolute distance from each of the cluster and assigning it to a particular cluster based on distance'''
            #print '\n\n','i',i,'val',val,'cent1', cent1,'cent2', cent2 
            if  abs(i - cent1) <  abs(i - cent2):
                point1_centroid.append(i)
                w1_centroid.append(val)
                sum1 = sum1 + (i * val)
                
                #print '\nselection 1'
            else:
                point2_centroid.append(i)
                w2_centroid.append(val)
                sum2 = sum2 + (i * val)
               
                #print '\nselection 2'

            
                
        
        sum_w1 = sum(w1_centroid)
        sum_w2 = sum(w2_centroid)
        if(sum_w1==0):
            sum_w1 = 1
        elif(sum_w2==0):
            sum_w2 = 1
            
        centroid1_avg = int(sum1)/sum_w1      
        centroid2_avg = int(sum2)/sum_w2     
        
            
        #print '\n\n','sum1',sum1,'sum2',sum2,'cent1', centroid1_avg,'cent2', centroid2_avg
    return [point1_centroid,point2_centroid] 





path = "F:\Projects\hhh\dataset\Backup"
image_list = []
image_name = []
color_index = []
width_value =[]
height_value =[]
std_devi=[]
number_contour=[]
os.chdir(path)

for file in glob.glob("*.jpg"):
    image = cv2.imread(file)
    image = cv2.resize(image,(80,80))
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

#cv2.imshow('Opening',opening)


#k-means clustering

    rows,columns = np.shape(arr)

    rand_points = [ random.randint(1, 255) for i in range(2)]

    '''finding the histogram of the image to obtain total number of pixels in each level'''

    hist,bins = np.histogram(arr,256,[1,256])

#print hist,bins

    centroid1_avg = 0
    centroid2_avg = 0



    res = kmeans(hist)
#print res


    end = np.zeros((rows,columns))

    if (len(res[1]) < len(res[0])):
        for i in range(rows):
            for j in range(columns):
    
                if (arr[i][j] in res[1]):
                    end[i][j] = int(255)

                else:
                    end[i][j] = int(0)

    elif(len(res[0])<len(res[1])):
        for i in range(rows):
            for j in range(columns):
    
                if (arr[i][j] in res[0]):
                    end[i][j] = int(255)

                else:
                    end[i][j] = int(0)




        
        


    print end

    img = np.uint8(image)

    #cv2.imshow("Segmented",end)


    mask = np.uint8(end)
    w,h,c = img.shape


    counts_non = np.count_nonzero(mask)
    count_zero = (w*h)-counts_non

    if(count_zero < counts_non):
           mask = np.invert(mask) 
        

    masked_area = cv2.bitwise_and(img,img,mask = mask)


    #cv2.imshow("Masked",masked_area)

#masked_gray = cv2.cvtColor(masked_area,cv2.COLOR_BGR2GRAY)

#blob = cv2.SimpleBlobDetector_create()

#keypoints = blob.detect(masked_gray)

#draw_keypoints= cv2.drawKeypoints(image, keypoints, np.array([]),(0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imshow("Blob",draw_keypoints)

    masked_hsv = cv2.cvtColor(masked_area, cv2.COLOR_BGR2HSV)

    mask_green = cv2.inRange(masked_hsv, (36,0,0),(100,255,255))

    inv_mask_green = np.invert(mask_green)

    final_mask = cv2.subtract(mask,mask_green)

    final_area = cv2.bitwise_and(img,img,mask = final_mask)

    #cv2.imshow("Defected Area", final_mask)

    cnts, p, hierarchy = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    try:
        cnt = p[0]
    except IndexError:
        pass
    area = cv2.contourArea(cnt)
    
    

    for i in p:
            (x,y),radius = cv2.minEnclosingCircle(i)
            center = (int(x),int(y))
            radius = int(radius)

            for i in p:
                if cv2.contourArea(i) > area:
                    cnt = i

                    area = cv2.contourArea(i)



        
            cv2.circle(img,center,radius,(0,0,255),1)
            print radius
        



    x1,y1,w1,h1 = cv2.boundingRect(cnt)
    width_value.append(w1)
    height_value.append(h1)

    print w1,h1

    

    croped = final_area[y1:y1+h1, x1:x1+w1]

    re_croped1 = cv2.resize(croped,(250,250))
    re_croped2 = croped

    #cv2.imshow("ReCropped",re_croped)

    


    np.seterr(divide='ignore', invalid='ignore')
    average = np.true_divide(croped.sum(1),(croped!=0).sum(1))

    mean = np.mean(average)
    color_index.append(mean)

    print mean

    std=np.nanstd(np.where(np.isclose(re_croped2,0), np.nan, re_croped2))

    print std

    std_devi.append(std)
        
    #cv2.imshow("Image",img)
    cv2.waitKey(0)

    cv2.destroyAllWindows




df = pd.DataFrame()
col1 = pd.Series(color_index)
df.insert(loc=0,column='mean',value=col1)

col2 = pd.Series(width_value)
df.insert(loc=1,column='width',value=col2)

col3 = pd.Series(height_value)
df.insert(loc=2,column='height',value = col3)

col4 = pd.Series(std_devi)
df.insert(loc=3,column='std deviation',value = col4)

col5 = pd.Series(image_name)
df.insert(loc=4,column='type',value=col5)
print df
df.to_csv('train.csv')
