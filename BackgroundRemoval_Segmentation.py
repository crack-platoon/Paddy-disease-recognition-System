import cv2
import numpy as np
import random

image = cv2.imread("train19.jpg")

img = cv2.resize(image,(256,256))

img = np.uint8(img)


hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

h,s,v = cv2.split(hsv)

cv2.imshow("Saturation",s)

equ = cv2.equalizeHist(s)

ret, thresh = cv2.threshold(equ,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

cv2.imshow('ThreshHold',thresh)




kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 3)



ret, mask = cv2.threshold(s, 30, 255, cv2.THRESH_BINARY)
opening1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 8)
#cv2.imshow('Opening',opening1)
#background_removed = cv2.bitwise_or(image,mask)



only_leaf = cv2.bitwise_and(image,image,mask = thresh)


gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

gray_resize = cv2.resize(gray,(256,256))
arr = np.asarray(gray_resize)

def kmeans(histogram):
    for k in range(0,21):
        #print '\niteration',k
        #''' First iteration assign random centroid points '''
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

mask = np.uint8(end)


masked_f = cv2.bitwise_and(img,img,mask = mask)

counts_non = np.count_nonzero(mask)
count_zero = (64*64)-counts_non

if(count_zero < counts_non):
    mask = np.invert(mask) 
        

masked_area = cv2.bitwise_and(img,img,mask = mask)

masked_area_resized = cv2.resize(masked_area,(256,256))

cv2.imshow("Masked",masked_area_resized)

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





cv2.imshow("Mask",final_mask)

cv2.imshow("Final_Area",final_area)


cv2.imshow("Masked",masked_f)

cv2.imshow("Background Removed",only_leaf)
cv2.imshow("image",image)




cv2.waitKey(0)
cv2.destroyAllWindows
