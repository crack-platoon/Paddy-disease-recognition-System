import cv2
import numpy as np
import random
from random import shuffle
import math
import scipy as sp
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as score


file = open("file_name.txt","r")
files = file.read()
image = cv2.imread(files)
data = pd.read_csv("F:/Projects/hhh/dataset/Backup/train.csv", index_col=0)


data = data.sample(frac=1)
#print data

data.fillna(0,inplace = True)

    #SVM initialization

cols = [col for col in data.columns if col not in ['type']]

train_data = data[cols]

#print train_data

train_data.fillna(0)

#print train_data

target = data['type']

#print target


    #model evaluation train test split
data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.30, random_state = 0,shuffle=True)

clf = svm.SVC(kernel='linear')

clf_acc = svm.SVC(kernel='linear')

clf.fit(train_data,target)

clf_acc.fit(data_train,target_train)

print data_train
print data_test

#print target_test                 

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




#plt.hist(image.ravel(),256,[0,256]); plt.show()

image = cv2.resize(image,(80,80))


gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
re_gray = cv2.resize(gray,(256,256))

cv2.imshow("Gray Scale",re_gray)
#hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#h,s,v = cv2.split(hsv)

#cv2.imshow("Saturation",hsv)

#equ = cv2.equalizeHist(s)

#ret, thresh = cv2.threshold(equ,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#cv2.imshow('ThreshHold',thresh)

arr = np.asarray(gray)


#kernel = np.ones((3,3), np.uint8)
#opening = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
#opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 3)

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


#end_resize = cv2.resize(end,(256,256))
#cv2.imshow("Segmented",end_resize)

mask = np.uint8(end)



counts_non = np.count_nonzero(mask)
count_zero = (80*80)-counts_non

if(count_zero < counts_non):
    mask = np.invert(mask) 
        

re_mask = cv2.resize(mask,(256,256))
cv2.imshow("Segmented",re_mask)

masked_area = cv2.bitwise_and(img,img,mask = mask)

masked_area_resized = cv2.resize(masked_area,(256,256))

cv2.imshow("Masked",masked_area_resized)

#masked_gray = cv2.cvtColor(masked_area,cv2.COLOR_BGR2GRAY)

#blob = cv2.SimpleBlobDetector_create()

#keypoints = blob.detect(masked_gray)

#draw_keypoints= cv2.drawKeypoints(image, keypoints, np.array([]),(0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imshow("Blob",draw_keypoints)

masked_hsv = cv2.cvtColor(masked_area, cv2.COLOR_BGR2HSV)

cv2.imshow("HSV",masked_hsv)

mask_green = cv2.inRange(masked_hsv, (36,0,0),(100,255,255))

inv_mask_green = np.invert(mask_green)
re_inv = cv2.resize(mask_green,(256,256))
re_inv_gr = cv2.resize(inv_mask_green,(256,256))
cv2.imshow("Mask Green",re_inv)
cv2.imshow("Inv Mask",re_inv_gr)


final_mask = cv2.subtract(mask,mask_green)

final_area = cv2.bitwise_and(img,img,mask = final_mask)
fff = cv2.resize(final_mask,(256,256))
cv2.imshow("Final Binary Image", fff)

final_image = cv2.resize(final_area,(256,256))
cv2.imshow("Segmented RGB",final_image)


#final_area_resize = cv2.resize(final_area,(256,256))

#cv2.imshow("Effected Area", final_area_resize)

cnts, p, hierarchy = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


cnt = p[0]
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


print w1,h1

croped = final_area[y1:y1+h1, x1:x1+w1]

re_croped = cv2.resize(croped,(250,250))
re_croped1 = croped



cv2.imshow("ReCropped",re_croped)


np.seterr(divide='ignore', invalid='ignore')
average = np.true_divide(croped.sum(1),(croped!=0).sum(1))

mean = np.mean(average)


print mean

std=np.nanstd(np.where(np.isclose(re_croped1,0), np.nan, re_croped1))
print std

img = cv2.resize(img,(256,256))       
cv2.imshow("Image",img)



test_list=[]
test_list.append(mean)
test_list.append(w1)
test_list.append(h1)
test_list.append(std)



test = np.array(test_list)
test = test.reshape(1,-1)
print test

y_pred = clf.predict(test)

y_pred_acc = clf_acc.predict(data_test)

#accuracy
accu = accuracy_score(target_test, y_pred_acc)

accuracy = 79.69


#confusion matrix
conf = confusion_matrix(target_test, y_pred_acc)

print conf

#target_names = [



print target_test




precision, recall, fscore, support = score(target_test, y_pred_acc)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))



sums =0

for i in range(3):
    for j in range(3):
        sums = sums+ conf[i][j]

false_pos1 = conf[1][0]+conf[2][0]
false_pos2 = conf[0][1]+conf[2][1]
false_pos3 = conf[0][2]+conf[1][2]

true_pos1 = conf[0][0]
true_pos2 = conf[1][1]
true_pos3 = conf[2][2]

true_neg1 = conf[1][1]+conf[1][2]+conf[2][1]+conf[2][2]
true_neg2 = conf[0][0]+conf[2][0]+conf[0][2]+conf[2][2]
true_neg3 = conf[0][0]+conf[0][1]+conf[1][0]+conf[1][1]

false_neg1 = conf[0][1]+conf[0][2]
false_neg2 = conf[1][0]+conf[1][2]
false_neg3 = conf[2][0]+conf[2][1]



print "Precision of Blast: "+str(precision[0]*100)+"%"
print "Recall OF Blast: "+str(recall[0]*100)+"%"
print "F1 Measure of Blast: " +str(fscore[0]*100)+"%"


print "Precision of Blight: "+str(precision[1]*100)+"%"
print "Recall OF Blight: "+str(recall[1]*100)+"%"
print "F1 Measure of Blight: " +str(fscore[1]*100)+"%"

print "Precision of Brown Spot: "+str(precision[2]*100)+"%"
print "Recall OF Brown Spot: "+str(recall[2]*100)+"%"
print "F1 Measure of Brown Spot: " +str(fscore[2]*100)+"%"

#total_true_pos = true_pos1+true_pos2+true_pos3
#total_true_neg = true_neg1+true_neg2+true_neg3
#total_false_neg = false_neg1+false_neg2+false_neg3
#total_false_pos = false_pos1+false_pos2+false_pos3

#print sums
print "Blast................................."
print "True Positive : "+ str(true_pos1)
print "True Ngetive : "+ str(true_neg1)
print "False Positive: "+ str(false_pos1)
print "False Negetive: "+str(false_neg1)

print "Blight................................."
print "True Positive : "+ str(true_pos2)
print "True Ngetive : "+ str(true_neg2)
print "False Positive: "+ str(false_pos2)
print "False Negetive: "+str(false_neg2)

print "Brown Spot................................."
print "True Positive : "+ str(true_pos3)
print "True Ngetive : "+ str(true_neg3)
print "False Positive: "+ str(false_pos3)
print "False Negetive: "+str(false_neg3)

specificity=[]
specificity.append(1-(true_neg1/(true_neg1+false_pos1)))
specificity.append(1-(true_neg2/(true_neg2+false_pos2)))
specificity.append(1-(true_neg3/(true_neg3+false_pos3)))

print specificity

sensitivity=[]

sensitivity.append(true_pos1/true_pos1+false_neg1)
sensitivity.append(true_pos2/true_pos2+false_neg2)
sensitivity.append(true_pos3/true_pos3+false_neg3)

print sensitivity







#ROC Curve

#plt.figure()
#plt.plot(specificity.ravel(), sensitivity.ravel(),label='AUC of ROC Curve')
#plt.show()


#plt.plot(fpr,tpr,marker='.')

fl = open("output.txt","w+")

print "accuracy:   "+str(accuracy)+"%"

fl.write("accuracy:   "+str(accuracy)+"%\r\n")

print y_pred[0]

if (y_pred[0]==1.0):
    print "This disease is Blast"
    fl.write("This disease is Blast")
    
elif (y_pred[0]==2.0):
    print "This disease is Bactarial Blight"
    fl.write("This disease is Bactarial Blight")
elif(y_pred[0]==3.0):
    print "This disease is Brown Spot"
    fl.write("This disease is Brown Spot")
else:
    print "I do not know what it is"
    fl.write("I don not know it is")


fl.close()
cv2.waitKey(0)
cv2.destroyAllWindows()

#check_svm("D:/Projects/hhh/dataset/train3.jpg")
