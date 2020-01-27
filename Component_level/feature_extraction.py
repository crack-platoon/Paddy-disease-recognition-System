import cv2
import numpy as np

end  = cv2.imread("end.jpg")

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
