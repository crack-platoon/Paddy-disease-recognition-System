import cv2
import numpy as np
import matplotlib.pyplot as plt


cropping = False
 
x_start, y_start, x_end, y_end = 0, 0, 0, 0
 
image = cv2.imread('test2.jpg')
oriImage = image.copy()
 
 
def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping
 
    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
 
    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
 
    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished
 
        refPoint = [(x_start, y_start), (x_end, y_end)]
 
        if len(refPoint) == 2: #when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imshow("Cropped", roi)


cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_crop)
 
while True:
 
    img = image.copy()
 
    if not cropping:
        cv2.imshow("image", image)
 
    elif cropping:
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        crop_img = img[y_start:y_end, x_start:x_end] 
        cv2.imshow("image", img)
 
    key = cv2.waitKey(1) & 0xFF


    if key == ord("c"):
        
        break
        

cv2.imshow("Croped",crop_img)

img = crop_img.copy()

average = img.mean(axis=0).mean(axis=0)

pixels = np.float32(img.reshape(-1, 3))

n_colors = 5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
flags = cv2.KMEANS_RANDOM_CENTERS

_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
_, counts = np.unique(labels, return_counts=True)

dominant = palette[np.argmax(counts)]

avg_patch = np.ones(shape=img.shape, dtype=np.uint8)*np.uint8(average)

indices = np.argsort(counts)[::-1]   
freqs = np.cumsum(np.hstack([[0], counts[indices]/counts.sum()]))
rows = np.int_(img.shape[0]*freqs)

dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
for i in range(len(rows) - 1):
    dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,6))
ax0.imshow(avg_patch)
ax0.set_title('Average color')
ax0.axis('off')
ax1.imshow(dom_patch)
ax1.set_title('Dominant colors')
ax1.axis('off')
plt.show(fig)

#median and Gaussian blur operation 
median = cv2.medianBlur(crop_img,5)

cv2.imshow("Median",median)

#Gaussian noise removal
blur = cv2.GaussianBlur(median,(5,5),0)


#gray scale convertion
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)



#equ = cv2.equalizeHist(gray)


ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

cv2.imshow('ThreshHold',thresh)


kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 3)

cv2.imshow('Opening',opening)


p, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

thresh_area = 20

wd,hd,c = image.shape

for i in p:
    area = cv2.contourArea(i)
    if(area>=thresh_area):
        
        x,y,w,h = cv2.boundingRect(i)
        print("Width of affected part:" + str(w))
        M = cv2.moments(i)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        #print(cx,cy)


    
        if(cx<wd and cy<hd):
            r,g,b = image[cx,cy]
            print("Color Index of affected areas"+" "+str(r)+","+str(g)+","+str(b))



#print (cnts)

average = img.mean(axis=0).mean(axis=0)

cv2.waitKey(0)
cv2.destroyAllWindows()
