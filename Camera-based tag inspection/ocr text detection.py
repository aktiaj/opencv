"C:\Program Files\Tesseract-OCR"

import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
import matplotlib.pyplot as plt


pytesseract.pytesseract.tesseract_cmd= "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

imgq= cv2.imread('C:/Users/CK069TX/Desktop/Machine learning/photogauge/Camera-based tag inspection/query.jpg')

plt.imshow(imgq)


resized_img = cv2.resize(imgq, (int(imgq.shape[1]/4), int(imgq.shape[0]/4)))


cv2.imshow("Output", resized_img)
cv2.waitKey(0)



orb = cv2.ORB_create(1500)
kp1, des1 = orb.detectAndCompute(resized_img, None)
#impKp1 = cv2.drawKeypoints(resized_img, kp1, None)



#cv2.imshow("KeyPoints", impKp1)
#cv2.waitKey(0)



path = "C:/Users/CK069TX/Desktop/Machine learning/photogauge/Camera-based tag inspection/Imapct Fulfillment"
mypiclist = os.listdir(path)
print(mypiclist)

per =25

roi = [[50, 60, 300, 50], "text", ""]

#roi = [[50, 60,300, 50], "text", "Customer", "Item", "Description"]


for j,y in enumerate(mypiclist): 
    img= cv2.imread(path +"/"+y)
    resized_img2 = cv2.resize(img, (int(img.shape[1]/5), int(img.shape[0]/5)))

   # cv2.imshow(y, resized_img2)
    #cv2.waitKey(0)
    kp2, des2 = orb.detectAndCompute(resized_img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)
    matches = tuple(sorted(matches, key=lambda x: x.distance))
    #matches.sort(key = lambda x: x.distance)
    good = matches[:int(len(matches)*(per/100))]
    imgMatch = cv2.drawMatches(resized_img2, kp2, resized_img, kp1, good[:100], None, 2)
    imgMatch = cv2.resize(imgMatch, (int(img.shape[1]/5), int(img.shape[0]/5)))

    cv2.imshow(y, imgMatch)
    cv2.waitKey(0)
    
    #srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    #dstPoints= np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    
    #M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    #imgscan = cv2.warpPerspective(resized_img2, M, (int(resized_img2.shape[1]), int(resized_img2.shape[0])))
    #imgscan = cv2.resize(imgscan, (int(imgscan.shape[1]/5), int(imgscan.shape[0]/5)))

    #cv2.imshow(y, imgscan)
    #cv2.waitKey(0)
    


gray = cv2.cvtColor(resized_img2, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)














    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    







































