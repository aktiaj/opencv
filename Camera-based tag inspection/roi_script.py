import cv2
import random
scale = 0.5
circles = []
counter = 0
counter2 = 0
pointl=[]
point2=[]
myPoints = []
myColor=[]

def mousePoints(event,x,y,flags,params):
    global counter,point1,point2, counter2, circles, mycolor
    if event == cv2.EVENT_LBUTTONDOWN:
        if counter==0:
            point1=int(x//scale),int(y//scale);
            counter +=1
            mycolor = (random.randint (0,2)*200,random.randint(0,2)*200,random.randint (0,2)*200)
        elif counter == l:
            point2=int(x//scale),int(y//scale)
            typ = input('Enter Type')
            name = input ('Enter Name')
            myPoints.append([pointl.point2,typ,name])
            counter=0
        circles.append([x,y,myColor])
        counter2 += 1
img = cv2.imread("C:/Users/CK069TX/Desktop/Machine learning/photogauge/Camera-based tag inspection/query.jpg")
img = cv2.resize(img, (int(img.shape[1]/5), int(img.shape[0]/5)), None, scale, scale)


while True:
# To Display points
    for x,y,color in circles:
        cv2. circle (img, (x,y),3, color, cv2.FILLED)
    cv2. imshow ("Original Image", img)
    cv2.setMouseCallback("Original Image", mousePoints)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print (myPoints)
        break



















