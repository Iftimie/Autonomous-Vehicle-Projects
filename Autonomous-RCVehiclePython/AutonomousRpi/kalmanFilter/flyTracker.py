import cv2
import numpy as np

videoCapture = cv2.VideoCapture("video.avi")
for i in range(80):
    ret,img = videoCapture.read()

kernel = np.ones((5,5),np.uint8)

def processImage(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    image = cv2.inRange(image,(0,230,230),(10,255,255))
    image = cv2.dilate(image,kernel,iterations = 1)
    im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(points) for points in contours]

    detectedPoints = []

    image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    for i in range(len(boundingBoxes)):
        x = boundingBoxes[i][0]
        y =boundingBoxes[i][1]
        x2 = boundingBoxes[i][0] + boundingBoxes[i][2]
        y2 = boundingBoxes[i][1] +boundingBoxes[i][3]
        image = cv2.rectangle(image,(x,y),(x2,y2),(0,0,255),2)

        #or x+w/2,y+h/2
        detectedPoints.append((int((x+x2)/2),int((y+y2)/2)))

    cv2.imshow("im2", image)
    cv2.waitKey(22)

    return detectedPoints

from kalmanFilter.kalman import KalmanFilter

kalman = KalmanFilter()

while True:
    ret,img = videoCapture.read()
    img = img[10:img.shape[0],10:img.shape[1]]
    detectedPoints = processImage(img)

    kalman.predict()
    if len(detectedPoints) ==19:
        print("pauza")
    kalman.update(detectedPoints)
    image = kalman.draw(img,detectedPoints)
    cv2.imshow("tracked",image)
    cv2.waitKey(22)


