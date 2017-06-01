from networkFunctions import *
from vehicleDetectionHelperFunctions import *
from tcpConnectionFunctions import createRequest,createSocket,getImageFromSocket
from kalmanFilter.kalman import KalmanFilter
from kalmanFilter.box import Box

img_rows = 480
img_cols = 640

model = get_small_unet(img_rows=img_rows,img_cols=img_cols)
model.load_weights("model_detect_SmallUnet.h5")


req = createRequest()
s = createSocket()



kalman = KalmanFilter(tkn_x=10,tkn_y=10)
box = Box()

while(True):
    s.send(req)
    image = getImageFromSocket(s,colorCode=cv2.IMREAD_COLOR,crop=False)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pred, im = test_new_img(model, image, new_cols=img_cols, new_rows=img_rows)
    im = np.array(im, dtype=np.uint8)
    im2 = np.copy(im)
    im_pred = np.array(255 * pred[0], dtype=np.uint8)
    im,centerPoints = getCenterPointsAndDraw(im,im_pred)
    kalman.predict()
    kalman.update(centerPoints)
    image = kalman.draw(im, centerPoints)

    states,speedVectors = kalman.getSpeedVectorsAndStatePoint()
    for i in range(len(speedVectors)):
        speedVectors[i]*=30
    for i in range(len(states)):
        p1 = (int(states[i][0]),int(states[i][1]))
        point2 = states[i]+speedVectors[i]

        p2 = (int(point2[0,0]),int(point2[0,1]))
        if(np.linalg.norm(point2-p1)<0.1):
            continue
        image = cv2.line(image,p1,p2,(255,4,170),2)

        point2 = np.array([point2[0,0],point2[0,1]])
        image = box.intersection(image,p1,point2)


    p1 = (int(box.lowerLeftCorner[0]),int(box.lowerLeftCorner[1]))
    p2 = (int(box.upperRightCorner[0]),int(box.upperRightCorner[1]))
    image = cv2.rectangle(image,p1,p2,(34,192,90),3)


    cv2.imshow("tracked", image)
    cv2.waitKey(1)