from threading import Thread
import time
import cv2
from sklearn.externals import joblib
import tensorflow as tf
from tcpConnectionFunctions import createRequest,createSocket,getImageFromSocket
from networkFunctions import *
from vehicleDetectionHelperFunctions import *




def threadGetterImage():
    global req
    global image

    req = createRequest()
    s = createSocket()

    while (True):
        s.send(req)
        image = getImageFromSocket(s,colorCode=cv2.IMREAD_COLOR,crop=False)
        cv2.imshow("ff",image)
        cv2.waitKey(33)



def threadSteering():
    global req
    global image
    global sess
    sess = tf.Session()

    scalerX = joblib.load('../scalerX.pkl')
    scalerY = joblib.load('../scalerY.pkl')

    with tf.device('/cpu:0'):
        optimizer, cost, pred, x, y_, keep_prob, new_saver = createNetwork()
        ckpt = tf.train.get_checkpoint_state('/home/iftimie/PycharmProjects/Autonomous-RCVehiclePython/AutonomousRpi/')
        if ckpt and ckpt.model_checkpoint_path:
            new_saver.restore(sess, ckpt.model_checkpoint_path)

        while True:
            imagecopy = image[230:480, 0:640]
            imagecopy = cv2.cvtColor(imagecopy,cv2.COLOR_BGR2GRAY)
            imagecopy = cv2.resize(imagecopy, (128, 64))

            imageInput = scalerX.transform(np.array([imagecopy.ravel()]))
            y_pred = sess.run(pred, feed_dict={x: imageInput, keep_prob: 1.0})
            y_pred = scalerY.inverse_transform(y_pred)

            req = createRequest(36, 0, int(y_pred))
            time.sleep(0.2)
            print (y_pred)


def threadDetection():
    global image
    img_rows = 480
    img_cols = 640

    model = get_small_unet(img_rows=img_rows, img_cols=img_cols)
    model.load_weights("../VehicleDetection/model_detect_SmallUnet.h5")


    while True:
        imagecopy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred, im = test_new_img(model, image, new_cols=img_cols, new_rows=img_rows)
        im = np.array(im, dtype=np.uint8)
        im_pred = np.array(255 * pred[0], dtype=np.uint8)
        rgb_mask_pred = cv2.cvtColor(im_pred, cv2.COLOR_GRAY2RGB)
        rgb_mask_pred[:, :, 1:3] = 0 * rgb_mask_pred[:, :, 1:2]
        img_pred = cv2.addWeighted(rgb_mask_pred, 0.55, im, 1, 0)
        draw_img = get_BB_new_img(model=model, img=im, img_cols=img_cols, img_rows=img_rows)

        cv2.imshow("out", draw_img)
        cv2.waitKey(22)

thread1 = Thread( target=threadGetterImage  )
thread2 = Thread( target=threadSteering )
thread3 = Thread(target=threadDetection)

thread1.start()
time.sleep(5)
thread2.start()
time.sleep(5)
thread3.start()