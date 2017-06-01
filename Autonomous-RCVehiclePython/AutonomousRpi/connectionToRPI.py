import socket
import struct
import cv2
import numpy as np
import tensorflow as tf

from sklearn.externals import joblib
scalerX = joblib.load('scalerX.pkl')
scalerY = joblib.load('scalerY.pkl')



from tcpConnectionFunctions import createRequest,createSocket,getImageFromSocket
from networkFunctions import createNetwork

req = createRequest()


with tf.Session() as sess:
    optimizer, cost, pred, x, y_, keep_prob, new_saver = createNetwork()
    ckpt = tf.train.get_checkpoint_state('/home/iftimie/PycharmProjects/Autonomous-RCVehiclePython/AutonomousRpi/')
    if ckpt and ckpt.model_checkpoint_path:
        new_saver.restore(sess, ckpt.model_checkpoint_path)

    s = createSocket()

    while(True):
        s.send(req)
        image = getImageFromSocket(s)

        cv2.imshow("image", image)
        cv2.waitKey(1)


        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (128, 64))
        #imageInput = scalerX.transform(np.squeeze(np.asarray(image.ravel())))
        imageInput = scalerX.transform(np.array([image.ravel()]))
        y_pred = sess.run(pred, feed_dict={x: imageInput, keep_prob: 1.0})
        y_pred = scalerY.inverse_transform(y_pred)


        req = createRequest(36,0,int(y_pred))

