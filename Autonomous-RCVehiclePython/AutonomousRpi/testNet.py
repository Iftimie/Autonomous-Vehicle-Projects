import cv2
import numpy as np
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from helperFunctions import loadDataFromVideo,loadDataFromBinary,saveData
from networkFunctions import createNetwork
from scipy.interpolate import interp1d

X,Y = loadDataFromBinary()
#scalerX = preprocessing.MinMaxScaler(feature_range=(-3,3))
#scalerY = preprocessing.MinMaxScaler(feature_range=(-10,10))
from sklearn.externals import joblib
# joblib.dump(scalerX, 'scalerX.pkl')
# joblib.dump(scalerY, 'scalerY.pkl')

scalerX = joblib.load('scalerX.pkl')
scalerY = joblib.load('scalerY.pkl')
X_transformed = scalerX.fit_transform(X)
Y = scalerY.fit_transform(Y)



with tf.Session() as sess:
    optimizer, cost, pred, x, y_, keep_prob, new_saver = createNetwork()
    ckpt = tf.train.get_checkpoint_state('/home/iftimie/PycharmProjects/Autonomous-RCVehiclePython/AutonomousRpi/')
    if ckpt and ckpt.model_checkpoint_path:
        new_saver.restore(sess, ckpt.model_checkpoint_path)


    Y_pred = sess.run(pred, feed_dict={x: X_transformed, keep_prob: 1.0})
    mapper = interp1d([-10,10],[640,0])
    Y_pred = mapper(Y_pred)
    Y = mapper(Y)
    for i in range(len(Y)):
        xcheck = cv2.resize(X[i].reshape((64, 128)), (640, 480 - 230))
        xcheck = cv2.cvtColor(xcheck,cv2.COLOR_GRAY2BGR)
        xcheck = cv2.circle(xcheck,(int(Y[i]),50),10,[255,0,0],9)
        xcheck = cv2.circle(xcheck,(int(Y_pred[i]),60),10,[0,0,255],9)
        cv2.imshow("out",xcheck)
        cv2.waitKey(22)



#
