import cv2
import numpy as np
from helperFunctions import loadDataFromVideo,loadDataFromBinary,saveData

#X,Y = loadDataFromVideo()
#saveData(X,Y)
X,Y = loadDataFromBinary()


import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn import cross_validation
from networkFunctions import createNetwork

#scalerX = preprocessing.StandardScaler()
scalerX = preprocessing.MinMaxScaler(feature_range=(-3,3))
X = scalerX.fit_transform(X)
scalerY = preprocessing.MinMaxScaler(feature_range=(-10,10))
Y = scalerY.fit_transform(Y)
X_train, X_test, Y_train,Y_test = cross_validation.train_test_split(X, Y, test_size=0.183673, random_state=236)


with tf.Session() as sess:

    optimizer, cost, pred, x, y_, keep_prob,new_saver = createNetwork()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state('/home/iftimie/PycharmProjects/Autonomous-RCVehiclePython/AutonomousRpi/')

    writer = tf.summary.FileWriter('/home/iftimie/PycharmProjects/Autonomous-RCVehiclePython/AutonomousRpi/', graph=tf.get_default_graph())

    if ckpt and ckpt.model_checkpoint_path:
        new_saver.restore(sess, ckpt.model_checkpoint_path)


    batch_size = 11
    for epoch in range(0):
        avg_cost = 0.
        total_batch = int(X_train.shape[0] / batch_size)
        # Loop over all batches
        for i in range(total_batch - 1):
            batch_x = X_train[i * batch_size:(i + 1) * batch_size]
            batch_y = Y_train[i * batch_size:(i + 1) * batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_x, y_: batch_y,keep_prob:0.9 })
            # Compute average loss
            avg_cost += c / total_batch

        # the dimension of train while predicting will be 1325*32* (128*64) * 4 bytes  = 1.29GB
        train = sess.run(pred, feed_dict={x: X_test, keep_prob: 1.0}) # when testing the probability is kept with 1
        trainCost = 0
        for predictedTrain, groundTruthTrain in zip(train, Y_test):
            trainCost += abs((predictedTrain - groundTruthTrain)) / train.shape[0]
        print("cost for train %.10f" % (trainCost))

        new_saver.save(sess, '/home/iftimie/PycharmProjects/Autonomous-RCVehiclePython/AutonomousRpi/',latest_filename="checkpoint")

    writer.close()



