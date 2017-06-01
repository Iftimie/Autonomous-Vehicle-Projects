import tensorflow as tf
import numpy as np

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def createNetwork():
    with tf.name_scope('train') as scope:
        x = tf.placeholder(tf.float32, shape=[None, 128 * 64])
        y_ = tf.placeholder(tf.float32, shape=[None, 1])

        x_image = tf.reshape(x, [-1, 128, 64, 1])

        W_conv1 = weight_variable([5, 5, 1, 8])  # kernel size, input channles, output channels
        b_conv1 = bias_variable([8])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 8, 16])
        b_conv2 = bias_variable([16])
        h_conv2 = tf.nn.tanh(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        h_pool2_flat = tf.reshape(h_pool2, [-1, 32 * 16 * 16])
        keep_prob = tf.placeholder(tf.float32)

        W_fc1 = weight_variable([32 * 16 * 16, 1024])
        b_fc1 = bias_variable([1024])
        h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, 300])
        b_fc2 = bias_variable([300])
        h_fc2 = tf.nn.tanh(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

        W_fc3 = weight_variable([300, 30])
        b_fc3 = bias_variable([30])
        h_fc3 = tf.nn.tanh(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
        h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

        W_fc4 = weight_variable([30, 1])
        b_fc4 = bias_variable([1])

        pred = tf.matmul(h_fc3_drop, W_fc4) + b_fc4

        cross_entropy = tf.reduce_mean(tf.square(y_ - pred))
        optimizer = tf.train.AdamOptimizer(0.000001).minimize(cross_entropy)
        cost = tf.reduce_mean(tf.square(pred - y_))

        new_saver = tf.train.Saver()
        return optimizer,cost,pred,x,y_,keep_prob,new_saver


from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D,Lambda


def get_small_unet(img_rows,img_cols):
    inputs = Input((img_rows, img_cols,3)) # before the number of channels was 3
    inputs_norm = Lambda(lambda x: x/127.5 - 1.)
    conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
    conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
    conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
    conv8 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
    conv9 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    return model