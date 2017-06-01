from PyQt4.QtCore import *
from PyQt4.QtGui import *
import numpy as np
import cv2


class Worker(QThread):

    def __init__(self, parent = None):

        QThread.__init__(self, parent)

    def run(self):

        videoCapture = cv2.VideoCapture("../kalmanFilter/video.avi")
        for x in range(0, 80):

            ret, imageSlot1 = videoCapture.read()


        for x in range(0,160):
            print ("dsad")

            ret,imageSlot1 = videoCapture.read()
            self.emit(SIGNAL("output(PyQt_PyObject)"),imageSlot1)



        #self.emit(SIGNAL("output(int, float)"),10, 10.2)