import sys
from PyQt4 import QtCore, QtGui
from untitled import Ui_Dialog
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from QtGUI.TcpConnectionThread import *
import cv2
import time

 
class MyDialog(QtGui.QDialog):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.thread = Worker()
        self.connect(self.thread, SIGNAL("output(PyQt_PyObject)"), self.addImage)
        print("connected")


        self.thread.start()

        self.ui.pushButton.clicked.connect(self.OK)

    def OK(self):
        print('OKvv  pressed.')

    def addImage(self,imageSlot1):
        cv2.imshow("dafsd", imageSlot1)
        cv2.waitKey(22)
        height, width, byteValue = imageSlot1.shape
        byteValue = byteValue * width
        #imageSlot1 = cv2.cvtColor(imageSlot1, cv2.COLOR_BGR2RGB)
        mQImage = QImage(imageSlot1, 281, 181, byteValue, QImage.Format_RGB888)
        self.ui.label.setPixmap(QtGui.QPixmap(mQImage))
        time.sleep(0.02)

 
if __name__ == "__main__":
        app = QtGui.QApplication(sys.argv)
        myapp = MyDialog()
        myapp.show()
        sys.exit(app.exec_())
