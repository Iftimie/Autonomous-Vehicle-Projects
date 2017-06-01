import sys
from PyQt4 import QtCore, QtGui
from untitled import Ui_Dialog
 
class MyDialog(QtGui.QDialog):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.OK)

    def OK(self):
        print('OK pressed.')
 
if __name__ == "__main__":
        app = QtGui.QApplication(sys.argv)
        myapp = MyDialog()
        myapp.show()
        sys.exit(app.exec_())
