#-------------------------------------------------
#
# Project created by QtCreator 2016-12-02T22:20:19
#
#-------------------------------------------------

QT       += core
QT+=gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

QT += widgets
QT +=network
QMAKE_CXXFLAGS += /bigobj
TARGET = WorkerClient
TEMPLATE = app

CONFIG+=app_bundle
DEFINES += QT_DEPRECATED_WARNINGS

HEADERS  += keylogger.h \
    threads/tcpconnectionthread.h \
    threads/threadsaver.h \
    threads/threadtrainer.h \
    threads/threadvideoplayer.h \
    threads/threadwebcam.h \
    neural_network/Cost.h \
    neural_network/Neural_Armadillo.h \
    GUI/labelselectdataset.h \
    GUI/mainwindow.h \
    GUI/MyLabel.h \
    threads/threadautopilot.h \
    threads/threadvehicletracking.h \
    neural_network/Kalman.hpp \
    neural_network/hungarian.hpp

SOURCES += main.cpp\
    keylogger.cpp \
    threads/tcpconnectionthread.cpp \
    threads/threadsaver.cpp \
    threads/threadtrainer.cpp \
    threads/threadvideoplayer.cpp \
    threads/threadwebcam.cpp \
    neural_network/Cost.cpp \
    neural_network/Neural_Armadillo.cpp \
    GUI/labelselectdataset.cpp \
    GUI/mainwindow.cpp \
    GUI/MyLabel.cpp \
    threads/threadautopilot.cpp \
    threads/threadvehicletracking.cpp



LIBS +="C:/Program Files (x86)/Windows Kits/10/Lib/10.0.10240.0/um/x64/User32.Lib"
LIBS +="C:/Program Files (x86)/Windows Kits/10/Lib/10.0.10240.0/um/x64/Gdi32.Lib"

INCLUDEPATH+=E:/programs/tiny-dnn/

INCLUDEPATH +=E:/programs/opencv/build/include
LIBS +=-LE:/programs/opencv/build/x64/vc14/lib \
opencv_world320d.lib

INCLUDEPATH +=E:/programs/Armadillo/include
LIBS +=-LE:/programs/Armadillo/examples/lib_win64 \
blas_win64_MT.lib \
lapack_win64_MT.lib



FORMS    += GUI/mainwindow.ui



RESOURCES += \
    GUI/myresources.qrc

DISTFILES += \
    README QT.txt
