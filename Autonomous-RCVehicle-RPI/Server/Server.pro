#-------------------------------------------------
#
# Project created by QtCreator 2016-08-23T22:02:13
#
#-------------------------------------------------

QT       += core

QT       -= gui
QT+=network

TARGET = Server
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

INCLUDEPATH +=/usr/local/include/opencv
INCLUDEPATH +=/usr/local/include/opencv2
LIBS+=-L/usr/local/lib \
-lopencv_shape \
-lopencv_stitching \
-lopencv_objdetect \
-lopencv_superres \
-lopencv_videostab \
-lopencv_calib3d \
-lopencv_features2d \
-lopencv_highgui \
-lopencv_videoio \
-lopencv_imgcodecs \
-lopencv_video \
-lopencv_photo \
-lopencv_ml \
-lopencv_imgproc \
-lopencv_flann \
-lopencv_core

LIBS+=-L/usr/local/lib \
-lwiringPi

SOURCES += main.cpp \
    Server.cpp \
    ServoThread.cpp

HEADERS += \
    Server.h \
    ServoThread.h
