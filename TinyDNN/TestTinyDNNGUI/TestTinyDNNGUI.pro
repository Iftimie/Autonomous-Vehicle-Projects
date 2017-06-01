#-------------------------------------------------
#
# Project created by QtCreator 2017-02-25T03:30:38
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = TestTinyDNNGUI
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0
INCLUDEPATH+=E:/programs/tiny-dnn/
INCLUDEPATH +=E:/programs/opencv/build/include
LIBS +=-LE:/programs/opencv/build/x64/vc14/lib \
opencv_world320d.lib
INCLUDEPATH +=E:/programs/Armadillo/include
LIBS +=-LE:/programs/Armadillo/examples/lib_win64 \
blas_win64_MT.lib \
lapack_win64_MT.lib

LIBS +="C:/Program Files (x86)/Windows Kits/10/Lib/10.0.10240.0/um/x64/User32.Lib"
LIBS +="C:/Program Files (x86)/Windows Kits/10/Lib/10.0.10240.0/um/x64/Gdi32.Lib"

SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui
