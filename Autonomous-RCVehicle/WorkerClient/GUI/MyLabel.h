
#include <qlabel.h>
#include <QMouseEvent>
#include <QTransform>
#include <QLabel>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "mainwindow.h"
#ifndef MYLABEL_H
#define MYLABEL_H

class MyLabel : public QLabel {
    Q_OBJECT
public:

    MyLabel(QWidget *parent) ;
	void mouseMoveEvent(QMouseEvent *ev);
	void mouseReleaseEvent(QMouseEvent *e);
    void mousePressEvent(QMouseEvent *e);
    void paintEvent(QPaintEvent * e);
signals:
    void steeringAngleChanged(int);
};

#endif
