#include "MyLabel.h"
#include <qlabel.h>
#include <QMouseEvent>
#include <QTransform>
#include <QLabel>
#include <QPainter>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "mainwindow.h"
#include <math.h>
MyLabel::MyLabel(QWidget *parent) : QLabel(parent) {
	setMouseTracking(true);
    update();
}

void MyLabel::paintEvent(QPaintEvent * e){
    QLabel::paintEvent(e);
    QPainter painter(this);
    QPen paintpen(Qt::red);
    paintpen.setWidth(5);
    painter.setPen(paintpen);

    int width = 490;
    int height = 40;
    for(int i=1;i<11;i++){
        QLineF line(i*(width/11),0,i*(width/11),height);
        painter.drawLine(line);
    }
}

void MyLabel::mouseMoveEvent(QMouseEvent *ev) {
	QTransform t;
	t.scale(1, -1);
	t.translate(0, -height() + 1);
	QPoint pos = (*ev).pos() * t;
    //int value = (pos.x()-0)*(9-20)/(491-0)+20;
    short value = (pos.x()-0)*(4000-8000)/(491-0)+8000;
    qDebug()<<"value is"<<value;
    //TcpConnectionThread::request[1]=(char)value;
    TcpConnectionThread::request[3]= value & 0xff;
    TcpConnectionThread::request[2] = (value >> 8) &0xff;
    emit steeringAngleChanged(value);
}

void MyLabel::mouseReleaseEvent(QMouseEvent *e){

}
void MyLabel::mousePressEvent(QMouseEvent *e)
{

}
