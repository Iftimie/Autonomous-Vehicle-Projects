#ifndef LABELSELECTDATASET_H
#define LABELSELECTDATASET_H


#include <qlabel.h>
#include <QMouseEvent>
#include <QTransform>
#include <QLabel>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "mainwindow.h"


class LabelSelectDataset : public QLabel {
    Q_OBJECT
public:

    LabelSelectDataset(QWidget *parent) ;
    void mouseMoveEvent(QMouseEvent *ev);
    void mouseReleaseEvent(QMouseEvent *e);
    void mousePressEvent(QMouseEvent *e);
    void wheelEvent ( QWheelEvent * event );

    static char vector[10];
    static int index;
    int selectedXY;
protected:

private:
    cv::Mat rotate_and_crop(double angle,cv::Mat &mat);
    cv::Mat brightness_and_contrast(cv::Mat &image,double alpha, double beta);
    void appendToFile(const char* fileName,const char* textToAppend);
    cv::Rect getLargestRect(double imageWidth, double imageHeight, double rotAngDeg, int type);
signals:
    void trainRoiChanged();
    void indexChanged();
    void setHSVSliders(int h,int s,int v);
};

#endif // LABELSELECTDATASET_H
