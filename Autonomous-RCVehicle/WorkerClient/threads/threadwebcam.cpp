#include "threadwebcam.h"
#include "GUI/mainwindow.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
bool ThreadWebCam::running=false;

ThreadWebCam::ThreadWebCam(QObject *parent):QThread(parent){

}

void ThreadWebCam::run(){

    cv::VideoCapture webcam(0);
    QThread::msleep(1000);
    cv::Mat image;
    while(ThreadWebCam::running){
        webcam>>image;
        if(MainWindow::treshold){
            colorTreshold(image);
        }
        MainWindow::storeGetImage(&image,"EN");
        emit imageWebCamChanged();
        QThread::msleep(30);
    }
}

void ThreadWebCam::colorTreshold(cv::Mat &image){

    cv::Mat copyImage;
    image.copyTo(copyImage);
    if(MainWindow::tresholdOnlyOne /*&& MainWindow::colorIndex>=0*/){
        cv::Mat hsv;
        cv::cvtColor(copyImage, hsv, COLOR_BGR2HSV);
        int t = MainWindow::hsvDifference;

//        Scalar low = MainWindow::HSVColorList[MainWindow::colorIndex]; low[0]-=t;low[1]-=t;low[2]-=t;
//        Scalar high = MainWindow::HSVColorList[MainWindow::colorIndex]; high[0]+=t;high[1]+=t;high[2]+=t;
        cv::Scalar low(MainWindow::minH,MainWindow::minS,MainWindow::minV);
        cv::Scalar high(MainWindow::maxH,MainWindow::maxS,MainWindow::maxV);
        cv::inRange(hsv, low, high, hsv);
        if(MainWindow::erode>0){
            cv::Mat element = cv::getStructuringElement(MORPH_ELLIPSE,
                       cv::Size(2 * MainWindow::erode+ 1, 2 * MainWindow::erode + 1),
                       Point(MainWindow::erode, MainWindow::erode));
            cv::erode(hsv, hsv, element);
        }
        if(MainWindow::dilate>0){
            Mat element = getStructuringElement(MORPH_ELLIPSE,
                       Size(2 * MainWindow::dilate+ 1, 2 * MainWindow::dilate + 1),
                       Point(MainWindow::dilate, MainWindow::dilate));
            cv::dilate(hsv,hsv,element);
        }
        imshow("binary",hsv);
        waitKey(1);

        vector<cv::Rect> boundingBoxes;
        vector< vector<cv::Point> > contours;
        vector<cv::Vec4i> hierarchy;
        findContours(hsv, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
        for(int i=0;i<contours.size();i++)
            boundingBoxes.push_back(cv::boundingRect(contours[i]));
        for(int i=0;i<contours.size();i++)
            if(boundingBoxes[i].width * boundingBoxes[i].height >1600 )
            rectangle(image, boundingBoxes[i], Scalar(0,255,255), 4, 8, 0);
    }else
        for(int i=0;i<MainWindow::minHSVColorList.size();i++){
            Mat hsv;
            cv::cvtColor(copyImage, hsv, COLOR_BGR2HSV);

            Scalar low = MainWindow::minHSVColorList[i];
            Scalar high = MainWindow::maxHSVColorList[i];
            cv::inRange(hsv, low, high, hsv);
            if(MainWindow::erode>0){
                Mat element = getStructuringElement(MORPH_ELLIPSE,
                           Size(2 * MainWindow::erode+ 1, 2 * MainWindow::erode + 1),
                           Point(MainWindow::erode, MainWindow::erode));
                cv::erode(hsv, hsv, element);
            }
            if(MainWindow::dilate>0){
                Mat element = getStructuringElement(MORPH_ELLIPSE,
                           Size(2 * MainWindow::dilate+ 1, 2 * MainWindow::dilate + 1),
                           Point(MainWindow::dilate, MainWindow::dilate));
                cv::dilate(hsv,hsv,element);
            }

            vector<cv::Rect> boundingBoxes;
            vector< vector<cv::Point> > contours;
            vector<cv::Vec4i> hierarchy;
            findContours(hsv, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
            for(int j=0;j<contours.size();j++)
                boundingBoxes.push_back(cv::boundingRect(contours[j]));
            for(int j=0;j<boundingBoxes.size();j++)
                if(boundingBoxes[j].width * boundingBoxes[j].height >1600 )
                rectangle(image, boundingBoxes[j], MainWindow::minHSVColorList[i], 1, 8, 0);
        }
}
