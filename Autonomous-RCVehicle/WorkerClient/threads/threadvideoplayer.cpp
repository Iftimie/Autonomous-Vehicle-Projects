#include "threadvideoplayer.h"
#include "GUI/mainwindow.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
bool ThreadVideoPlayer::running=true;

ThreadVideoPlayer::ThreadVideoPlayer(QObject *parent,const char *filename):QThread(parent){
    this->capture.open(filename);
    strcpy(this->filename,filename);
}

void ThreadVideoPlayer::run(){

    QThread::msleep(1000);
    Mat image;
    while(ThreadVideoPlayer::running){
        for(int i=0;i<4;i++)
            capture>>image;
        if(image.rows==0 ||image.cols==0){
            capture.open(this->filename);
            continue;
        }
        processSteeringImage(image);
        MainWindow::storeGetImage(&image,"EN");
        emit playerImageChanged();
        QThread::msleep(30);
    }
}

void ThreadVideoPlayer::processSteeringImage(Mat &image){
    Rect region_of_interest = Rect(Point(MainWindow::ROIX1Steer,MainWindow::ROIY1Steer),Point(MainWindow::ROIX2Steer,MainWindow::ROIY2Steer));
       image = image(region_of_interest);

       Mat cannyImage;
       cvtColor(image,cannyImage,CV_BGR2GRAY);
       Canny(cannyImage,cannyImage,MainWindow::cannyLowSteer,MainWindow::cannyHighSteer,3);
       Mat element = cv::getStructuringElement(MORPH_ELLIPSE,Size(2*MainWindow::dilateSteer+1,2*MainWindow::dilateSteer+1),Point(MainWindow::dilateSteer,MainWindow::dilateSteer));
       cv::dilate(cannyImage,cannyImage,element);

       Mat hsvImage;
       cvtColor(image,hsvImage,CV_BGR2HSV);
       cv::inRange(hsvImage,Scalar(MainWindow::minHSteer,MainWindow::minSSteer,MainWindow::minVSteer),Scalar(MainWindow::maxHSteer,MainWindow::maxSSteer,MainWindow::maxVSteer),hsvImage);
       Mat ErodedHSV;hsvImage.copyTo(ErodedHSV);
       element = getStructuringElement(MORPH_ELLIPSE,Size(2*MainWindow::erodeSteer+3,2*MainWindow::erodeSteer+3),Point(MainWindow::erodeSteer+1,MainWindow::erodeSteer+1));
       cv::erode(ErodedHSV,ErodedHSV,element);
       element = getStructuringElement(MORPH_ELLIPSE,Size(2*MainWindow::dilateSteer+1,2*MainWindow::dilateSteer+1),Point(MainWindow::dilateSteer,MainWindow::dilateSteer));
       cv::dilate(cannyImage,cannyImage,element);

       Mat cannyANDHSV;
       cv::bitwise_and(cannyImage,hsvImage,cannyANDHSV);

       cv::bitwise_or(cannyANDHSV,ErodedHSV,image);
   //    element = getStructuringElement(MORPH_ELLIPSE,Size(2*MainWindow::erodeSteer+1,2*MainWindow::erodeSteer+1),Point(MainWindow::erodeSteer,MainWindow::erodeSteer));
   //    cv::erode(combined,combined,element);
}


