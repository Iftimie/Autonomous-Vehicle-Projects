#include "threadvehicletracking.h"
#include <opencv2/imgproc/imgproc.hpp>
#include  <opencv2/objdetect/objdetect.hpp>
#include "neural_network/Kalman.hpp"
#include "GUI/mainwindow.h"
using namespace cv;

bool ThreadVehicleTracing::running = false;
ThreadVehicleTracing::ThreadVehicleTracing(QObject *parent):QThread(parent){

}

void ThreadVehicleTracing::run(){
    CascadeClassifier car_cascade;
    if (!car_cascade.load("cars.xml")) {
        qDebug()<<"Unable to load classifier  XML";
        return;
    }

    Mat greyscaleFrame;
    KalmanFilter kalman;
    while(ThreadVehicleTracing::running){
        Mat *image = MainWindow::storeGetImage(NULL,"EX");
        if(image ==nullptr){
            qDebug()<<"Received NULL Image in Thread Vehicle Tracking ";
            QThread::msleep(33);
            continue;
        }

        cvtColor(*image, greyscaleFrame, CV_BGR2GRAY);
        equalizeHist(greyscaleFrame,greyscaleFrame);
        std::vector<Rect> cars;
        std::vector<Point> detections;
        car_cascade.detectMultiScale(greyscaleFrame, cars,1.1,3/* MainWindow::scaleFactor, MainWindow::minNeighbours,MainWindow::flag,Size(MainWindow::minSize,MainWindow::minSize),Size(MainWindow::maxSize,MainWindow::maxSize)/*, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30)*/);
        for (int i = 0; i < cars.size(); i++){
            if(cars[i].width * cars[i].height > 13500)
                detections.push_back(Point((cars[i].x + cars[i].width/2),(cars[i].y + cars[i].height/2)));
            else
                continue;
            Point pt1(cars[i].x + cars[i].width, cars[i].y + cars[i].height);
            Point pt2(cars[i].x, cars[i].y);
            qDebug() << cars[i].width * cars[i].height;
            rectangle(*image, pt1, pt2, cvScalar(0, 225, 0, 0), 1, 8, 0);

        }
        kalman.predict();
        kalman.update(detections);
        kalman.draw(*image, detections);
        imshow("car tracking ",*image);
        delete image;
    }
}
