#include "threadautopilot.h"
#include "tcpconnectionthread.h"
#include "threadvideoplayer.h"
#include "GUI/mainwindow.h"
#include "threadtrainer.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <tiny_dnn/tiny_dnn.h>
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace cv;
bool ThreadAutopilot::running = false;
ThreadAutopilot::ThreadAutopilot(QObject *parent):QThread(parent){

}

void ThreadAutopilot::run(){
    int STARTING_ANGLE = 10;


    while(ThreadAutopilot::running){
        Mat *image = MainWindow::storeGetImage(NULL,"EX");
        if(image ==nullptr){
            qDebug()<<"Received NULL Image in ThreadAutopilot ";
            QThread::msleep(33);
            continue;
        }
        //ThreadVideoPlayer::processSteeringImage(*image);//the image should be gray
        //i believe i trained the net on color images
        //cvtColor(*image,*image,CV_BGR2GRAY);
        Rect region_of_interest = Rect(Point(MainWindow::ROIX1Steer,MainWindow::ROIY1Steer),Point(MainWindow::ROIX2Steer,MainWindow::ROIY2Steer));
        *image = (*image)(region_of_interest);
        cv::cvtColor(*image,*image,CV_RGB2GRAY); //when i saved it was CV_GRAY2RGB
        cv::resize(*image,*image,Size(128,64));

        vec_t inputVector;

         for(int i=0;i<64;i++){
              for(int j=0;j<128;j++){
                  inputVector.push_back((*image).at<char>(i,j)/*/255*/);
              }
          }
        vec_t res = MainWindow::nn.predict(inputVector);
        double max = -1;
        int indexMax=-1;
        for(int j=0;j<res.size();j++){
            if(res[j]>max){max =res[j];indexMax = j;}
        }
        qDebug()<<res;
        int angle = indexMax+STARTING_ANGLE;
        TcpConnectionThread::request[1] = angle;
        emit predictedAngle(angle);
        delete image;
    }
}
