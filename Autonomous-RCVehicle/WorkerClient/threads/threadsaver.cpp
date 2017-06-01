#include "threadsaver.h"
#include <windows.h>
#include <winuser.h>
#include <stdio.h>
#include "GUI/mainwindow.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "tcpconnectionthread.h"
using namespace cv;
bool ThreadSaver::saving=false;


ThreadSaver::ThreadSaver(QObject *parent) : QThread(parent)
{
    running=true;
}

void ThreadSaver::run(){

    //VideoWriter video("out.avi", -1, 15, Size(640, 480), true);
    //in opencv 2.4.10 VideoWriter video("out.avi", CV_FOURCC('M','J','P','G'), 15, Size(640, 480), true);
    VideoWriter video("out.avi", CV_FOURCC('M','J','P','G'), 15, Size(640, 480), true);
    Mat *image=nullptr;
    while(running){

        image=MainWindow::storeGetImage(0,"EX");
        if(ThreadSaver::saving==false){
            qDebug()<<"waiting to press saving\n";
            QThread::msleep(500);
            continue;
        }
        saveImage(video,*image);
        delete image;
        image=nullptr;
        QThread::msleep(33);
    }
    video.release();
    emit saveCompleted();
}

void ThreadSaver::appendToFile(const char* fileName,const char* textToAppend){
    FILE *file = fopen(fileName, "a");
    if (file == NULL) {
        perror("Error opening file.");
    }
    else {
        fseek(file, 0, SEEK_END);
        fprintf(file, "%s\n", textToAppend);
        fclose(file);
    }
}

void ThreadSaver::createFile(const char* fileName){
    FILE *file = fopen(fileName, "w");
    if (file == NULL) {
        perror("Error opening file.");
    }
    else {
        fclose(file);
    }
}

int ThreadSaver::saveImage(VideoWriter &video,Mat &image){
    //cv::cvtColor(image,image,CV_BGR2RGB);
    video.write(image);
    char fullName[250];

    short value = int (TcpConnectionThread::request[2] << 8 | TcpConnectionThread::request[3]);
    qDebug()<<"value is          "<<value;
    sprintf(fullName, "%d",  value);

//    if(TcpConnectionThread::request[1]==20)
//        sprintf(fullName, "%s",  "1 0 0 0 0 0 0 0 0 0 0");
//    else if(TcpConnectionThread::request[1]==19 )
//        sprintf(fullName, "%s",  "0 1 0 0 0 0 0 0 0 0 0");
//    else if(TcpConnectionThread::request[1]==18)
//        sprintf(fullName, "%s",  "0 0 1 0 0 0 0 0 0 0 0");
//    else if(TcpConnectionThread::request[1]==17)
//        sprintf(fullName, "%s",  "0 0 0 1 0 0 0 0 0 0 0");
//    else if(TcpConnectionThread::request[1]==16)
//        sprintf(fullName, "%s",  "0 0 0 0 1 0 0 0 0 0 0");
//    else if(TcpConnectionThread::request[1]==15)
//        sprintf(fullName, "%s",  "0 0 0 0 0 1 0 0 0 0 0");
//    else if(TcpConnectionThread::request[1]==14)
//        sprintf(fullName, "%s",  "0 0 0 0 0 0 1 0 0 0 0");
//    else if(TcpConnectionThread::request[1]==13)
//        sprintf(fullName, "%s",  "0 0 0 0 0 0 0 1 0 0 0");
//    else if(TcpConnectionThread::request[1]==12)
//        sprintf(fullName, "%s",  "0 0 0 0 0 0 0 0 1 0 0");
//    else if(TcpConnectionThread::request[1]==11)
//        sprintf(fullName,"%s",   "0 0 0 0 0 0 0 0 0 1 0");
//    else if(TcpConnectionThread::request[1]==10)
//        sprintf(fullName, "%s",  "0 0 0 0 0 0 0 0 0 0 1");

    qDebug()<<fullName;
    appendToFile("def.txt", fullName);
    return 0;
}
