#include "tcpconnectionthread.h"
#include "GUI/mainwindow.h"
#include <QTcpSocket>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
bool TcpConnectionThread::running=true;
//unsigned char TcpConnectionThread::request[3]={0,15,0};//third position means forward = 0 or backwards =1
unsigned char TcpConnectionThread::request[4]={0,0,23,112};//second position means forward = 0 or backwards =1 //the last two digits should mean 6000
QString TcpConnectionThread::ip;
//first position is speed second is steering angle

TcpConnectionThread::TcpConnectionThread(QObject *parent):QThread(parent){

}

void TcpConnectionThread::run(){
    QTcpSocket *socket = new QTcpSocket(0);
    socket->connectToHost(this->ip.toLatin1().data(),1234);
    if(socket->waitForConnected())qDebug()<<"connected";
    else {
        qDebug()<<"Not connected";return;
    }
    char *sockData = new char[921600];

    int oldAcceleration = 0;
    while(TcpConnectionThread::running){
        socket->write((char*)request,4);
        socket->waitForBytesWritten();

        int size = 0;
        int bytes = 0;
        socket->waitForReadyRead();
        socket->read((char*)&size,sizeof(int));
        for(int i=0;i<size;i+=bytes){
            socket->waitForReadyRead();
            bytes = socket->read(sockData+i,size-i);
            if(bytes==-1){
                printf("error");
                break;
            }
        }

        std::vector<char> data(sockData,sockData+size);
        if(data.size()==0)continue;
        Mat temporary = cv::imdecode(data,CV_LOAD_IMAGE_COLOR);
        MainWindow::storeGetImage(&temporary,"EN");
        emit imageChanged();

        if(oldAcceleration!=MainWindow::accelerationPower){
            emit changedAcceleration();
            oldAcceleration = MainWindow::accelerationPower;
        }
    }
    socket->write("ba",2);
    socket->waitForBytesWritten();
    socket->close();
}
