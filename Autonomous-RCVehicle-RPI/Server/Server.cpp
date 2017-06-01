#include "Server.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <unistd.h>
#include <string.h>
#include <ServoThread.h>
#include <wiringPi.h>
#include <softPwm.h>

//incluse headers for pololu
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <termios.h>
///

int Server::compression = 50;

int Server::maestroSetTarget(int fd, unsigned char channel, unsigned short target)
{
  unsigned char command[] = {0x84, channel, target & 0x7F, target >> 7 & 0x7F};
    //unsigned char command[] = {0xAA, 0x0C, 0x04,0x00, 0x70,0x2E};
  if (write(fd, command, sizeof(command)) == -1)
  {
    perror("error writing");
    return -1;
  }
  return 0;
}


#define MULTIPLIER 1

Server::Server(QObject *parent) : QObject(parent)
{
    server  = new QTcpServer(this);
    connect(server,SIGNAL(newConnection()),this,SLOT(newConnection()));
    if(!server->listen(QHostAddress::Any,1234)){
        qDebug()<<"Server could not start";
    }else{
        qDebug()<<"Server started";
    }
//    servoThread = new ServoThread(this);
//    servoThread->start();

    if(wiringPiSetup() <0){
        qDebug()<<"WiringPi not started";

    }else{
        this->steeringValue=15;
        //pinMode(3,PWM_OUTPUT);
        //softPwmCreate(3,steeringValue*MULTIPLIER,200*MULTIPLIER);


        //softPwmStop(3);

        this->speedValue=0;
        pinMode(2,PWM_OUTPUT);//forward or backward
        //pinMode(2,OUTPUT);
        //digitalWrite(2,LOW);
        softPwmCreate(2,0,200*MULTIPLIER);

        this->speedValue=0;
        pinMode(0,PWM_OUTPUT); //forward or backward
        //pinMode(0,OUTPUT);
        digitalWrite(0,LOW);
        softPwmCreate(0,0,200*MULTIPLIER);

        qDebug()<<"WiringPi started";
    }


    //code here for pololu
    const char * device = "/dev/ttyACM0";  // Linux
    this->fd = open(device, O_RDWR | O_NOCTTY);
    if (fd == -1){
        perror(device);
        exit(-1);
    }
    struct termios options;
      tcgetattr(fd, &options);
      options.c_iflag &= ~(INLCR | IGNCR | ICRNL | IXON | IXOFF);
      options.c_oflag &= ~(ONLCR | OCRNL);
      options.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
      tcsetattr(fd, TCSANOW, &options);

    maestroSetTarget(fd, 0, 6000);
    sleep(1);
    qDebug()<<"pololu driver initialized";

}


void Server::newConnection(){
    QTcpSocket *socket = server->nextPendingConnection();
    socket->waitForConnected();
    cv::VideoCapture capture(0);
    usleep(1000000*1);

    qDebug()<<"new connection";
    cv::Mat img;
    std::vector<uchar> encodedJpg;
    std::vector<int> encodingParams;encodingParams.push_back(CV_IMWRITE_JPEG_QUALITY);encodingParams.push_back(Server::compression);

    bool result;
    char req[4];

    while(1){
        result = socket->waitForReadyRead();
        if(result==false)
            break;
        socket->read(req,4);
        this->decodeRequest(req);
//        ServoThread::request[0]=req[0];
//        ServoThread::request[1]=req[1];
//        ServoThread::request[2]=req[2];
        //for(int i=0;i<2;i++)
            capture>>img; //maybe add a for 3 to eliminate buffer images
        encodedJpg.clear();
        imencode(".jpg",img,encodedJpg,encodingParams);
        int size = encodedJpg.size();
        //qDebug()<<"size of int"<<sizeof(int) <<"size is "<<size;
        socket->write((char *)&size,sizeof(size));
        socket->waitForBytesWritten();
        socket->write((char*)encodedJpg.data(),encodedJpg.size());
        socket->waitForBytesWritten();
    }
    socket->close();
    qDebug()<<"Reached end";
}

void Server::decodeRequest(char *request){
    char text[100];
    sprintf(text,"%d %d %d %d",request[0],request[1],request[2],request[3]);
    qDebug()<<text;
    if(request[1]==0 && request[0]!=this->speedValue){//0 is backwards
        this->speedValue=request[0];
        //qDebug()<<"changed value speed backwards";
        softPwmWrite(0,0);
        softPwmWrite(2,this->speedValue*MULTIPLIER);
        //digitalWrite(0,0);
        //int value = this->speedValue>0 ? 1 : 0;
        //digitalWrite(2,value);

    }else if(request[1]==1 && request[0]!=this->speedValue){//1 is forward
        //qDebug()<<"changed value speed forward";
        this->speedValue=request[0];
        softPwmWrite(2,0);
        softPwmWrite(0,speedValue*MULTIPLIER);
        //digitalWrite(2,0);
        //int value = this->speedValue>0 ? 1 : 0;
        //digitalWrite(0,value);
    }
    short steeringValue = int (request[2]<< 8 | request[3] ) ;
    qDebug()<<"steering value" <<steeringValue;

    if(steeringValue!=this->steeringValue){
        //qDebug()<<"changed value angle";
        this->steeringValue=steeringValue;
//        float A = 10,B = 20;
//        float a = 4000,b = 8000;
//        int value = ((float)this->steeringValue - A)*(b-a)/(B-A)+a;
        //softPwmWrite(3,steeringValue);
        //softPwmCreate(3,steeringValue,200);
        //usleep(500000*1);
        //softPwmStop(3);

        //qDebug()<<"value is "<<value;
        maestroSetTarget(fd, 0, steeringValue);
    }
}

