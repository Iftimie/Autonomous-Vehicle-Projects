#include "ServoThread.h"
#include <unistd.h>
#include <wiringPi.h>
#include <softPwm.h>
#include <QDebug>


char ServoThread::request[3];
bool ServoThread::running=true;
ServoThread::ServoThread(QObject* parent):QThread(parent){
    request[0]=0;
    request[1]=15;
    request[2]=0;

    if(wiringPiSetup() <0){
        qDebug()<<"WiringPi not started";

    }else{
        this->steeringValue=15;
        pinMode(3,PWM_OUTPUT);
        softPwmCreate(3,steeringValue,200);
        sleep(1);
        softPwmStop(3);

        this->speedValue=0;
        pinMode(2,PWM_OUTPUT);//forward or backward
        digitalWrite(2,LOW);
        softPwmCreate(2,0,200);

        this->speedValue=0;
        pinMode(0,PWM_OUTPUT); //forward or backward
        digitalWrite(0,LOW);
        softPwmCreate(0,0,200);

        qDebug()<<"WiringPi started";
    }
}

void ServoThread::run(){
    while(running){
        char text[100];
        sprintf(text,"%d %d %d",request[0],request[1],request[2]);
        qDebug()<<text;
        if(request[2]==0 && request[0]!=this->speedValue){//0 is backwards
            this->speedValue=request[0];
            qDebug()<<"changed value speed backwards";
            softPwmWrite(2,this->speedValue);

        }else if(request[2]==1 && request[0]!=this->speedValue){//1 is forward
            qDebug()<<"changed value speed forward";
            this->speedValue=request[0];
            softPwmWrite(0,request[0]);
        }
        if(request[1]!=this->steeringValue){
            qDebug()<<"changed value angle";
            this->steeringValue=request[1];
            softPwmCreate(3,steeringValue,200);
            softPwmWrite(3,this->steeringValue);
            usleep(500000*1);
            softPwmStop(3);
        }
        usleep(500000*1);
    }
}
