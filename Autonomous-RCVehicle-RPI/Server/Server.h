#ifndef SERVER_H
#define SERVER_H

#include <QObject>
#include <QDebug>
#include <QTcpServer>
#include <QTcpSocket>
#include "ServoThread.h"

class Server : public QObject
{
    Q_OBJECT
public:
    explicit Server(QObject *parent = 0);
    static int compression;

signals:

public slots:
       void newConnection();

private:
       QTcpServer *server;
       ServoThread* servoThread;
       void decodeRequest(char *request);
       int maestroSetTarget(int fd, unsigned char channel, unsigned short target);
       int fd;
       int steeringValue;
       int speedValue;
};

#endif // SERVER_H
