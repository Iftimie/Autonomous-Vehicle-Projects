#ifndef TCPCONNECTIONTHREAD_H
#define TCPCONNECTIONTHREAD_H

#include <QtCore>
#include <QtWidgets/QMainWindow>
#include <opencv2/core/core.hpp>

class TcpConnectionThread : public QThread{
    Q_OBJECT
public:
    static bool running;
    //static unsigned char request[3];
    static unsigned char request[4];
    static QString ip;
public:
    TcpConnectionThread(QObject *parent );
    void run();

signals:
    void imageChanged();
    void changedAcceleration();
};

#endif // TCPCONNECTIONTHREAD_H
