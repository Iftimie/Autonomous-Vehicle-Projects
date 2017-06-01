#ifndef SERVOTHREAD_H
#define SERVOTHREAD_H
#include <QThread>
#include <QObject>

class ServoThread:public QThread{
    Q_OBJECT
public:
    ServoThread(QObject *parent);
    static char request[3];
    static bool running;
    int steeringValue;
    int speedValue;

    void run();
};

#endif // SERVOTHREAD_H
