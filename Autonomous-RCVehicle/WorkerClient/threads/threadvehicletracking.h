#ifndef THREADVEHICLETRACKING_H
#define THREADVEHICLETRACKING_H

#include <QtCore>
#include <QtWidgets/QMainWindow>
#include <opencv2/core/core.hpp>

class ThreadVehicleTracing : public QThread{
    Q_OBJECT
public:
    static bool running;
public:
    ThreadVehicleTracing(QObject *parent );
    void run();

signals:
    //void predictedAngle(int angle);
};

#endif // THREADVEHICLETRACKING_H
