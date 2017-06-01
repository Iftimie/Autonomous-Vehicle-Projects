#ifndef THREADAUTOPILOT_H
#define THREADAUTOPILOT_H
#include <QtCore>
#include <QtWidgets/QMainWindow>
#include <opencv2/core/core.hpp>

class ThreadAutopilot : public QThread{
    Q_OBJECT
public:
    static bool running;
public:
    ThreadAutopilot(QObject *parent );
    void run();

signals:
    void predictedAngle(int angle);
};


#endif // THREADAUTOPILOT_H
