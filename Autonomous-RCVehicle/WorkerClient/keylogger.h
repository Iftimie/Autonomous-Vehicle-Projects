#ifndef KEYLOGGER_H
#define KEYLOGGER_H

#include <QObject>
#include <QThread>
#define NOMINMAX
#include <Windows.h>

class KeyLogger : public QThread
{
    Q_OBJECT
public:
    explicit KeyLogger(QObject *parent = 0);
    bool running;
    static bool wheelChanged;
signals:
    void mouseWheelChanged();
public slots:
    void run();
};

#endif // KEYLOGGER_H
