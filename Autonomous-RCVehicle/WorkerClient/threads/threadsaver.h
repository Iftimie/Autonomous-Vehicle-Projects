#ifndef THREADSAVER_H
#define THREADSAVER_H

#include <QObject>
#include <QThread>
#include <opencv2/core/core.hpp>
#include <Windows.h>
#include <opencv2/highgui/highgui.hpp>


class ThreadSaver : public QThread
{
    Q_OBJECT
public:
    ThreadSaver(QObject *parent = 0);
    static bool saving;
    bool running;
    void appendToFile(const char* fileName,const char* textToAppend);
    static void createFile(const char* fileName);
    int saveImage(cv::VideoWriter &video,cv::Mat &hsv);


signals:
    void saveCompleted();
public slots:
    void run();

};

#endif // THREADSAVER_H
