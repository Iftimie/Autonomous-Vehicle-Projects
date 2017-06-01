#ifndef THREADVIDEOPLAYER_H
#define THREADVIDEOPLAYER_H

#include <QtCore>
#include <QtWidgets/QMainWindow>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class ThreadVideoPlayer : public QThread{
    Q_OBJECT
public:
    static bool running;
    cv::VideoCapture capture;
    char filename[200];
public:
    ThreadVideoPlayer(QObject *parent ,const char *filename);
    void run();
    static void processSteeringImage(cv::Mat &image);

signals:
    void playerImageChanged();
};

#endif // THREADVIDEOPLAYER_H
