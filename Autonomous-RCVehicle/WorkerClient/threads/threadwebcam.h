#ifndef THREADWEBCAM_H
#define THREADWEBCAM_H

#include <QtCore>
#include <QtWidgets/QMainWindow>
#include <opencv2/core/core.hpp>


class ThreadWebCam : public QThread{
    Q_OBJECT
public:
    static bool running;
public:
    ThreadWebCam(QObject *parent );
    void run();
    void colorTreshold(cv::Mat &image);

signals:
    void imageWebCamChanged();
};

#endif // THREADWEBCAM_H
