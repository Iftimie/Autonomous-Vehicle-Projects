


#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include "tiny_dnn/tiny_dnn.h"
#include <iostream>
#include <vector>

#include <QMainWindow>
#include <opencv2/core/core.hpp>
#include "threads/tcpconnectionthread.h"
#include "keylogger.h"
#include "threads/threadsaver.h"
#include "threads/threadwebcam.h"
#include "threads/threadtrainer.h"
#include <vector>
#include "threads/threadvideoplayer.h"
#include "threads/threadautopilot.h"
#include "threads/threadvehicletracking.h"


namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    static cv::Mat originalImage;
    static ThreadSaver* threadSaver;
    static cv::Rect roi;
    static Neural* net;
    static bool treshold;
    static bool tresholdOnlyOne,squareSelect;
    //static int colorIndex;
    static int hsvDifference;
    static vector<cv::Scalar> minHSVColorList;
    static vector<cv::Scalar> maxHSVColorList;
    static int minH,minS,minV,maxH,maxS,maxV;
    static int erode,dilate;
    static int cannyLowSteer,cannyHighSteer;
    static int minHSteer,minSSteer,minVSteer,maxHSteer,maxSSteer,maxVSteer;
    static int erodeSteer,dilateSteer;
    static int ROIX1Steer,ROIY1Steer,ROIX2Steer,ROIY2Steer;
    static int accelerationPower;
    static int outputs;

    static double scaleFactor;
    static int minNeighbours,minSize,maxSize;
    static int flag;

    static tiny_dnn::network<tiny_dnn::sequential> nn;

    ~MainWindow();
    static cv::Mat* storeGetImage(cv::Mat *img,const char* action);
    static cv::Rect getLargestRect(double imageWidth, double imageHeight, double rotAngDeg, int type);
    static cv::Mat rotate_and_crop(double angle,cv::Mat &mat);
    static void augmentData(char* defFilename,cv::VideoCapture &capture);
    static cv::Mat translateImg(cv::Mat &img, int offsetx, int offsety);

private slots:
    void on_btnConnect_clicked();
    void on_ImageChanged();
    void on_steeringAngleChanged(int);
    void on_verticalSlider_valueChanged(int value);

    void on_pushButton_clicked();

    void on_tabWidget_tabBarClicked(int index);

    void on_btnOpenVideo_clicked();

    void on_btnNextFrame_clicked();

    void on_trainRoiChanged();

    void on_pushButton_2_clicked();

    void on_btnRecreateDefFile_clicked();

    void on_radioPositive_clicked();

    void on_radioNegative_clicked();

    void on_spinIndexToStart_valueChanged(int arg1);

    void on_indexChanged();

    void on_btnOpenCamera_clicked();

    void on_imageCameraChanged();

    void closeEvent(QCloseEvent *event);

    void on_btnTrain_clicked();

    void on_FinishedTrainingNeural();


    void on_btnCreateNN_clicked();

    void on_doubleSpinBox_valueChanged(double arg1);

    void on_btnLoadNN_clicked();


    void on_comboIP_currentTextChanged(const QString &arg1);

    void on_btnDetect_clicked();

    void on_btnColorSelector_clicked();

    void on_btnSaveColorList_clicked();

    void on_btnLoadColorList_clicked();

    void on_listWidget_doubleClicked(const QModelIndex &index);

    void on_listWidget_clicked(const QModelIndex &index);

    void on_chkTreshold_clicked();

    void on_chkOneColorTresh_toggled(bool checked);

    void on_horizontalSlider_valueChanged(int value);

    void on_setHSVSliders(int h,int s,int v);

    void on_btnAddColor_clicked();

    void on_sliMinH_valueChanged(int value);

    void on_sliMaxH_valueChanged(int value);

    void on_sliMinS_valueChanged(int value);

    void on_sliMaxS_valueChanged(int value);

    void on_sliMinV_valueChanged(int value);

    void on_sliMaxV_valueChanged(int value);

    void on_horizontalSlider_2_valueChanged(int value);

    void on_horizontalSlider_3_valueChanged(int value);

    void on_chkRectSelect_toggled(bool checked);

    void on_sliMinH_Steering_valueChanged(int value);

    void on_sliMaxH_Steering_valueChanged(int value);

    void on_sliMinS_Steering_valueChanged(int value);

    void on_sliMaxS_Steering_valueChanged(int value);

    void on_sliMinV_Steering_valueChanged(int value);

    void on_sliMaxV_Steering_valueChanged(int value);

    void on_sli_lowCanny_valueChanged(int value);

    void on_sli_highCanny_valueChanged(int value);

    void on_sli_Erode_Steering_valueChanged(int value);

    void on_sli_Dilate_Steering_valueChanged(int value);

    void on_btn_Play_Video_clicked();

    void on_playerImageChanged();

    void on_btnSteeringParams_clicked();



    void on_sliROIx1Steer_valueChanged(int value);

    void on_sliROIy1Steer_valueChanged(int value);

    void on_sliROIx2Steer_valueChanged(int value);

    void on_sliROIy2Steer_valueChanged(int value);

    void on_btnConvertToTraining_clicked();

    void on_changedAcceleration();

    void on_btnTrainSteeringNN_clicked();

    void on_btnLoadSteeringNN_clicked();

    void on_btnEngageAutopilot_clicked();

    void on_predictedAngle(int angle);

    void on_btnTrackVehicles_clicked();

    void on_doubleSpinBox_2_valueChanged(double arg1);

    void on_spinBox_valueChanged(int arg1);

    void on_checkBox_clicked(bool checked);

    void on_spinBox_2_valueChanged(int arg1);

    void on_spinBox_3_valueChanged(int arg1);

    void on_btnAugmentData_clicked();

private:
    Ui::MainWindow *ui;
    TcpConnectionThread* tcpConnectionThread;
    KeyLogger* keyLoggerThread;
    cv::VideoCapture videoIn;
    int tabIndex;
    ThreadWebCam* threadWebcam;
    ThreadTrainer* threadTrainer;
    ThreadVideoPlayer* threadVideoPlayer;
    ThreadAutopilot* threadAutopilot;
    ThreadVehicleTracing* threadVehicleTracking;


    void construct_net();
    void loadSteeringParams();
    void keyPressEvent(QKeyEvent *ev);

};

#endif // MAINWINDOW_H
