#ifndef THREADTRAINER_H
#define THREADTRAINER_H

#include <QtCore>
#include <QtWidgets/QMainWindow>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "neural_network/Neural_Armadillo.h"
#include <tiny_dnn/tiny_dnn.h>
#include <vector>
class ThreadTrainer : public QThread{
    Q_OBJECT
public:

    static bool running;
    bool trainingSteering;
    bool trainingDetection;
    int mini_batch_size;
    static double stopTreshold;
    static int inputSize;

public:
    ThreadTrainer(QObject *parent ,bool trSteering,bool trDetection);
    void run();
    static void saveStandardNeuralNet(Neural &net);
    void ThreadTrainer::loadSteeringData(char *defFileName,cv::VideoCapture capture,vector<tiny_dnn::vec_t> &training_data,vector<tiny_dnn::vec_t> &label_vector);
    void loadDetectionData(vector<arma::Mat<double>>& training_data);
    void equalizeData(std::vector<tiny_dnn::vec_t> &training_data,std::vector<tiny_dnn::vec_t> &label_vector);

signals:
    void FinishedTrainingNeural();
};


#endif // STANDARDNEURALTRAINER_H
