#include "threadtrainer.h"
#include <QtCore>
#include <QDebug>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "GUI/mainwindow.h"
#include <string.h>
#include <armadillo>
#include <stdio.h>
#include <map>
#include  <random>
#include  <iterator>
#include <tiny_dnn/tiny_dnn.h>
#include <iostream>
#include <vector>
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace std;

using namespace cv;

bool ThreadTrainer::running=true;
double ThreadTrainer::stopTreshold=0.1;

ThreadTrainer::ThreadTrainer(QObject *parent,bool trSteering,bool trDetection) : QThread(parent){
    this->trainingDetection=trDetection;
    this->trainingSteering = trSteering;
    this->mini_batch_size=10;
    this->stopTreshold=0.1;

}


//int myrandom(int i) { return std::rand() % i; }
//it is allready defined in Neural_armadillo.cpp


void ThreadTrainer::run(){
    vector<vec_t> training_data;
    vector<vec_t> label_vector;

    if(trainingSteering){
        for(int i=0;i<9;i++){
            char defFilename[100];
            sprintf(defFilename,"../def%d.txt",i);
            char processedImages[100];
            sprintf(processedImages,"../out%d.avi",i);

            VideoCapture capture(processedImages);
            loadSteeringData(defFilename,capture,training_data,label_vector);
                equalizeData(training_data,label_vector);



            //correct shuffling
            vector<vec_t> shuffled_training_data;
            vector<vec_t> shuffled_label_vector;
            std::vector<int> indexes;
            indexes.reserve(training_data.size());
            for (int i = 0; i < training_data.size(); i++)
                indexes.push_back(i);
            std::random_shuffle(indexes.begin(), indexes.end());

            for(int i=0;i<indexes.size();i++){
                shuffled_training_data.push_back(training_data[indexes[i]]);
                shuffled_label_vector.push_back(label_vector[indexes[i]]);
            }
            training_data.clear();
            label_vector.clear();
            //

            //std::random_shuffle(training_data.begin(), training_data.end(), Neural::myrandom);
            std::size_t const a_fifth = shuffled_training_data.size() / 5;//o cincime

            std::vector<vec_t> training_set_X(shuffled_training_data.begin(), shuffled_training_data.begin() + (shuffled_training_data.size()-a_fifth));
            std::vector<vec_t> training_set_Y(shuffled_label_vector.begin(),shuffled_label_vector.begin()+(shuffled_label_vector.size()-a_fifth));

            std::vector<vec_t> validation_set_X(shuffled_training_data.begin() + (shuffled_training_data.size()-a_fifth), shuffled_training_data.end());
            std::vector<vec_t> validation_set_Y(shuffled_label_vector.begin()+(shuffled_label_vector.size()-a_fifth),shuffled_label_vector.end());

            while(training_set_X.size()%30!=0){
                training_set_X.pop_back();
                training_set_Y.pop_back();
            }

            adam optimizer;
            double learning_rate =0.1;
            optimizer.alpha *=  learning_rate;

            double lastCost = 9999;
            auto on_enumerate_epoch = [&](){///////////////////////////////////ATENTIE NU SE PUNE PARANTEZA INAINTE DE EGAL
                //result res = MainWindow::nn.test(validation_set_X,validation_set_Y);
                int good = 0;
                qDebug()<<(MainWindow::nn.predict(training_set_X[0]));
                for(int i=0;i<validation_set_X.size();i++){
                    vec_t res = MainWindow::nn.predict(validation_set_X[i]);
                    double max = -1;
                    int indexMax=-1;
                    for(int j=0;j<res.size();j++){
                        if(res[j]>max){max =res[j];indexMax = j;}
                    }
                    int indexMaxLabel = -1;
                    max = -1;

                    for(int j=0;j<res.size();j++){
                        if(validation_set_Y[i][j]>max){max =validation_set_Y[i][j];indexMaxLabel = j;}
                    }
                    if(indexMaxLabel == indexMax)
                        good++;

                }
                qDebug()<< "accuracy validation: "<<good <<" / "<< validation_set_Y.size();

                good = 0;
                //qDebug()<<(MainWindow::nn.predict(validation_set_X[0]))[0];
                for(int i=0;i<training_set_X.size();i++){
                    vec_t res = MainWindow::nn.predict(training_set_X[i]);
                    double max = -1; int indexMax=-1;
                    for(int j=0;j<res.size();j++){
                        if(res[j]>max){max =res[j];indexMax = j;}
                    }
                    int indexMaxLabel = -1;max = -1;
                    for(int j=0;j<res.size();j++){
                        if(training_set_Y[i][j]>max){max =training_set_Y[i][j];indexMaxLabel = j;}
                    }
                    if(indexMaxLabel == indexMax)
                        good++;

                }
                qDebug()<< "accuracy training: "<<good <<" / "<< training_set_Y.size();

                double loss = MainWindow::nn.get_loss<mse>(training_set_X, training_set_Y);
                if(loss >lastCost)
                    optimizer.alpha*=0.9;
                lastCost = loss;
                qDebug()<<"Loss training: "<<loss;
            };
            auto on_enumerate_minibatch = [&](){    ///////////////////////////////////ATENTIE NU SE PUNE PARANTEZA INAINTE DE EGAL

            };
            MainWindow::nn.weight_init(weight_init::xavier(1.0));
            MainWindow::nn.bias_init(weight_init::xavier(1.0));
            MainWindow::nn.fit<mse>(optimizer,training_set_X,training_set_Y,30,1000,on_enumerate_minibatch,on_enumerate_epoch);
            //nn.fit<cross_entropy>(optimizer,training_set_X,training_set_Y,30,30,on_batch_enumerate,on_enumerate_epoch);

           //nn.fit<cross_entropy>(optimizer, training_set_X, training_set_Y, 30,30, on_enumerate_minibatch,on_enumerate_epoch);

//            network<sequential> nn = tiny_dnn::network<tiny_dnn::sequential>();
//                        nn <<convolutional_layer<relu>(64,32,5,1,9)//49*19 size, 5 kernel size, 1 input channel, 9 output channels
//                                <<average_pooling_layer<relu>(60,28,9,2)
//                               <<convolutional_layer<relu>(30,14,5,9,18)
//                              <<average_pooling_layer<relu>(26,10,18,2)
//                             <<fully_connected_layer<relu>(14*5*18,11);
//            auto on_enumerate_epoch = [&]() {

//            };

//            auto on_enumerate_minibatch = [&]() { };

//            // training
//            nn.fit<cross_entropy>(optimizer, training_set_X, training_set_Y, 30,
//                30, on_enumerate_minibatch,
//                on_enumerate_epoch);


            //should i clear or should i append training data?
        }
        emit FinishedTrainingNeural();

    }else if(trainingDetection){
        //loadDetectionData(training_data);
//        MainWindow::net->SGD(training_data,30, mini_batch_size, 0.03, 5,true,true,false,false);
//        saveStandardNeuralNet(*MainWindow::net);
        emit FinishedTrainingNeural();
    }
}

void ThreadTrainer::saveStandardNeuralNet(Neural &net){
    for(int i=0;i<net.weights.size();i++){
        char buffer[50];
        sprintf(buffer,"w%d.mat",i);
        net.weights[i].save(buffer);
        sprintf(buffer,"b%d.mat",i);
        net.biases[i].save(buffer);
    }
}

template<typename Iter, typename RandomGenerator>
Iter select_randomly(Iter start, Iter end, RandomGenerator& g) {
    std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(g));
    return start;
}

template<typename Iter>
Iter select_randomly(Iter start, Iter end) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return select_randomly(start, end, gen);
}

void ThreadTrainer::equalizeData(vector<tiny_dnn::vec_t> &training_data,vector<tiny_dnn::vec_t> &label_vector){
    qDebug()<<"before error";
    map<int,int> distribution;
    int outputSize = 9;
    for(int i = 0;i<label_vector.size();i++){
        for (int j=0;j<outputSize;j++){
            if(label_vector[i][j]>0.1){//it is different than 0, that means it is equal to 1
                distribution[j]+=1;
                break;
            }
        }
    }
    for(int i=0;i<distribution.size();i++){
        qDebug()<<"size of "<<i<<" is "<<distribution[i];
    }
    int minim = 9999999;
    for(int i=0;i<distribution.size();i++){
        if(distribution[i] < minim)minim=distribution[i];
    }
    int mean = training_data.size()/distribution.size();
    qDebug()<<"min is "<<minim;
    qDebug()<<"distribution size should be "<<distribution.size();
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator


    for(int i = 0;i<distribution.size();i++){
        int difference = distribution[i]-mean;
        if(difference > 0){
            for(int j=0;j<difference;j++){
                std::uniform_int_distribution<> indexDistr(0, training_data.size()-1); // define the range

                int index = indexDistr(eng);
                while(label_vector[index][i]<0.9)
                    index = indexDistr(eng);
                label_vector.erase(label_vector.begin()+index);
                training_data.erase(training_data.begin()+index);
            }
        }else {
            difference*=-1;
            for(int j=0;j<difference;j++){
                std::uniform_int_distribution<> indexDistr(0, training_data.size()-1); // define the range

                int index = indexDistr(eng);
                while(label_vector[index][i]<0.9)
                    index = indexDistr(eng);
                vec_t nouX = training_data[index];
                vec_t nouY = label_vector[index];

                training_data.push_back(nouX);
                label_vector.push_back(nouY);
            }
        }
    }
    qDebug()<<"calculation new distribution";
    distribution.clear();
    for(int i = 0;i<label_vector.size();i++){
        for (int j=0;j<outputSize;j++){
            if(label_vector[i][j]>0.1){//it is different than 0, that means it is equal to 1
                distribution[j]+=1;
                break;
            }
        }
    }
    for(int i=0;i<distribution.size();i++){
        qDebug()<<"size of "<<i<<" is "<<distribution[i];
    }
}

int ThreadTrainer::inputSize = 49*19;
void ThreadTrainer::loadSteeringData(char *defFileName,VideoCapture capture,vector<vec_t> &training_data,vector<vec_t> &label_vector){
    FILE *def = fopen(defFileName,"r");
    if(!def){
        qDebug("no def file found!!!");
    }else{
        int outputSize = 9; //MAKE IT 11
        qDebug()<<"USING OUTPUTSIZE"<<outputSize;
        vec_t output;
        int value;
        for(int j=0;j<outputSize;j++){
            fscanf(def,"%d",&value);
            output.push_back(value);
        }
        Mat img;
        capture>>img;

        while (!feof(def)){

            label_vector.push_back(output);


            //imshow("img",img);
            //waitKey(33);
            cv::cvtColor(img,img,CV_BGR2GRAY); //when i saved it was CV_GRAY2RGB

            cv::resize(img,img,Size(64,32));

            //unsigned char* p = img.ptr<unsigned char>();
              vec_t inputVector;

               for(int i=0;i<32;i++){
                    for(int j=0;j<64;j++){
                        inputVector.push_back(img.at<char>(i,j)/*/255*/);
                    }
                }
            //std::transform(p,p+img.cols*img.rows,std::back_inserter(v),[&](unsigned char v){return v/255.0;});
            //vec_t vec;
            //for (int i = 0; i < img.rows; ++i) {
            //    vec.insert(vec.end(), (uchar*)img.ptr<uchar>(i), (uchar*)img.ptr<uchar>(i)+img.cols);
            //}/*
            //for(int i=v.size()-10;i<v.size();i++){
            //    qDebug()<<v[i];
            //}*/
            training_data.push_back(inputVector);

            output.clear();
            for(int j=0;j<outputSize;j++){
                fscanf(def, "%d", &value);
                output.push_back(value);
            }
            capture>>img;
        }
//        while(training_data.size()%30 !=0){
//            training_data.pop_back();
//        }
        fclose(def);
    }
}

void ThreadTrainer::loadDetectionData(vector<arma::Mat<double>>& training_data){
    FILE* def = fopen("trainingImages/def.txt", "r");
    if (!def){
        qDebug("no def file found!!");
    }
    else{
        int outputSize =2;
        std::vector<double> output;
        int value;

        char path[100];
        fscanf(def,"%s",path);

        for(int j=0;j<outputSize;j++){
            fscanf(def, "%d", &value);
            output.push_back(value);
        }

        int inputSize = 1600/*MainWindow::net->sizes(0,0)*/;
        arma::Mat<double> b(inputSize, 1); b.fill(255);
        arma::Mat<double> c(inputSize, 1); c.fill(1);
        while (!feof(def)){
            Mat img;
            img = cv::imread(path);
            cv::resize(img,img,Size(40,40));
            cv::cvtColor(img,img,CV_RGB2GRAY); //when i saved it was CV_GRAY2RGB
            std::vector<double> vec;
            for (int i = 0; i < img.rows; ++i) {
                vec.insert(vec.end(), (uchar*)img.ptr<uchar>(i), (uchar*)img.ptr<uchar>(i)+img.cols);
            }
            arma::Mat<double> image(vec);
            //image = (((image - a) / (b - a)) % (B - A)) + A;
            image = (image+c) / b;
            image.resize(inputSize+outputSize, 1);
            for (int i = 0; i < outputSize; i++){
                image(inputSize + i, 0) = output[i];
            }

            training_data.push_back(image);

            output.clear();

            fscanf(def,"%s",path);
            for(int j=0;j<outputSize;j++){
                fscanf(def, "%d", &value);
                output.push_back(value);
            }
        }
        while(training_data.size()%10 !=0){
            training_data.pop_back();
        }
        fclose(def);
    }
}

