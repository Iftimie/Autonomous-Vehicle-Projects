#include <QCoreApplication>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string.h>
#include <stdio.h>
#include <map>
#include  <random>
#include  <iterator>
#include "tiny_dnn/tiny_dnn.h"
#include <iostream>
#include <vector>
#include <QDebug>

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace std;
using namespace cv;

int minHSteer,maxHSteer,minSSteer,maxSSteer,minVSteer,maxVSteer;
int cannyLowSteer,cannyHighSteer,erodeSteer,dilateSteer;
int ROIX1Steer,ROIY1Steer,ROIX2Steer,ROIY2Steer;

void loadSteeringParams(){
    FILE *f = fopen("steeringParams.txt","r");
    if(!f){
        qDebug()<<"error while opening steeringParams.txt";
    }else{
        fscanf(f,"%d %d %d %d %d %d\n",&minHSteer,&maxHSteer,&minSSteer,&maxSSteer,&minVSteer,&maxVSteer);
        fscanf(f,"%d %d %d %d\n",&cannyLowSteer,&cannyHighSteer,&erodeSteer,&dilateSteer);
        fscanf(f,"%d %d %d %d",&ROIX1Steer,&ROIY1Steer,&ROIX2Steer,&ROIY2Steer);

        fclose(f);
    }
}

bool equalizeData(vector<tiny_dnn::vec_t> &training_data,vector<tiny_dnn::vec_t> &label_vector){
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
    int maxim = 0;
    for(int i=0;i<distribution.size();i++){
        if(distribution[i] < minim)minim=distribution[i];
        if(distribution[i]>maxim)maxim = distribution[i];
    }
    int mean = training_data.size()/distribution.size();
    qDebug()<<"min is "<<minim;
    qDebug()<<"distribution size should be "<<distribution.size();
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    bool emptyDistribution=false;

    for(int i = 0;i<distribution.size();i++){
        //int difference = distribution[i]-minim;
        //int difference = distribution[i]-mean;
        int difference = distribution[i]-maxim;
        if(distribution[i]==0){
            emptyDistribution= true;
            qDebug()<<"empty";
            continue;
        }
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
    return emptyDistribution;
}

void loadSteeringData(int i,vector<vec_t> &training_data,vector<vec_t> &label_vector){
    char defFilename[100];
    sprintf(defFilename,"defAugmented%d.txt",i);
    FILE *def = fopen(defFilename,"r");
    char processedImages[100];
    sprintf(processedImages,"outAugmented%d.avi",i);
    VideoCapture capture;
    if(!capture.open(processedImages))
        qDebug("plm");
    if(!def){
        qDebug("no def file found!!!");
    }else{
        int outputSize = 11; //MAKE IT 11
        qDebug()<<"USING OUTPUTSIZE"<<outputSize;
        vec_t output;
        int value;
        for(int j=0;j<outputSize;j++){
            fscanf(def,"%d",&value);
            output.push_back(value);
        }
        Mat img;
        capture>>img;
        Rect rect = Rect(Point(ROIX1Steer,ROIY1Steer),Point(ROIX2Steer,ROIY2Steer));

        while (!feof(def)){

            label_vector.push_back(output);


            //imshow("img",img);
            //waitKey(33);
            img=img(rect);

            cv::cvtColor(img,img,CV_RGB2GRAY); //when i saved it was CV_GRAY2RGB

            cv::resize(img,img,Size(128,64));
            //imshow("img",img);
            //waitKey(33);
            //unsigned char* p = img.ptr<unsigned char>();
              vec_t inputVector;

               for(int i=0;i<64;i++){
                    for(int j=0;j<128;j++){
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

int main(int argc, char *argv[])
{

    loadSteeringParams();

    tiny_dnn::network<tiny_dnn::sequential> nn;
    int outputSize = 11;
//    nn = tiny_dnn::network<tiny_dnn::sequential>();
//    nn <<convolutional_layer<relu>(128,64,5,1,4);//49*19 size, 5 kernel size, 1 input channel, 9 output channels
//    nn<<average_pooling_layer<relu>(124,60,4,2);
//    nn<<convolutional_layer<relu>(62,30,3,4,8);
//    nn<<average_pooling_layer<relu>(60,28,8,2);
//    nn<<convolutional_layer<relu>(30,14,3,8,16);
//    nn<<average_pooling_layer<relu>(28,12,16,2);
//    nn<<fully_connected_layer<sigmoid>(14*6*16,outputSize);
    //tiny_dnn::network<tiny_dnn::sequential> nn;
    nn.load("nn-not-aug-all-papers");

    vector<vec_t> training_data;
    vector<vec_t> label_vector;

    for(int i=0;i<1;i++){
        loadSteeringData(i,training_data,label_vector);
        if (equalizeData(training_data,label_vector)==true){
            qDebug()<<"EMPTY";
            continue;
        }
    }
        qDebug()<<training_data.size();
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
        //training_data.clear();
        //label_vector.clear();
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

        int passedEpochs = 0;

        double lastCost = 9999;
        auto on_enumerate_epoch = [&](){///////////////////////////////////ATENTIE NU SE PUNE PARANTEZA INAINTE DE EGAL
            //result res = MainWindow::nn.test(validation_set_X,validation_set_Y);
            passedEpochs++;


            if(passedEpochs%5 !=0 ){

                qDebug()<<"epoch "<<passedEpochs;
                double loss = nn.get_loss<mse>(training_set_X, training_set_Y);
                if(loss >lastCost)
                    optimizer.alpha*=0.6;
                lastCost = loss;
                qDebug()<<"Loss training: "<<loss<<" alpha is "<<optimizer.alpha;
                return;
            }
            int good = 0;
            qDebug()<<(nn.predict(training_set_X[0]));
            for(int i=0;i<validation_set_X.size();i++){
                vec_t res = nn.predict(validation_set_X[i]);
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
            qDebug()<< "accuracy validation: "<<good <<" / "<< validation_set_Y.size() <<" "<<(double)good/(double)validation_set_Y.size();

            good = 0;
            //qDebug()<<(MainWindow::nn.predict(validation_set_X[0]))[0];
            for(int i=0;i<training_set_X.size();i++){
                vec_t res = nn.predict(training_set_X[i]);
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
            qDebug()<< "accuracy training: "<<good <<" / "<< training_set_Y.size()<<" "<<(double)good/(double)training_set_Y.size();



            nn.save("nn-not-aug-all-papers");
        };
        auto on_enumerate_minibatch = [&](){    ///////////////////////////////////ATENTIE NU SE PUNE PARANTEZA INAINTE DE EGAL

        };
        nn.weight_init(weight_init::xavier(1.0));
        nn.bias_init(weight_init::xavier(1.0));
        nn.fit<mse>(optimizer,training_set_X,training_set_Y,10,1000,on_enumerate_minibatch,on_enumerate_epoch);


    return 0;
}
