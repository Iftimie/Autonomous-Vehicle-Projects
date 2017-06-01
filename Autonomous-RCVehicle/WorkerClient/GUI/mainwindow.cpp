#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QTcpSocket>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <QMessageBox>
#include "GUI/labelselectdataset.h"
#include <QColorDialog>
#include <QListWidgetItem>
#include "threads/threadvideoplayer.h"
#include <time.h>
#include <ctime>
#include <random>
#include <fstream>
#include <tiny_dnn/tiny_dnn.h>

//https://developer.microsoft.com/en-us/windows/downloads/windows-8-1-sdk
using namespace cv;
using namespace tiny_dnn;
using namespace tiny_dnn::layers;

cv::Mat MainWindow::originalImage;
ThreadSaver* MainWindow::threadSaver=nullptr;
cv::Rect MainWindow::roi(10,10,100,100);
Neural* MainWindow::net=nullptr;
bool MainWindow::treshold=false;
//int MainWindow::colorIndex=0;
vector<cv::Scalar> MainWindow::minHSVColorList;
vector<cv::Scalar> MainWindow::maxHSVColorList;


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    connect(this->ui->lblAngle,SIGNAL(steeringAngleChanged(int)),this,SLOT(on_steeringAngleChanged(int)));
    connect(this->ui->lblDisplayTrain,SIGNAL(trainRoiChanged()),this,SLOT(on_trainRoiChanged()));
    connect(this->ui->lblDisplayTrain,SIGNAL(indexChanged()),this,SLOT(on_indexChanged()));
    connect(this->ui->lblDisplayTrain,SIGNAL(setHSVSliders(int,int,int)),this,SLOT(on_setHSVSliders(int,int,int)));
    this->keyLoggerThread = new KeyLogger(this);
    this->keyLoggerThread->start();
    this->tabIndex=this->ui->tabWidget->currentIndex();
    this->threadTrainer=nullptr;
    this->threadWebcam=nullptr;
    this->threadVideoPlayer = nullptr;
    this->threadAutopilot = nullptr;
    this->threadVehicleTracking = nullptr;

    this->ui->comboIP->addItem("192.168.0.106");
    this->ui->comboIP->addItem("192.168.3.2");
    TcpConnectionThread::ip = "192.168.0.106";

    loadSteeringParams();
}

void MainWindow::on_trainRoiChanged(){
    if((this->threadWebcam!=nullptr && this->threadWebcam->running==true) ||
            MainWindow::originalImage.rows==0 ||MainWindow::originalImage.cols==0 )
        return;
    cv::Mat *resized = storeGetImage(nullptr,"EX");
    cv::cvtColor(*resized, *resized, CV_BGR2RGB);
    rectangle(*resized, this->roi.tl(), this->roi.br(), Scalar(64, 255, 64), 1);
    QImage imdisplay((uchar*)resized->data, resized->cols, resized->rows, resized->step, QImage::Format_RGB888); //Converts the CV image into Qt standard format
    int w = this->ui->lblDisplayTrain->width();
    int h = this->ui->lblDisplayTrain->height();
    this->ui->lblDisplayTrain->setPixmap(QPixmap::fromImage(imdisplay).scaled(w,h,Qt::KeepAspectRatio));
    delete resized;
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_btnConnect_clicked()
{
    if(this->ui->btnConnect->text()=="Connect"){
        this->tcpConnectionThread = new TcpConnectionThread(this);
        connect(this->tcpConnectionThread,SIGNAL(imageChanged()),this,SLOT(on_ImageChanged()));
        connect(this->tcpConnectionThread,SIGNAL(changedAcceleration()),this,SLOT(on_changedAcceleration()));
        this->tcpConnectionThread->start();
        this->ui->btnConnect->setText("Disconnect");
    }else {
        this->tcpConnectionThread->running=false;
        QThread::msleep(1000);
        this->ui->btnConnect->setText("Connect");
        delete this->tcpConnectionThread;
    }
}

void MainWindow::on_ImageChanged(){
    if(this->tabIndex==0){//tabIndex
        cv::Mat *resized =MainWindow::storeGetImage(nullptr,"EX");
        cv::cvtColor(*resized, *resized, CV_BGR2RGB);
        QImage imdisplay((uchar*)resized->data, resized->cols, resized->rows, resized->step, QImage::Format_RGB888); //Converts the CV image into Qt standard format

        int w = this->ui->lblDisplayOriginal->width();
        int h = this->ui->lblDisplayOriginal->height();
        this->ui->lblDisplayOriginal->setPixmap(QPixmap::fromImage(imdisplay).scaled(w,h,Qt::KeepAspectRatio));
        delete resized;
    }else{
        cv::Mat *resized =MainWindow::storeGetImage(nullptr,"EX");

        cv::cvtColor(*resized, *resized, CV_BGR2RGB);
        QImage imdisplay((uchar*)resized->data, resized->cols, resized->rows, resized->step, QImage::Format_RGB888); //Converts the CV image into Qt standard format

        int w = this->ui->lblDisplayTrain->width();
        int h = this->ui->lblDisplayTrain->height();
        this->ui->lblDisplayTrain->setPixmap(QPixmap::fromImage(imdisplay).scaled(w,h,Qt::KeepAspectRatio));
        delete resized;
    }
}

int MainWindow::accelerationPower = 0;
void MainWindow::on_verticalSlider_valueChanged(int value)
{
    int adder = 0;
    int threshold = 20;
    accelerationPower = value;
    if(value>0){
        //value = 100+value;
        value = adder + value;
        //if(value<120)
        if(value < threshold)
            value=0;
        TcpConnectionThread::request[0]=(unsigned char)value;
        TcpConnectionThread::request[1]=(unsigned char)0; //forward
        this->ui->lblSpeedAngle->setText(QString::asprintf("Speed %d\nAngle %d",value,TcpConnectionThread::request[1]));
    }if(value<0){
        //value = 100+(-value);
        value = adder + (-value);
        //if(value<120)
        if(value < threshold)
            value=0;
        TcpConnectionThread::request[0]=(unsigned char)value;
        TcpConnectionThread::request[1]=(unsigned char)1; //backwards
        this->ui->lblSpeedAngle->setText(QString::asprintf("Speed -%d\nAngle %d",value,TcpConnectionThread::request[1]));
    }
}

void MainWindow::on_changedAcceleration(){
    this->ui->verticalSlider->setValue(MainWindow::accelerationPower);
}

void MainWindow::on_steeringAngleChanged(int value){
    this->ui->lblSpeedAngle->setText(QString::asprintf("Speed %d\nAngle %d",TcpConnectionThread::request[0],value));
}

void MainWindow::on_pushButton_clicked()
{
    if(MainWindow::threadSaver==nullptr){
        MainWindow::threadSaver= new ThreadSaver(this);
        ThreadSaver::createFile("def.txt");
        threadSaver->start();
        this->ui->pushButton->setText("Stop Saving");
    }else if(MainWindow::threadSaver!=nullptr){
        MainWindow::threadSaver->running=false;
        QThread::msleep(1000);
        delete MainWindow::threadSaver;
        MainWindow::threadSaver=nullptr;
        this->ui->pushButton->setText("Start Saving");
    }
}

cv::Mat* MainWindow::storeGetImage(cv::Mat *img,const char* action){
    QMutex mutex;
    mutex.lock();
    if (strcmp(action, "EN") == 0){
        img->copyTo(MainWindow::originalImage);
        mutex.unlock();

    }
    else if (strcmp(action, "EX") == 0){
        if(MainWindow::originalImage.rows==0 ||MainWindow::originalImage.cols==0){
            mutex.unlock();
            return nullptr;
        }
        cv::Mat* newImg = new cv::Mat();
        MainWindow::originalImage.copyTo(*newImg);
        mutex.unlock();
        return newImg;
    }

    return 0;
}

void MainWindow::on_tabWidget_tabBarClicked(int index)
{
    this->tabIndex = index;
    if(index==2){
        loadSteeringParams();
    }
}


void MainWindow::on_btnOpenVideo_clicked()
{
    this->videoIn = VideoCapture("out.avi");
    QMessageBox box;
    box.setText("Video opened");
    box.exec();
}

void MainWindow::on_btnNextFrame_clicked()
{
    videoIn >> this->originalImage;
    cv::Mat resized;
    this->originalImage.copyTo(resized);
    cv::cvtColor(resized, resized, CV_BGR2RGB);
    QImage imdisplay((uchar*)resized.data, resized.cols, resized.rows, resized.step, QImage::Format_RGB888); //Converts the CV image into Qt standard format
    int w = this->ui->lblDisplayTrain->width();
    int h = this->ui->lblDisplayTrain->height();
    this->ui->lblDisplayTrain->setPixmap(QPixmap::fromImage(imdisplay).scaled(w,h,Qt::KeepAspectRatio));
}

void MainWindow::keyPressEvent(QKeyEvent *ev)
{
    if(ev->key()==Qt::Key_Up){
        videoIn >> this->originalImage;
        cv::Mat resized;
        this->originalImage.copyTo(resized);
        cv::cvtColor(resized, resized, CV_BGR2RGB);
        QImage imdisplay((uchar*)resized.data, resized.cols, resized.rows, resized.step, QImage::Format_RGB888); //Converts the CV image into Qt standard format
        int w = this->ui->lblDisplayTrain->width();
        int h = this->ui->lblDisplayTrain->height();
        this->ui->lblDisplayTrain->setPixmap(QPixmap::fromImage(imdisplay).scaled(w,h,Qt::KeepAspectRatio));
    }
}

void MainWindow::closeEvent(QCloseEvent *event){
    QMessageBox msgBox;
    msgBox.setText("Are you sure you want to exit?");
    msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::Cancel);
    int res = msgBox.exec();
    if(res == QMessageBox::Yes){
        if(this->threadWebcam!=nullptr){
            this->threadWebcam->running=false;
            QThread::msleep(1000);
            delete this->threadWebcam;
            this->threadWebcam=nullptr;
        }
        event->accept();
    }else{
        event->ignore();
    }

}

void MainWindow::on_pushButton_2_clicked()
{
    FILE *f = fopen("trainingImages/def.txt","r");
    if(!f){
        QMessageBox msg;msg.setText("No def file found!");
        msg.exec();
    }else{
        int i=0;
        char buffer[30];int v1,v2;
        fscanf(f,"%s %d %d\n",buffer,&v1,&v2);
        i++;
        while(!feof(f)){
            fscanf(f,"%s %d %d\n",buffer,&v1,&v2);
            i++;
        }
        fclose(f);
        this->ui->spinIndexToStart->setValue(i);
    }
}

void MainWindow::on_btnRecreateDefFile_clicked()
{
    FILE *f = fopen("def.txt","r");
    if(!f){
        FILE *f2 = fopen("def.txt","w");
        fclose(f2);
    }else{
        QMessageBox::StandardButton reply;
        reply = QMessageBox::question(this, "Test", "Do you want to rewrite def file?",QMessageBox::Yes|QMessageBox::No);
        if (reply == QMessageBox::Yes) {
            fclose(f);
            f = fopen("def.txt","w");
            fclose(f);
        } else{
            fclose(f);
        }

    }
}

void MainWindow::on_radioPositive_clicked()
{
    strcpy(LabelSelectDataset::vector,"1 0");
    qDebug()<<LabelSelectDataset::vector;
}

void MainWindow::on_radioNegative_clicked()
{
    strcpy(LabelSelectDataset::vector,"0 1");
    qDebug()<<LabelSelectDataset::vector;
}

void MainWindow::on_spinIndexToStart_valueChanged(int arg1)
{
    LabelSelectDataset::index = arg1;
}

void MainWindow::on_indexChanged(){
    this->ui->spinIndexToStart->setValue(LabelSelectDataset::index);

}

void MainWindow::on_btnOpenCamera_clicked()
{
    if(this->ui->btnOpenCamera->text()=="OpenCamera"){
        this->threadWebcam = new ThreadWebCam(this);
        connect(this->threadWebcam,SIGNAL(imageWebCamChanged()),this,SLOT(on_imageCameraChanged()));
        this->threadWebcam->running=true;
        this->threadWebcam->start();
        this->ui->btnOpenCamera->setText("StopCamera");
    }else{
        this->threadWebcam->running=false;
        QThread::msleep(1000);
        delete this->threadWebcam;
        this->threadWebcam=nullptr;
        this->ui->btnOpenCamera->setText("OpenCamera");
    }
}

void MainWindow::on_imageCameraChanged(){

    if(this->tabIndex==1){//tabIndex
        cv::Mat *resized =MainWindow::storeGetImage(nullptr,"EX");
        if(resized==nullptr){
            qDebug()<<"Error. Camera probably not opened";
            return;
        }
        rectangle(*resized, this->roi.tl(), this->roi.br(), Scalar(64, 255, 64), 1);

        cv::cvtColor(*resized, *resized, CV_BGR2RGB);
        QImage imdisplay((uchar*)resized->data, resized->cols, resized->rows, resized->step, QImage::Format_RGB888); //Converts the CV image into Qt standard format

        int w = this->ui->lblDisplayTrain->width();
        int h = this->ui->lblDisplayTrain->height();
        this->ui->lblDisplayTrain->setPixmap(QPixmap::fromImage(imdisplay).scaled(w,h,Qt::KeepAspectRatio));
        delete resized;
    }
}

void MainWindow::on_btnTrain_clicked()
{
    if(this->ui->btnTrain->text()=="Start Train"){
        if(this->net==nullptr){
            QMessageBox msgBox;
            msgBox.setText("Create NN?");
            msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::Cancel);
            int res = msgBox.exec();
            if(res == QMessageBox::Yes){
                arma::Mat<int> sizes(1,4);
                sizes(0,0)=1600;
                sizes(0,1)=60;
                sizes(0,2)=20;
                sizes(0,3)=2;
                this->net = new Neural(sizes);
            }else{
                return;
            }
        }else if(this->net!=nullptr){
            QMessageBox msgBox;
            msgBox.setText("ReWrite NN?");
            msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::Cancel);
            int res = msgBox.exec();
            if(res == QMessageBox::Yes){
                arma::Mat<int> sizes(1,4);
                sizes(0,0)=1600;
                sizes(0,1)=60;
                sizes(0,2)=20;
                sizes(0,3)=2;
                this->net = new Neural(sizes);
            }else{

            }
        }
        this->threadTrainer= new ThreadTrainer(this,false,true);
        connect(this->threadTrainer,SIGNAL(FinishedTrainingNeural()),this,SLOT(on_FinishedTrainingNeural()));
        this->threadTrainer->start();
        this->ui->btnTrain->setText("Stop Train");
    }else {
        this->threadTrainer->running=false;
    }
}

void MainWindow::on_FinishedTrainingNeural(){
    QMessageBox msg;
    msg.setText("Finished Training Neural");
    msg.exec();
    delete this->threadTrainer;
    this->threadTrainer=nullptr;
    this->ui->btnTrain->setText("Start Train");
}

void MainWindow::on_btnCreateNN_clicked()
{
    arma::Mat<int> sizes(1,4);
    sizes(0,0)=1600;
    sizes(0,1)=60;
    sizes(0,2)=20;
    sizes(0,3)=2;
    MainWindow::net = new Neural(sizes);
    QMessageBox msg;
    msg.setText("Neural Net created");
    msg.exec();
}

void MainWindow::on_doubleSpinBox_valueChanged(double arg1)
{
    ThreadTrainer::stopTreshold=arg1;
}

void MainWindow::on_btnLoadNN_clicked()
{
//    arma::Mat<int> sizes(1,4);
//    sizes(0,0)=1600;
//    sizes(0,1)=60;
//    sizes(0,2)=20;
//    sizes(0,3)=2;
//    MainWindow::net = new Neural(sizes);
//    for(int i=0;i<MainWindow::net->weights.size();i++){
//        char buffer[50];
//        sprintf(buffer,"w%d.mat",i);
//        MainWindow::net->weights.at(i).load(buffer);
//        sprintf(buffer,"b%d.mat",i);
//        MainWindow::net->biases.at(i).load(buffer);
//    }
    this->nn.load("nn-aug-first-paper");
    QMessageBox msg;
    msg.setText("Neural Net loaded");
    msg.exec();
}

void MainWindow::on_comboIP_currentTextChanged(const QString &arg1)
{
    TcpConnectionThread::ip = arg1;
}

void MainWindow::on_btnDetect_clicked()
{
    int inputSize = 1600;
    Mat* image = storeGetImage(nullptr,"EX");
    if(image==nullptr){
        QMessageBox msg;
        msg.setText("Null image");
        msg.exec();
        return;
    }
    arma::Mat<double> b(inputSize, 1); b.fill(255);
    arma::Mat<double> c(inputSize, 1); c.fill(1);

    double scale = 0.6;
    int slidingWindowSize = 130;
    Mat display;
    image->copyTo(display);
    for(int i = 0;i<image->cols-slidingWindowSize-1;i+=10){
        for(int j=0;j<image->rows-slidingWindowSize-1;j+=10){
            Mat image_roi = (*image)(Rect(i,j,slidingWindowSize,slidingWindowSize));
            cv::resize(image_roi,image_roi,Size(40,40));
            std::vector<double> vec;
            for (int i = 0; i < image_roi.rows; ++i) {
                vec.insert(vec.end(), (uchar*)image_roi.ptr<uchar>(i), (uchar*)image_roi.ptr<uchar>(i)+image_roi.cols);
            }
            arma::Mat<double> image(vec);
            image = (image+c) / b;
            arma::Mat<double> a = this->net->feedForward(image);
            if(a.index_max()==0 &&a(a.index_max(),0) > 0.8){
                circle(display, Point2f(i+20,j+20), 3, cv::Scalar(0, 255, 0), -1, 8);
            }
        }
    }
    imshow("asdasd",display);
    waitKey(33);
}

void MainWindow::on_btnColorSelector_clicked()
{
    QColor color = QColorDialog::getColor(Qt::white);
    //QListWidgetItem *colorItem = new QListWidgetItem(QString::asprintf("%d,%d,%d",color.hue(),color.saturation(),color.value()));
    //colorItem->setForeground(QBrush(QColor(Qt::white)));
    //colorItem->setBackground(QBrush(color));
    //this->ui->listWidget->addItem(colorItem);
    int h,s,v;
    color.getHsv(&h,&s,&v);

    this->ui->sliMinH->setValue(h/2);
    this->ui->sliMaxH->setValue(h/2);
    this->ui->sliMinS->setValue(s);
    this->ui->sliMaxS->setValue(s);
    this->ui->sliMinV->setValue(v);
    this->ui->sliMaxV->setValue(v);
}



void MainWindow::on_btnSaveColorList_clicked()
{
    FILE* f = fopen("colorlist.txt","w");
    if(!f)qDebug()<<"error opening colorlist.txt";
    else{
        //qDebug()<<this->ui->listWidget->size();
        //qDebug()<<this->ui->listWidget->size().height();
        for(int i=0;i<this->ui->listWidget->count();i++){
            int h,s,v;
            QListWidgetItem* item = ui->listWidget->item(i);
            QColor color= item->backgroundColor();
            color.getHsv(&h,&s,&v);
            QString text= item->text();
            fprintf(f,"%s\n",text.toLatin1().data());
            //qDebug()<<text.toLatin1().data();
            //fprintf(f,"%d,%d,%d\n",h,s,v);
        }
        fclose(f);
    }
}

void MainWindow::on_btnLoadColorList_clicked()
{
    for(int i=this->ui->listWidget->count();i>0;i--){
         QListWidgetItem* item = ui->listWidget->takeItem(i-1);
         delete item;
    }

    FILE* f = fopen("colorlist.txt","r");
    if(!f)qDebug()<<"error opening colorlist.txt";
    else{
        int minh,mins,minv;
        int maxh,maxs,maxv;
        fscanf(f,"%d,%d,%d,%d,%d,%d",&minh,&maxh,&mins,&maxs,&minv,&maxv);
        while(!feof(f)){
            QColor color;
            color.setHsv((minh+maxh)/2,(mins+maxs)/2,(minv+maxv)/2);
            QListWidgetItem *colorItem = new QListWidgetItem(QString::asprintf("%d,%d,%d,%d,%d,%d",minh,maxh,mins,maxs,minv,maxv));
            colorItem->setForeground(QBrush(QColor(Qt::white)));
            colorItem->setBackground(QBrush(color));
            this->ui->listWidget->addItem(colorItem);
            fscanf(f,"%d,%d,%d,%d,%d,%d",&minh,&maxh,&mins,&maxs,&minv,&maxv);
        }
        fclose(f);
        this->minHSVColorList.clear();
        this->maxHSVColorList.clear();
        for(int i=0;i<this->ui->listWidget->count();i++){
            int minh,mins,minv;
            int maxh,maxs,maxv;
            QListWidgetItem* item = ui->listWidget->item(i);
            QString text = item->text();
            sscanf(text.toLatin1().data(),"%d,%d,%d,%d,%d,%d",&minh,&maxh,&mins,&maxs,&minv,&maxv);
            this->minHSVColorList.push_back(Scalar(minh,mins,minv));
            this->maxHSVColorList.push_back(Scalar(maxh,maxs,maxv));
        }
    }
}

void MainWindow::on_listWidget_doubleClicked(const QModelIndex &index)
{
    qDebug()<<index.row();
    QListWidgetItem* item = ui->listWidget->takeItem(index.row());
    delete item;
    this->minHSVColorList.clear();
    this->maxHSVColorList.clear();
    for(int i=0;i<this->ui->listWidget->count();i++){
        int minh,mins,minv;
        int maxh,maxs,maxv;
        QListWidgetItem* item = ui->listWidget->item(i);
        QString text = item->text();
        sscanf(text.toLatin1().data(),"%d,%d,%d,%d,%d,%d",&minh,&maxh,&mins,&maxs,&minv,&maxv);
        this->minHSVColorList.push_back(Scalar(minh,mins,minv));
        this->maxHSVColorList.push_back(Scalar(maxh,maxs,maxv));
    }
}

void MainWindow::on_listWidget_clicked(const QModelIndex &index)
{
    //this->colorIndex = index.row();
    //qDebug()<<"clicked";
    int minh,mins,minv;
    int maxh,maxs,maxv;
    QListWidgetItem* item = ui->listWidget->item(index.row());
    QString text = item->text();
    sscanf(text.toLatin1().data(),"%d,%d,%d,%d,%d,%d",&minh,&maxh,&mins,&maxs,&minv,&maxv);
//    minh/=2;
//    maxh/=2;
    this->ui->sliMinH->setValue(minh);
    this->ui->sliMaxH->setValue(maxh);
    this->ui->sliMinS->setValue(mins);
    this->ui->sliMaxS->setValue(maxs);
    this->ui->sliMinV->setValue(minv);
    this->ui->sliMaxV->setValue(maxv);
}

void MainWindow::on_chkTreshold_clicked()
{
    this->minHSVColorList.clear();
    this->maxHSVColorList.clear();
    for(int i=0;i<this->ui->listWidget->count();i++){
        int minh,mins,minv;
        int maxh,maxs,maxv;
        QListWidgetItem* item = ui->listWidget->item(i);
        QString text = item->text();
        sscanf(text.toLatin1().data(),"%d,%d,%d,%d,%d,%d",&minh,&maxh,&mins,&maxs,&minv,&maxv);
        this->minHSVColorList.push_back(Scalar(minh,mins,minv));
        this->maxHSVColorList.push_back(Scalar(maxh,maxs,maxv));
    }
    if(this->ui->chkTreshold->isChecked())
        MainWindow::treshold=true;
    else
        MainWindow::treshold=false;
}



bool MainWindow::tresholdOnlyOne=false;
void MainWindow::on_chkOneColorTresh_toggled(bool checked)
{
    MainWindow::tresholdOnlyOne=checked;
}

int MainWindow::hsvDifference=20;
void MainWindow::on_horizontalSlider_valueChanged(int value)
{
    MainWindow::hsvDifference=value;
    int h,s,v;
    h = this->ui->sliMinH->value();
    s = this->ui->sliMinS->value();
    v = this->ui->sliMinV->value();

    this->ui->sliMinH->setValue(h);
    this->ui->sliMaxH->setValue(h +value*2);
    this->ui->sliMinS->setValue(s);
    this->ui->sliMaxS->setValue(s +value*2);
    this->ui->sliMinV->setValue(v);
    this->ui->sliMaxV->setValue(v +value*2);
}

void MainWindow::on_setHSVSliders(int h,int s,int v){
    this->ui->sliMinH->setValue(h);
    this->ui->sliMaxH->setValue(h);
    this->ui->sliMinS->setValue(s);
    this->ui->sliMaxS->setValue(s);
    this->ui->sliMinV->setValue(v);
    this->ui->sliMaxV->setValue(v);
}

void MainWindow::on_btnAddColor_clicked()
{
    QColor color;
    int h = (this->ui->sliMinH->value()+this->ui->sliMaxH->value())/2;
    int s = (this->ui->sliMinS->value()+this->ui->sliMaxS->value())/2;
    int v = (this->ui->sliMinV->value()+this->ui->sliMaxV->value())/2;
    color.setHsv(h,s,v);
    QListWidgetItem *colorItem = new QListWidgetItem(QString::asprintf("%d,%d,%d,%d,%d,%d",this->ui->sliMinH->value(),
                                                                       this->ui->sliMaxH->value(),
                                                                       this->ui->sliMinS->value(),
                                                                       this->ui->sliMaxS->value(),
                                                                       this->ui->sliMinV->value(),
                                                                       this->ui->sliMaxV->value()));
    colorItem->setForeground(QBrush(QColor(Qt::white)));
    colorItem->setBackground(QBrush(color));
    this->ui->listWidget->addItem(colorItem);

    this->minHSVColorList.clear();
    this->maxHSVColorList.clear();
    for(int i=0;i<this->ui->listWidget->count();i++){
        int minh,mins,minv;
        int maxh,maxs,maxv;
        QListWidgetItem* item = ui->listWidget->item(i);
        QString text = item->text();
        sscanf(text.toLatin1().data(),"%d,%d,%d,%d,%d,%d",&minh,&maxh,&mins,&maxs,&minv,&maxv);
        minh/=2;
        maxh/=2;
        this->minHSVColorList.push_back(Scalar(minh,mins,minv));
        this->maxHSVColorList.push_back(Scalar(maxh,maxs,maxv));
    }
}

int MainWindow::minH;int MainWindow::minS;int MainWindow::minV;
int MainWindow::maxH;int MainWindow::maxS;int MainWindow::maxV;
void MainWindow::on_sliMinH_valueChanged(int value)
{
    MainWindow::minH=value;
}

void MainWindow::on_sliMaxH_valueChanged(int value)
{
    MainWindow::maxH=value;
}

void MainWindow::on_sliMinS_valueChanged(int value)
{
    MainWindow::minS=value;
}

void MainWindow::on_sliMaxS_valueChanged(int value)
{
    MainWindow::maxS=value;
}

void MainWindow::on_sliMinV_valueChanged(int value)
{
    MainWindow::minV=value;
}

void MainWindow::on_sliMaxV_valueChanged(int value)
{
    MainWindow::maxV=value;
}

int MainWindow::erode;
int MainWindow::dilate;
void MainWindow::on_horizontalSlider_2_valueChanged(int value)
{
    MainWindow::erode=value;
}

void MainWindow::on_horizontalSlider_3_valueChanged(int value)
{
    MainWindow::dilate=value;
}

bool MainWindow::squareSelect = false;
void MainWindow::on_chkRectSelect_toggled(bool checked)
{
    MainWindow::squareSelect = checked;
    MainWindow::roi.width=MainWindow::roi.height;
}

int MainWindow::cannyLowSteer;
int MainWindow::cannyHighSteer;

int MainWindow::minHSteer;int MainWindow::minSSteer;int MainWindow::minVSteer;
int MainWindow::maxHSteer;int MainWindow::maxSSteer;int MainWindow::maxVSteer;
int MainWindow::erodeSteer;
int MainWindow::dilateSteer;
void MainWindow::on_sliMinH_Steering_valueChanged(int value)
{
    this->minHSteer = value;
}

void MainWindow::on_sliMaxH_Steering_valueChanged(int value)
{
    this->maxHSteer=value;
}

void MainWindow::on_sliMinS_Steering_valueChanged(int value)
{
    this->minSSteer = value;
}

void MainWindow::on_sliMaxS_Steering_valueChanged(int value)
{
    this->maxSSteer = value;
}

void MainWindow::on_sliMinV_Steering_valueChanged(int value)
{
    this->minVSteer = value;
}

void MainWindow::on_sliMaxV_Steering_valueChanged(int value)
{
    this->maxVSteer = value;
}

void MainWindow::on_sli_lowCanny_valueChanged(int value)
{
    this->cannyLowSteer = value;
}

void MainWindow::on_sli_highCanny_valueChanged(int value)
{
    this->cannyHighSteer = value;
}

void MainWindow::on_sli_Erode_Steering_valueChanged(int value)
{
    this->erodeSteer = value;
}

void MainWindow::on_sli_Dilate_Steering_valueChanged(int value)
{
    this->dilateSteer = value;
}


void MainWindow::on_btn_Play_Video_clicked()
{
    if(this->ui->btn_Play_Video->text()=="Play Video"){
        this->threadVideoPlayer = new ThreadVideoPlayer(this,"out.avi");
        connect(this->threadVideoPlayer,SIGNAL(playerImageChanged()),this,SLOT(on_playerImageChanged()));
        this->threadVideoPlayer->running =true;
        this->threadVideoPlayer->start();
        this->ui->btn_Play_Video->setText("Stop Video");
    }else{
        this->threadVideoPlayer->running=false;
        QThread::msleep(3000);
        delete this->threadVideoPlayer;
        this->threadVideoPlayer=nullptr;
        this->ui->btn_Play_Video->setText("Play Video");
    }
}

void MainWindow::on_playerImageChanged(){
    if(this->tabIndex==2){
        cv::Mat *resized = MainWindow::storeGetImage(nullptr,"EX");
        if(resized==nullptr){
            qDebug()<<"Error. Cannot play video";
            return;
        }
        cv::cvtColor(*resized, *resized, CV_GRAY2RGB);
        QImage imdisplay((uchar*)resized->data, resized->cols, resized->rows, resized->step, QImage::Format_RGB888); //Converts the CV image into Qt standard format
        int w = this->ui->lblSteeringProcessing->width();
        int h = this->ui->lblSteeringProcessing->height();
        this->ui->lblSteeringProcessing->setPixmap(QPixmap::fromImage(imdisplay).scaled(w,h,Qt::KeepAspectRatio));
        delete resized;
    }
}

void MainWindow::on_btnSteeringParams_clicked()
{
    FILE *f = fopen("steeringParams.txt","w");
    if(!f){
        qDebug()<<"error while opening steeringParams.txt";
    }else{
        fprintf(f,"%d %d %d %d %d %d\n",minHSteer,maxHSteer,minSSteer,maxSSteer,minVSteer,maxVSteer);
        fprintf(f,"%d %d %d %d\n",cannyLowSteer,cannyHighSteer,erodeSteer,dilateSteer);
        fprintf(f,"%d %d %d %d",ROIX1Steer,ROIY1Steer,ROIX2Steer,ROIY2Steer);
        fclose(f);
    }
}

void MainWindow::loadSteeringParams(){
    FILE *f = fopen("steeringParams.txt","r");
    if(!f){
        qDebug()<<"error while opening steeringParams.txt";
    }else{
        fscanf(f,"%d %d %d %d %d %d\n",&minHSteer,&maxHSteer,&minSSteer,&maxSSteer,&minVSteer,&maxVSteer);
        fscanf(f,"%d %d %d %d\n",&cannyLowSteer,&cannyHighSteer,&erodeSteer,&dilateSteer);
        fscanf(f,"%d %d %d %d",&ROIX1Steer,&ROIY1Steer,&ROIX2Steer,&ROIY2Steer);
        this->ui->sliMinH_Steering->setValue(minHSteer);
        this->ui->sliMaxH_Steering->setValue(maxHSteer);
        this->ui->sliMinS_Steering->setValue(minSSteer);
        this->ui->sliMaxS_Steering->setValue(maxSSteer);
        this->ui->sliMinV_Steering->setValue(minVSteer);
        this->ui->sliMaxV_Steering->setValue(maxVSteer);

        this->ui->sli_highCanny->setValue(cannyHighSteer);
        this->ui->sli_lowCanny->setValue(cannyLowSteer);
        this->ui->sli_Erode_Steering->setValue(erodeSteer);
        this->ui->sli_Dilate_Steering->setValue(dilateSteer);

        this->ui->sliROIx1Steer->setValue(ROIX1Steer);
        this->ui->sliROIy1Steer->setValue(ROIY1Steer);
        this->ui->sliROIx2Steer->setValue(ROIX2Steer);
        this->ui->sliROIy2Steer->setValue(ROIY2Steer);
        fclose(f);
    }
}

int MainWindow::ROIX1Steer=0;int MainWindow::ROIX2Steer=640;int MainWindow::ROIY1Steer=0;int MainWindow::ROIY2Steer=480;


void MainWindow::on_sliROIx1Steer_valueChanged(int value)
{
    MainWindow::ROIX1Steer = value;
}

void MainWindow::on_sliROIy1Steer_valueChanged(int value)
{
    MainWindow::ROIY1Steer = value;
}

void MainWindow::on_sliROIx2Steer_valueChanged(int value)
{
    MainWindow::ROIX2Steer = value;
    qDebug()<<MainWindow::originalImage.cols;
}

void MainWindow::on_sliROIy2Steer_valueChanged(int value)
{
    MainWindow::ROIY2Steer = value;
}

void MainWindow::on_btnConvertToTraining_clicked()
{
    for(int i=0;i<9;i++){
        char filename[100];
        sprintf(filename,"../out%d.avi",i);
        VideoCapture capture(filename);
        char outputFilename[100];
        sprintf(outputFilename,"../outProcessed%d.avi",i);
        Rect roi = Rect(Point(MainWindow::ROIX1Steer,MainWindow::ROIY1Steer),Point(MainWindow::ROIX2Steer,MainWindow::ROIY2Steer));
        VideoWriter video(outputFilename, CV_FOURCC('M','J','P','G'), 15, Size(roi.width, roi.height), true);
        Mat img;
        capture >>img;

        while(img.rows!=0 && img.cols!=0){
            ThreadVideoPlayer::processSteeringImage(img);
            cvtColor(img,img,CV_GRAY2BGR);
            video.write(img);
            capture>>img;
        }
        video.release();
    }
}

tiny_dnn::network<tiny_dnn::sequential> MainWindow::nn;
void MainWindow::construct_net(){

    int outputSize = 9;

    this->nn = tiny_dnn::network<tiny_dnn::sequential>();
    this->nn <<convolutional_layer<relu>(64,32,5,1,9);//49*19 size, 5 kernel size, 1 input channel, 9 output channels
    this->nn<<average_pooling_layer<relu>(60,28,9,2);
    this->nn<<convolutional_layer<relu>(30,14,3,9,18);
    this->nn<<average_pooling_layer<relu>(28,12,18,2);
    this->nn<<fully_connected_layer<sigmoid>(14*6*18,outputSize);
//    this->nn = tiny_dnn::network<tiny_dnn::sequential>();
//    this->nn <<fully_connected_layer<sigmoid>(64*32,50);
//    this->nn<<fully_connected_layer<sigmoid>(50,outputSize);
}

void MainWindow::on_btnTrainSteeringNN_clicked()
{
    if(this->ui->btnTrainSteeringNN->text()=="Train Steering NN"){
        if(this->net==nullptr){
            QMessageBox msgBox;
            msgBox.setText("Create NN?");
            msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::Cancel);
            int res = msgBox.exec();
            if(res == QMessageBox::Yes){
                construct_net();
            }else{
                return;
            }
        }else if(this->net!=nullptr){
            QMessageBox msgBox;
            msgBox.setText("ReWrite NN?");
            msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::Cancel);
            int res = msgBox.exec();
            if(res == QMessageBox::Yes){
                construct_net();
            }else{

            }
        }
        //this->threadTrainer= new ThreadTrainer(this,trSteering = true,trDetection = false);
        this->threadTrainer= new ThreadTrainer(this, true, false);
        connect(this->threadTrainer,SIGNAL(FinishedTrainingNeural()),this,SLOT(on_FinishedTrainingNeural()));
        this->threadTrainer->start();
        this->ui->btnTrain->setText("Stop Train Steering NN");
    }else {
        this->threadTrainer->running=false;
        //FOR NOW JUST BREAK THE EXECUTION OF THE PROGRAM
    }
}

void MainWindow::on_btnLoadSteeringNN_clicked()
{
//    arma::Mat<int> sizes(1,3);
//    sizes(0,0)=49*19;
//    sizes(0,1)=50;
//    sizes(0,2)=9;
//    MainWindow::net = new Neural(sizes);
//    for(int i=0;i<MainWindow::net->weights.size();i++){
//        char buffer[50];
//        sprintf(buffer,"w%d.mat",i);
//        MainWindow::net->weights.at(i).load(buffer);
//        sprintf(buffer,"b%d.mat",i);
//        MainWindow::net->biases.at(i).load(buffer);
//    }
    this->nn.load("nn-aug-first-paper");
    QMessageBox msg;
    msg.setText("Neural Net loaded");
    msg.exec();
}

void MainWindow::on_btnEngageAutopilot_clicked()
{
    if(this->ui->btnEngageAutopilot->text()=="Engage Autopilot"){
        this->threadAutopilot = new ThreadAutopilot(this);
        connect(this->threadAutopilot,SIGNAL(predictedAngle(int)),this,SLOT(on_predictedAngle(int)));
        this->threadAutopilot->running = true;
        this->threadAutopilot->start();
        this->ui->btnEngageAutopilot->setText("Disengage Autopilot");
    }else{
        this->threadAutopilot->running = false;
        QThread::msleep(3000);
        delete this->threadAutopilot;
        this->threadAutopilot = nullptr;
        qDebug()<<"Pilot disengaged";
        this->ui->btnEngageAutopilot->setText("Engage Autopilot");
    }
}

void MainWindow::on_predictedAngle(int angle){
    char format[200];
    sprintf(format,"Speed %d\n Predicted Angle %d",this->ui->verticalSlider->value(),angle);
    this->ui->lblSpeedAngle->setText(format);
}

void MainWindow::on_btnTrackVehicles_clicked()
{
    if(this->ui->btnTrackVehicles->text()=="Track Vehicles"){
        this->threadVehicleTracking = new ThreadVehicleTracing(this);
        this->threadVehicleTracking->running=true;
        this->threadVehicleTracking->start();
        this->ui->btnTrackVehicles->setText("Stop Tracking");
    }else{
        this->threadVehicleTracking->running= false;
        QThread::msleep(3000);
        delete this->threadVehicleTracking;
        this->threadVehicleTracking = nullptr;
        this->ui->btnTrackVehicles->setText("Track Vehicles");
    }
}

double MainWindow::scaleFactor = 1.1;
int MainWindow::minNeighbours = 3;
int MainWindow::flag = 0;
int MainWindow::minSize = 10;
int MainWindow::maxSize = 1000;
void MainWindow::on_doubleSpinBox_2_valueChanged(double arg1)
{
    MainWindow::scaleFactor = arg1;
}

void MainWindow::on_spinBox_valueChanged(int arg1)
{
    MainWindow::minNeighbours = arg1;
}

void MainWindow::on_checkBox_clicked(bool checked)
{
    if(checked){
        MainWindow::flag = CV_HAAR_SCALE_IMAGE;
    }else{
         MainWindow::flag = 0;
    }
}

void MainWindow::on_spinBox_2_valueChanged(int arg1)
{
    MainWindow::minSize=arg1;
}

void MainWindow::on_spinBox_3_valueChanged(int arg1)
{
    MainWindow::maxSize=arg1;
}

int MainWindow::outputs = 0;
void MainWindow::on_btnAugmentData_clicked()
{
    int numberOfVideos = 4;
    for(int i=0;i<numberOfVideos;i++){
        qDebug()<<"dfgsdf";
        char defFilename[100];
        sprintf(defFilename,"../def%d.txt",i);


        char processedImages[100];
        sprintf(processedImages,"../out%d.avi",i);
        VideoCapture capture(processedImages);

        augmentData(defFilename,capture);



    }
}

void MainWindow::augmentData(char* defFilename,VideoCapture &capture){
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_int_distribution<> angleDistr(-4, 4); // define the range
    std::uniform_int_distribution<> translationDistr(-4, 4); // define the range
    std::uniform_int_distribution<> brighnessDistr(-90, 90); // define the range

    VideoWriter outputVideo;
    char outputFileVideo[100];
    char defFilenameAugmented[100];
    sprintf(defFilenameAugmented,"../defAugmented%d.txt",outputs);
    sprintf(outputFileVideo,"../outAugmented%d.avi",outputs++);

    //Rect rect = Rect(Point(MainWindow::ROIX1Steer,MainWindow::ROIY1Steer),Point(MainWindow::ROIX2Steer,MainWindow::ROIY2Steer));
    outputVideo.open(outputFileVideo, CV_FOURCC('M','J','P','G'), 15, Size(640,480), true);
    if (!outputVideo.isOpened())
    {
        cout  << "Could not open the output video for write: " <<endl;
        return ;
    }
    qDebug()<<"runnig dfgfaugemnting";
    std::string line;
    std::ifstream infile(defFilename);
    if(!infile){
        qDebug("no def file found!!!");
    }else{
        FILE *defAugmented = fopen(defFilenameAugmented,"w");

        Mat img;
        capture>>img;
        int processedImages = 0;

        while (std::getline(infile, line)){

            //img = img(rect);
            Mat copyImage;
            for(int i=0;i<3;i++){

                double brigthness = brighnessDistr(eng);
//                double angle = angleDistr(eng);
//                double dx = translationDistr(eng);
//                double dy = translationDistr(eng);
                img.copyTo(copyImage);
                cvtColor(copyImage,copyImage,CV_BGR2HSV);
                copyImage +=Scalar(0,0,brigthness);
                cvtColor(copyImage,copyImage,CV_HSV2BGR);

                //copyImage = MainWindow::rotate_and_crop(angle,copyImage);
                //copyImage = MainWindow::translateImg(copyImage,dx,dy);
                //cv::resize(copyImage,copyImage,Size(640,480));
                fprintf(defAugmented,line.c_str());
                fprintf(defAugmented,"\n");
                outputVideo.write(copyImage);
            }
            for(int i=0;i<3;i++){

                double brigthness = brighnessDistr(eng);

                img.copyTo(copyImage);
                cv::flip(copyImage, copyImage, 1);
                cvtColor(copyImage,copyImage,CV_BGR2HSV);
                copyImage +=Scalar(0,0,brigthness);
                cvtColor(copyImage,copyImage,CV_HSV2BGR);

                //copyImage = MainWindow::rotate_and_crop(angle,copyImage);
                //copyImage = MainWindow::translateImg(copyImage,dx,dy);
                //cv::resize(copyImage,copyImage,Size(640,480));
                int value = atoi(line.c_str());
                int A = 4000;
                int B = 8000;
                int a = 8000;
                int b = 4000;
                value = (value-A)*(b-a)/(B-A)+a;
                char text[200];sprintf(text,"%d",value);
                fprintf(defAugmented,text);
                fprintf(defAugmented,"\n");
                outputVideo.write(copyImage);
            }


            qDebug()<<"processed " << processedImages++;
            capture>>img;
        }
        infile.close();
        fclose(defAugmented);
        outputVideo.release();
    }
}

Mat MainWindow::translateImg(Mat &img, int offsetx, int offsety){
    Mat trans_mat = (Mat_<double>(2,3) << 1, 0, offsetx, 0, 1, offsety);
    warpAffine(img,img,trans_mat,img.size(),
               cv::INTER_LINEAR,
               cv::BORDER_CONSTANT,
               cv::Scalar(0, 0, 0));
    return img;
}

cv::Rect MainWindow::getLargestRect(double imageWidth, double imageHeight, double rotAngDeg, int type){
    Rect rect;
    double rotateAngleDeg = std::fmod(rotAngDeg, 180);
    if (rotateAngleDeg < 0){
        rotateAngleDeg += 360;
        rotateAngleDeg = std::fmod(rotateAngleDeg, 180);;
    }
    double imgWidth = imageWidth;
    double imgHeight = imageHeight;
    if (rotateAngleDeg == 0 || rotateAngleDeg == 180){
        //Angle is 0, no change needed
        rect = Rect(0, 0, (int)imgHeight, (int)imgWidth);
        return rect;
    }
    if (rotateAngleDeg > 90){
        // Angle > 90 therefore angle = 90 - ("+rotateAngleDeg+" - 90) = "+(90 - (rotateAngleDeg - 90))
        rotateAngleDeg = 90 - (rotateAngleDeg - 90);
    }
    double rotateAngle = (rotateAngleDeg*M_PI) / 180.;
    double sinRotAng = sin(rotateAngle);
    double cosRotAng = cos(rotateAngle);
    double tanRotAng = tan(rotateAngle);
    // Point 1 of rotated rectangle
    double x1 = sinRotAng * imgHeight;
    double y1 = 0;
    // Point 2 of rotated rectangle
    double x2 = cosRotAng * imgWidth + x1;
    double y2 = sinRotAng * imgWidth;
    // Point 3 of rotated rectangle
    double x3 = x2 - x1;
    double y3 = y2 + cosRotAng * imgHeight;
    // Point 4 of rotated rectangle
    double x4 = 0;
    double y4 = y3 - y2;
    // MidPoint of rotated image
    double midx = x2 / 2;
    double midy = y3 / 2;

    // Angle for new rectangle (based on image width and height)
    double imgAngle = atan(imgHeight / imgWidth);
    double imgRotAngle = atan(imgWidth / imgHeight);
    double tanImgAng = tan(imgAngle);
    double tanImgRotAng = tan(imgRotAngle);
    // X Point for new rectangle on bottom line
    double ibx1 = midy / tanImgAng + midx;
    double ibx2 = midy * tanImgAng + midx;

    // First intersecting lines
    // y = ax + b  ,  y = cx + d  ==>  x = (d - b) / (a - c)
    double a = y2 / x3;
    double b = tanRotAng * -x1;
    double c = -imgHeight / imgWidth;
    double d = tanImgAng * ibx1;

    // Intersecting point 1
    double ix1 = (d - b) / (a - c);
    double iy1 = a * ix1 + b;

    // Second intersecting lines
    c = -imgWidth / imgHeight;
    d = tanImgRotAng * ibx2;

    // Intersecting point 2
    double ix2 = (d - b) / (a - c);
    double iy2 = a * ix2 + b;

    // Work out smallest rectangle
    double radx1 = abs(midx - ix1);
    double rady1 = abs(midy - iy1);
    double radx2 = abs(midx - ix2);
    double rady2 = abs(midy - iy2);
    // Work out area of rectangles
    double area1 = radx1 * rady1;
    double area2 = radx2 * rady2;

    Rect rect1 = Rect((int)round(midx - radx1), (int)round(midy - rady1), (int)round(radx1 * 2), (int)round(rady1 * 2));
    Rect rect2 = Rect((int)round(midx - radx2), (int)round(midy - rady2), (int)round(radx2 * 2), (int)round(rady2 * 2));
    switch (type) {
    case 0: rect = (area1 > area2 ? rect1 : rect2); break;
    case 1: rect = (area1 < area2 ? rect1 : rect2); break;
    case 2: rect = (radx1 > radx2 ? rect1 : rect2); break;
    case 3: rect = (rady1 > rady2 ? rect1 : rect2); break;
    }

    return rect;
}

cv::Mat MainWindow::rotate_and_crop(double angle,cv::Mat &mat){
    if(angle==0)return mat;
    cv::Point2f center(mat.cols / 2.0, mat.rows / 2.0);
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Rect bbox = cv::RotatedRect(center, mat.size(), angle).boundingRect();
    rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
    rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;
    cv::Mat dst;
    cv::warpAffine(mat, dst, rot, bbox.size());
    Rect rect = getLargestRect(mat.cols, mat.rows, angle, 0);
    if(rect.width==mat.cols )rect.width -=1;
    if(rect.height==mat.rows)rect.height-=1;
    cv::Mat cropped = dst(rect);
    return cropped;
}
