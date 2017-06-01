#include "labelselectdataset.h"
#include <qlabel.h>
#include <QMouseEvent>
#include <QTransform>
#include <QLabel>
#include <QPainter>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
char LabelSelectDataset::vector[10];
int LabelSelectDataset::index=0;
LabelSelectDataset::LabelSelectDataset(QWidget *parent) : QLabel(parent) {
    setMouseTracking(true);
    strcpy(LabelSelectDataset::vector,"1 0");
    this->selectedXY = false;
}

void LabelSelectDataset::mouseMoveEvent(QMouseEvent *e) {
    QTransform t;
    t.scale(1, -1);
    t.translate(0, -height() + 1);
    QPoint pos = (*e).pos() * t;
    if(MainWindow::squareSelect==true || this->selectedXY==false){
        MainWindow::roi.x = pos.x();
        MainWindow::roi.y = 480-pos.y();
    }else if(this->selectedXY==true){
        MainWindow::roi.width = pos.x() - MainWindow::roi.x;
        MainWindow::roi.height = (480-pos.y()) - MainWindow::roi.y;
    }
    emit trainRoiChanged();
}

void LabelSelectDataset::mouseReleaseEvent(QMouseEvent *e){
    Mat *image = MainWindow::storeGetImage(nullptr,"EX");
    char fullName[100];
    if(MainWindow::squareSelect==true){
        for(double deg = -12;deg<12;deg+=6){
            for(double alpha = 0.6;alpha<=1.2;alpha+=0.6){
                for(double beta = 1;beta<=64;beta+=31){

                    Mat roi = (*image)(MainWindow::roi);
                    roi = this->rotate_and_crop(deg,roi);
                    roi = brightness_and_contrast(roi,alpha,beta);

                    sprintf(fullName, "trainingImages/img%d.jpg", LabelSelectDataset::index);
                    imwrite(fullName, roi);
                    sprintf(fullName, "trainingImages/img%d.jpg %s", LabelSelectDataset::index++,LabelSelectDataset::vector);
                    appendToFile("trainingImages/def.txt", fullName);

                }
            }
        }
    }else{
        for(double alpha = 0.6;alpha<=1.2;alpha+=0.6){
            for(double beta = 1;beta<=64;beta+=31){

                Mat roi = (*image)(MainWindow::roi);
                roi = brightness_and_contrast(roi,alpha,beta);
                sprintf(fullName, "trainingImages/img%d.jpg", LabelSelectDataset::index);
                imwrite(fullName, roi);
                sprintf(fullName, "trainingImages/img%d.jpg %s", LabelSelectDataset::index++,LabelSelectDataset::vector);
                appendToFile("trainingImages/def.txt", fullName);

            }
        }
    }
    delete image;
    emit indexChanged();
    this->selectedXY=false;
    qDebug()<<"released";
}

void LabelSelectDataset::wheelEvent ( QWheelEvent * event ){
    if(event->delta()<0){
        MainWindow::roi.width--;
        MainWindow::roi.height--;
    }else{
        MainWindow::roi.width++;
        MainWindow::roi.height++;
    }
    qDebug()<< MainWindow::roi.width;
    emit trainRoiChanged();
}

void LabelSelectDataset::mousePressEvent(QMouseEvent *e)
{
    if(e->buttons() == Qt::LeftButton){

        QTransform t;
        t.scale(1, -1);
        t.translate(0, -height() + 1);
        QPoint pos = (*e).pos() * t;
        MainWindow::roi.x = pos.x();
        MainWindow::roi.y = 480-pos.y();
        this->selectedXY=true;

    }else if(e->buttons()==Qt::RightButton){
        QTransform t;
        t.scale(1, -1);
        t.translate(0, -height() + 1);
        QPoint pos = (*e).pos() * t;
        int x = pos.x();
        int y = 480-pos.y();

        Mat *image = MainWindow::storeGetImage(nullptr,"EX");
        Point3_<uchar>* p = image->ptr<Point3_<uchar> >(y,x);
        QColor hsvColor;
        qDebug()<<p->z <<" " <<p->y <<" "<<p->x;
        hsvColor.setRgb(p->z,p->x,p->y);
        int h,s,v;
        hsvColor.getHsv(&h,&s,&v);
        qDebug()<<h <<" " <<s <<" "<<v;
        emit setHSVSliders(180-h/2,s,v);
        delete image;

    }
}

cv::Mat LabelSelectDataset::rotate_and_crop(double angle,cv::Mat &mat){
    cv::Point2f center(mat.cols / 2.0, mat.rows / 2.0);
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Rect bbox = cv::RotatedRect(center, mat.size(), angle).boundingRect();
    rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
    rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;
    cv::Mat dst;
    cv::warpAffine(mat, dst, rot, bbox.size());
    Rect rect = getLargestRect(mat.rows, mat.cols, angle, 0);
    cv::Mat cropped = dst(rect);
    return cropped;
}

cv::Mat LabelSelectDataset::brightness_and_contrast(cv::Mat &image,double alpha, double beta){
    Mat new_image = Mat::zeros(image.size(), image.type());
        /// Do the operation new_image(i,j) = alpha*image(i,j) + beta
        for (int y = 0; y < image.rows; y++)
        {
            for (int x = 0; x < image.cols; x++)
            {
                for (int c = 0; c < 3; c++)
                {
                    new_image.at<Vec3b>(y, x)[c] =
                        saturate_cast<uchar>(alpha*(image.at<Vec3b>(y, x)[c]) + beta);
                }
            }
        }
        return new_image;
}

void LabelSelectDataset::appendToFile(const char* fileName,const char* textToAppend){
    FILE *file = fopen(fileName, "a");
    if (file == NULL) {
        perror("Error opening file.");
    }
    else {
        fseek(file, 0, SEEK_END);
        fprintf(file, "%s\n", textToAppend);
        fclose(file);
    }
}

cv::Rect LabelSelectDataset::getLargestRect(double imageWidth, double imageHeight, double rotAngDeg, int type){
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
