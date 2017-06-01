#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "tiny_dnn/tiny_dnn.h"
#include <iostream>
#include <vector>
#include "tiny_dnn/tiny_dnn.h"
#include <opencv2/core/core.hpp>
#include <Windows.h>
#include <windows.h>
#include <winuser.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/types.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <armadillo>
#include <vector>
using namespace arma;
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace std;
namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_pushButton_clicked();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
