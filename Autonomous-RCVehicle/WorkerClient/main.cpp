#define NOMINMAX
#include <tiny_dnn/tiny_dnn.h>
#include "GUI/mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    return a.exec();
}
