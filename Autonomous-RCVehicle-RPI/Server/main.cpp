#include <QCoreApplication>
#include "Server.h"
#include <QDebug>

int main(int argc, char *argv[])
{
    if(argc==2){
        Server::compression = atoi(argv[1]);

    }
    else{
        qDebug()<< "Warning: default 50 compression level";
    }
    QCoreApplication a(argc, argv);

    Server s;
    return a.exec();
}
