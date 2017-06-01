#include "keylogger.h"
#include <stdio.h>
#include <QDebug>
#include "GUI/mainwindow.h"
#define NOMINMAX
#include <windows.h>
#include <winuser.h>

bool KeyLogger::wheelChanged=false;
KeyLogger::KeyLogger(QObject *parent) : QThread(parent)
{
    this->running=true;


}

LRESULT CALLBACK MouseHookProc(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode >= 0) {
        if (wParam == WM_RBUTTONDOWN) printf("right mouse down\n");
        if (wParam == WM_RBUTTONUP) printf("right mouse up\n");
        //if (wParam == WM_MOUSEMOVE)printf("mouse move\n");

        //using namespace MouseLog;
        MSLLHOOKSTRUCT * pMouseStruct = (MSLLHOOKSTRUCT *)lParam;
        if (pMouseStruct != NULL)
            {
                if (wParam == WM_MOUSEWHEEL)
                {
                    if (HIWORD(pMouseStruct->mouseData) == 120) {MainWindow::accelerationPower++;}

                    else MainWindow::accelerationPower--;
                    KeyLogger::wheelChanged=true;
                }
                if (wParam == WM_MBUTTONDOWN)
                    qDebug("btn scroll");

                //qDebug("Mouse position X = %d  Mouse Position Y = %d\n", pMouseStruct->pt.x, pMouseStruct->pt.y);
            }
        POINT p;
        if (GetCursorPos(&p))
        {

            //qDebug("x%d y%d\n", p.x, p.y);
        }

    }
    return CallNextHookEx(0, nCode, wParam, lParam);
}

LRESULT CALLBACK KeyboardHookProc(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode >= 0) {
        for (int i = 8; i <= 190; i++){
            if (GetAsyncKeyState(i) == -32767){
                qDebug()<<"key presed"<<i;
                if(i==87){//W
                    ThreadSaver::saving=true;
                }else if(i==83){//S
                    ThreadSaver::saving=false;
                }
            }
        }
    }
    return CallNextHookEx(0, nCode, wParam, lParam);
}

void KeyLogger::run(){
    HHOOK mousehook = SetWindowsHookEx(WH_MOUSE_LL, MouseHookProc, NULL, 0);
    HHOOK keyboardhook = SetWindowsHookEx(WH_KEYBOARD_LL, KeyboardHookProc, NULL, 0);
    while(1){
        MSG msg;
        if (GetMessage(&msg, 0, 0, 0)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        if(KeyLogger::wheelChanged){
            qDebug()<<"changed";
            wheelChanged =false;
            emit mouseWheelChanged();
        }

    }
}

