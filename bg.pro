TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt
CONFIG += -Wnosign-compare
SOURCES += main.cpp

LIBS += -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_videoio

HEADERS += \
    sgb.h
