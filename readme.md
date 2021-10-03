# Background estimation tool based on Stauffer-Grimson Gaussian Mixtures.

Compile:
```
make
```
Clean:
```
make cean
```

Alternatively, you can use bg.pro with qtcreator and qmake.

Has been tested with opencv4 and g++.

To show usage:
```
./bg
```
Sample run:
```
./bg method SG resize 300 300 openclose 2 2 1 1
```
