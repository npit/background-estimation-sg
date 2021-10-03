all:
	g++ main.cpp -I/usr/include/opencv4  -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_videoio -o bg
clean:
	rm bg
