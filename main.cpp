// C++ inclusions
#include <iostream>
#include <array>

// opencv inclusions
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

#include "sgb.h"

/**
 * @brief Provides information about a Mat object.
 * @param m
 */
void matinfo(cv::Mat &m)
{
    cout << "size " << m.rows << " x " << m.cols << " chan: " << m.channels() << " type : " << m.type() << endl;
}

/**
 * @brief Displays input frame on the "monitor" window.
 * @param frame
 */
void monitor(cv::Mat & frame)
{
    // show the frame on monitor window
    cv::imshow("monitor",frame);
}

/**
 * @brief Displays input frame on the "output" window.
 * @param frame
 */
void background(cv::Mat & frame)
{
    // show the frame on monitor window
    cv::imshow("raw output",frame);
}
/**
 * @brief Displays the processing output on the "processed output" window
 * @param frame
 */
void processed(cv::Mat & frame)
{
    cv::imshow("processed output",frame);
}

/**
 * @brief The options class receives, modifies by user input and stores CLI arguments
 *
 * method: the background estimation method to use
 * imageSize: the size in pixels to resize the image
 * numErode: how many times to apply erosion
 * numDilate: how many times to apply dilation
 * strelSize: structuring element dimension
 */

class options
{

public:
    /// print current settings to stdout
    void info()
    {
        if(!doUpdate) return;
        cout << "Options:";

        cout << "\nOpen-reconstr strel/num: " << strelOpenDim << " " << numErode << endl;
        cout << "Open-reconstr strel/num: " << strelCloseDim << " " << numDilate << endl;
        doUpdate = false;
    }

    static const int MAXITER = 25;
    /// resize the input frame
    void resize(cv::Mat & frame)
    {
        if(resizeImage)
            cv::resize(frame,frame,cv::Size(imageSize.first,imageSize.second));
    }
    /// set the imageSize options by peeking at the frame
    void peek(cv::Mat & frame)
    {
        if(resizeImage) return;
        imageSize.first = frame.cols;
        imageSize.second = frame.rows;
    }
    /// checks whether input dimension is valid
    bool goodDim(int v)
    {
        bool val =  v>0 && v < max(imageSize.first/2,imageSize.second/2);
        if(val) doUpdate=true;
        return val;
    }
    /// checks whether input iteration is valid
    bool goodIter(int v)
    {
        bool val = v>=0 && v<=options::MAXITER;
        if(val) doUpdate=true;
        return val;
    }
    /// parse command line arguments
    bool argparse(int argc,char **argv)
    {
        // defaults
        doUpdate = false;
        useVideo = false;
        resizeImage = false;
        imageSize = pair<int,int>(-1,-1);
        method = SGb::name;
        strelOpenDim = 3;
        strelCloseDim = 5;
        numErode = 3;
        numDilate = 3;

        // iterate over arguments. Each case may advance the argument index.
        for(int i=1;i<argc;++i)
        {
            cout << "Parsing CLI argument " << argv[i] << endl;
            if(string(argv[i]) == "resize")
            {
                // get next 2 arguments
                ++i; imageSize.first = atoi(argv[i]);
                ++i; imageSize.second = atoi(argv[i]);
                resizeImage = true;
                cout << "Set resize to " << imageSize.first << " x " << imageSize.second << endl;
            }
            else if(string(argv[i]) == "method")
            {
                ++i;
                method = string(argv[i]);
                cout << "Set method " << method << endl;
            }
            else if(string(argv[i]) == "video")
            {
                useVideo = true;
                ++i;
                videoFile = string(argv[i]);
            }
            else if(string(argv[i]) == "openclose")
            {

                ++i;
                strelOpenDim= stoi(argv[i]);
                ++i;
                strelCloseDim= stoi(argv[i]);
                ++i;
                numErode= stoi(argv[i]);
                ++i;
                numDilate= stoi(argv[i]);
                cout << "Set openclose arguments " << strelOpenDim << " "  << strelCloseDim<< " " << numErode << " "<< numDilate << endl;
            }
            else
            {
                // the argument has to be for the background estimation method
                methodOpts.push_back(string(argv[i]));

            }

        }
        return true;
    }
    /// print accepted usage
    static void usage()
    {
        cout << "General options:" << endl;
        cout << "video <videoPath>" << endl;
        cout << "openclose <strelDelete> <strelFill> <numDelete> <numFIll>" << endl;
        cout << "resize <width> <height>" << endl;
        cout << "method <SG>" << endl;
        cout << "openclose <strelsize><numErode><numDilate>" << endl;

        cout << "<method parameters>" << endl;
    }
    /**
     * @brief usage_controls Display control interface of main.
     */
    static void usage_controls()
    {
        cout << "a/s : increase/decrease opening by reconstruction strel" << endl;
        cout << "d/f : increase/decrease closing by reconstruction strel" << endl;
        cout << "z/x : increase/decrease opening by reconstruction initial iterations" << endl;
        cout << "c/v : increase/decrease closing by reconstruction initial iterations" << endl;
    }
    /// Container to pass on options to the method that will run.
    vector<string> methodOpts;
    /// Specified image size, std::pair.
    pair<int,int> imageSize;
     /// Should resize image or not.
    bool resizeImage,
        /// Should get input from a video file.
        useVideo,
        /// Should print updated parameters.
        doUpdate;
    /// Opening by reconstruction strel dimension.
    int strelOpenDim,
    /// Closing by reconstruction strel dimension.
    strelCloseDim,
    /// How many times to erode in opening by reconstruction.
    numErode,
    /// How many times to dilate in closing by reconstruction.
    numDilate;
    /// Method to use.
    string method,
    /// Video path.
    videoFile;
};

/**
 * @brief Performs opening by reconstruction
 * @param input is the image to apply the operation on
 * @param Opts the options object
 *
 * This function first erodes the image Opts.numErode times, then dilates
 * with the same strel until no change occurs.
 */
void openrec(cv::Mat & input,options Opts)
{
    if(Opts.numErode == 0) return;
    cv::Mat output = cv::Mat(input), previousConjunction;
    // define the structuring element
    cv::Mat strel = cv::Mat::ones(Opts.strelOpenDim,Opts.strelOpenDim,input.type());
    // erode
    cv::erode(input,output,strel,cv::Point(-1,-1),Opts.numErode);

    // dilate
    int numIter = 0;
    while(true)
    {
        ++numIter;
        cv::dilate(output,output,strel);
        cv::bitwise_and(output,input,output);

        if(previousConjunction.empty())
        {
            previousConjunction = output;
            continue;
        }
        else
        {
            // compare the matrices for equality
            bool equal = std::equal(previousConjunction.begin<uchar>(), previousConjunction.end<uchar>(), output.begin<uchar>());
            if(equal)
            {
                previousConjunction.copyTo(input);
                break;
            }
            continue;
        }
    }

}
/**
 * @brief Performs closing by reconstruction
 * @param input is the image to apply the operation on
 * @param Opts the options object
 *
 * This function first dilates the image Opts.numErode times, then erodes
 * with the same strel until no change occurs.
 */
void closerec(cv::Mat & input,options Opts)
{
    if(Opts.numDilate == 0) return;
    cv::Mat output = cv::Mat(input), previousConjunction;
    int numIter = 0;
    // fill holes: erode, then dilate
    cv::Mat strel = cv::Mat::ones(Opts.strelCloseDim,Opts.strelCloseDim,input.type());

    cv::dilate(output,output,strel,cv::Point(-1,1),Opts.numDilate);
    while(true)
    {
        ++numIter;
        cv::erode(output,output,strel);
        cv::bitwise_and(output,input,output);

        if(previousConjunction.empty())
        {
            previousConjunction = output;
            continue;
        }
        else
        {
            // compare the matrices for equality
            bool equal = std::equal(previousConjunction.begin<uchar>(), previousConjunction.end<uchar>(), output.begin<uchar>());
            if(equal)
            {
                previousConjunction.copyTo(input);
                break;
            }
            continue;
        }
    }
}

/**
 * @brief Apply post-processing operations to the input
 * @param input is the image to apply the operations to
 * @param Opts is the options object
 */
void post_process(cv::Mat & input,options Opts)
{

        openrec(input,Opts);
        closerec(input,Opts);

}



/**
 * @brief Wrapper for all the usage prints.
 */
void usage()
{
cout << "bg <method> <method parameters>" << endl;
options::usage();
SGb::usage();
}

/**
 * @brief The main function
 * @param argc CLI argument
 * @param argv CLI argument
 * @return
 */
int main(int argc, char ** argv)
{

    // invalid args
    if(argc<2)
    {
        usage();
        return EXIT_FAILURE;
    }
    // declare options class, parse arguments
    options Opts;
    if(Opts.argparse(argc,argv) == false)
    {
        cerr << "Arguments error." << endl;
        usage();
        return EXIT_FAILURE;
    }

    // connect to the default webcam or read video
    cv::VideoCapture camera;
    bool openSuccess;
    if(Opts.useVideo) openSuccess = camera.open(Opts.videoFile);
    else openSuccess = camera.open(0);
    if(! openSuccess)
    {
        if(Opts.useVideo)
            cerr << "Failed to open video file : " << Opts.videoFile << "." << endl;
        else
            cerr << "Failed to connect to default webcam." << endl;
        return EXIT_FAILURE;
    }

    cout << "Connected to video source." << endl;
    // create monitor and output windows
    cv::namedWindow("monitor", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("raw output", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("processed output", cv::WINDOW_AUTOSIZE);

    cv::Mat frame, output;

    // peek video dimension, potentially resize
    camera.read(frame);
    Opts.peek(frame);

    Opts.resize(frame);

    // future expansions should switch() here
    // or construct method abstractions

    // initialize the SG method
    SGb bg(Opts.imageSize.first,Opts.imageSize.second);
    if(!bg.initialize(Opts.methodOpts,frame))
    {
        cerr << " Failed to initialize SG background estimation method. Exiting." << endl;
        return EXIT_FAILURE;
    }

    // display usage
    options::usage_controls();
    // infinite loop
    while(1)
    {
        // get frame from source
        camera.read(frame);
        // convert to grayscale
        cv::cvtColor(frame,frame,cv::COLOR_BGR2GRAY);
        // potentially resize
        Opts.resize(frame);
        // monitor regular feed
        monitor(frame);

        // Control
        // ---------------------------------
        int key = cv::waitKey(1);

        if(key!=-1)
            cout << "key :" << key << " char : " << (char)key << endl;
        // pause
        if((char) key == 'p')
        {
            cout << "paused" ;
            key = cv::waitKey(0);
            cout << " - unpaused" << endl;

        }

        // thresh
        if((char)      key == 't')bg.increaseThreshold();
        else if((char) key == 'r')bg.decreaseThreshold();
        // learning rate
        else if((char) key == 'k')bg.increaseLR();
        else if((char) key == 'j')bg.decreaseLR();
        // strel Open-reconstr
        else if((char) key == 'a')(Opts.goodDim(Opts.strelOpenDim-1)) ? --(Opts.strelOpenDim) : 0 ;
        else if((char) key == 's')(Opts.goodDim(Opts.strelOpenDim+1)) ? ++(Opts.strelOpenDim) : 0;
        // strel Close-reconstr
        else if((char) key == 'd')(Opts.goodDim(Opts.strelCloseDim-1)) ? --(Opts.strelCloseDim) : 0;
        else if((char) key == 'f')(Opts.goodDim(Opts.strelCloseDim+1)) ? ++(Opts.strelCloseDim) : 0;
        // num erode in open-reconstr
        else if((char) key == 'z')(Opts.goodIter(Opts.numErode-1)) ? --(Opts.numErode) : 0;
        else if((char) key == 'x')(Opts.goodIter(Opts.numErode+1)) ? ++(Opts.numErode) : 0;
        // num dilate in close-reconstr
        else if((char) key == 'c')(Opts.goodIter(Opts.numDilate-1)) ? --(Opts.numDilate) : 0;
        else if((char) key == 'v')(Opts.goodIter(Opts.numDilate+1)) ? ++(Opts.numDilate) : 0;

        Opts.info();
        if(key == 27 || ((char) key == 'q') || ((char) key == 'Q')) break;
        // ---------------------------------

        // produce & show processed feed
        // produce the segmentation output
        output = bg.process(frame);
        // show the output frame
        background(output);
        // post-processing
        post_process(output,Opts);
        // show the post-processed image
        processed(output);

    }
    cout << "Exiting." << endl;
    return 0;
}

