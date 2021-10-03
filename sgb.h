#ifndef SGB
#define SGB

#define VERBOSE if(verbose)

void matinfo(cv::Mat &m);


/**
 * @brief A class to house the SG estimation for a single pixel
 *
 * This class contains the gaussian mixture model (means, standard deviations, number of gaussians) for a pixel.
 */
class pixel
{
public:

    /**
     * @brief Initialize the class with parameters passed on from the SGB class.
     * @param num number of gaussians
     * @param lr learning rate
     * @param thresh threshold
     * @return successfull initialization
     */
    bool init( int num, double lr, double thresh)
    {

        // set values and init containers

        // operation parameters
        this->learningRate = lr;
        this->maxGaussians = num;
        numGaussians = 3;
        this->thresh = thresh;
        penalizeWeight = false;
        // allocation and initialization of gaussians
        mu = new double[maxGaussians];
        sigma = new double[maxGaussians];
        weights = new double[maxGaussians];
        // check memory
        if(mu == NULL || sigma == NULL || weights == NULL)
        {
            cerr << "Failed to allocate gaussians for pixel!" << endl << "Paused." << endl;
            return false;
        }

        // initialize the first gaussian for the pixel
        initializeGaussian(&mu[0],&sigma[0],&weights[0]);
        return true;
    }
    /**
     * @brief normalize the gaussian weights to unit length
     */

    void normalize()
    {
        double sum = 0.0;
        for(int i=0;i<numGaussians;++i)
            sum += weights[i];
        for(int i=0;i<numGaussians;++i)
            weights[i] /= sum;
    }
    /**
     * @brief setPixelValue assign the pixel intensity
     * @param px the pixel intensity value
     */
    void setPixelValue(float px)
    {
        pixelValue = px;

    }


    /**
     * @brief processValue processes an incoming pixel value.
     *
     *
     * @return true if it belongs in the background, else false.
     *
     *
     * The intensity is checked with the gaussian match criterion and the distributions update their parameters if they match the
     * input or not, and update their weights as well. If a match does not occur, a new distribution is added to the collection.
     * The wsigma computation and partitioning follows, and the function returns true if the pixel matched a background distribution.
     */
    bool processValue()
    {

        // measure whether pixel px belongs to a distribution,
        // at a window of 2.5 sigma around the mean (~0.98758 probability)
        bool matchOccured = false;
        int matchingGaussian;
        for(int g=0;g<numGaussians;++g)
        {
            if (abs(pixelValue - mu[g]) <= 2.5 * sigma[g])
            {
                VERBOSE
                cout << "Value : " << pixelValue << ", match: " << g << endl;
                // it's a match. Update the weight
                weights[g] = (1-learningRate) * weights[g] + learningRate;
                matchingGaussian = g;
                matchOccured = true;
                break;
            }
            else
                // no match
                weights[g] = (1-learningRate) * weights[g];
        }

        printGaussians("After matching: ");
        // update the gaussians accordingly:
        if(!matchOccured)
        {
            // the pixel value did not match any gaussian.
            // if the number of gaussians is < maxGaussians, just add a new one
            // else, delete the least likely one, by the wsigma metric
            int gaussianToModify;
            if(numGaussians < maxGaussians)
            {
                VERBOSE
                cout << numGaussians << " gausssians - adding one." << endl;
                // there is room. append one at the end & mark it as the matching gaussian
                ++numGaussians;
                gaussianToModify = numGaussians-1;
                matchingGaussian = gaussianToModify;
                // we are adding a new gaussian
                penalizeWeight=true;
            }
            else
            {
                // calculate the w/sigma values
                calculateWSigma();
                printGaussiansWS();
                // modify the least likely one
                gaussianToModify = wsigma.back().second;
                VERBOSE
                cout << numGaussians << "gausssians - replacing ." << gaussianToModify << endl;
            }
            // add the new gaussian
            initializeGaussian(&mu[gaussianToModify],&sigma[gaussianToModify],&weights[gaussianToModify]);
        }
        else
        {
            // a match did occur: modify the mu and sigma of the
            // matching distro to match that pixel better
            updateGaussian(matchingGaussian);
        }
        printGaussians("After update: ");
        // renormalize the weights to form a unit vector
        normalize();
        // calculate the new w/sigma values
        calculateWSigma();
        printGaussians( "After norm.: ");
        printGaussiansWS();
        // check whether the match is on the
        // background or the foreground and classify the pixel as such

        // partition existing gaussians
        partitionGaussians();
        // if it belongs in the background
        if(find(backgroundIdx.begin(),backgroundIdx.end(),matchingGaussian)!=backgroundIdx.end())
        {
            // it matched a background gaussian
            return true;
        }

        // else false
        return false;
    }
    /**
     * @brief initializeGaussian initializes a gaussian distribution.
     * @param mu    pointer to the mu value of the gaussian
     * @param sigma pointer to the stdev value of the gaussian
     * @param weight pointer to the weight value of the gaussian
     *
     * Mu is set to the current pixel value, sigma to a high, const value (default 10), and the weight
     * to a low value of 1/numGaussians^2.
     */
    void initializeGaussian(double * mu, double * sigma,double * weight)
    {
        *mu = pixelValue;
        *sigma = initialSigma;
        *weight = 1/((float)numGaussians);
        if(penalizeWeight)
        {
            *weight  = *weight /((float)numGaussians);
            penalizeWeight=false;
        }
    }
    /**
     * @brief calculateWSigma Calculate the weight/stdev metric for sorting gaussians.
     *
     * This method uses std::vector and a lamda comparator to std::sort to sort the gaussians by the
     * w/sigma metric.
     */
    void calculateWSigma()
    {
        // calculate the ratio weight/sigma for each gaussian.
        // order them by that quantity. Use std::pair to keep track of
        // the index as well

        wsigma.clear();
        for(int g=0;g<numGaussians;++g)
            wsigma.push_back(pair<double,int>(weights[g]/sigma[g],g));

        // order them by the wsigma value
        sort(wsigma.begin(),wsigma.end(),
             [&](pair<double,int> & e1,pair<double,int> & e2)
                {
                    return e1.first > e2.first;
                }

             );
    }
    /**
     * @brief partitionGaussians Specify which gaussians will form the background
     *
     * A running sum of the gaussian weights is computed. The gaussians are considered by
     * decreasing w/sigma value. When the threshold value is reached, the partitioning ends, and
     * icluded gaussians form the background.
     */
    void partitionGaussians()
    {
        // note : wsigma is assumed to be computed by now
        // partition the gaussians into background and foreground distros
        // clear the background-foreground index
        backgroundIdx.clear();

        // keep as many gaussians as are needed to reach the threshold value
        // starting with the ones with the most w/sigma
        double sum=0.0; int g=-1;
        while(sum < thresh && ++g < numGaussians)
        {
            // add the index to the background
            // gaussians
            backgroundIdx.push_back(wsigma[g].second);
            // add its weight value to the running sum
            sum+= weights[wsigma[g].second];
        }
    }

    /** @brief Destructor
      *
      */


    ~pixel()
    {
        delete [] mu;
        delete [] sigma;
        delete [] weights;
    }
    /// The current pixel value.
    float pixelValue;
    /// Max number of gaussians
    int maxGaussians,
        /// Current number of gaussians
        numGaussians;
    /// Vector where to store the w/sigma quantities, per gaussian
    vector<pair<double,int>> wsigma;
    // distro parameters
    /// Array of gaussian means
    double  * mu,
    /// Array of gaussian standard deviations
    *sigma,
    /// Array of gaussian weights
    * weights,
    /// The learning rate
    learningRate,
    /// Threshold value
    thresh;
    /// Vector of indexes of the gaussians that form the background
    vector<int> backgroundIdx;
    /// Initial large standard deviation value for new gaussians
    const float initialSigma = 10;
    /// For debug printing
    const bool verbose = false;
    /// Posterior addition to penalize the weight of new gaussians
    bool penalizeWeight;

    /**
     * @brief setMetaparameters Parameter setter
     * @param lr learning rate
     * @param thr threshold
     */
    void setMetaparameters(double lr, double thr)
    {
        thresh = thr;
        learningRate = lr;
    }
    /**
     * @brief gaussian function evaluator
     * @param idx Which gaussian we want to compute.
     * @return The computed value.
     */
    double gaussian(int idx)
    {
        return exp(-0.5 * pow((pixelValue-mu[idx])/sigma[idx],2)) / (sigma[idx] * sqrt(2*M_PI));

    }
    /**
     * @brief updateGaussian Recompute gaussian parameters
     * @param index The index of the gaussian wwe want to modify.
     *
     * This method uses the current pixel field value to alter the mean and stdev values.
     */
    void updateGaussian(int index)
    {
        double rho = learningRate * gaussian(index);
        double newmu,newsigma;
        newmu = (1-rho) * mu[index] + rho * pixelValue;
        newsigma = sqrt((1-rho) * sigma[index]*sigma[index] +
                     rho * (pixelValue-mu[index]) * (pixelValue-mu[index]));
        mu[index] = newmu;
        sigma[index] = newsigma;
    }
    /**
     * @brief printGaussians Debugging print function.
     * @param msg
     */
    void printGaussians(string msg)
    {
        if(!verbose) return;
        cout << msg;
        for(int g=0;g<numGaussians;++g)
        {
            cout << g << ": (" << mu[g] << " " << sigma[g] << " " << weights[g] << ") ";
        }
        cout << endl << flush;
    }
    /**
     * @brief printGaussiansWS Debugging print function.
     */
    void printGaussiansWS()
    {
        if(!verbose) return;

        for(int g=0;g<numGaussians;++g)
        {
            cout << g << ": (" <<wsigma[g].first << " " << wsigma[g].second << ") ";
        }
        cout << endl << flush;
    }

};

/**
 * @brief The Stauffer-Grimson class
 */
class SGb
{
public:
    /**
     * @brief SGb Constructor
     * @param cols Image columns
     * @param rows Image rows
     */
    SGb(int cols, int rows)
    {
        this->rows = rows;
        this->cols = cols;
    }
    /** @brief Destructor
      *
      * */
    ~SGb()
    {
        delete[] pixels;
    }

    static const string  name;
    /**
     * @brief usage Display method parameters
     */
    static void usage()
    {
        cout << "\tmethod " << SGb::name<< " parameters:" << endl;
        cout << "\t\t numgaussians learningrate threshold" << endl;
    }
    /**
     * @brief usage_controls Display user interface infrmation
     */
    static void usage_controls()
    {
        cout << "t/r : increase/decrease threshold" << endl;
        cout << "j/k : increase/decrease learning rate" << endl;
    }
    /**
     * @brief initialize Initialize the class,
     * @param opts Options string vector
     * @param frame Initial frame
     * @return Successful initialization
     */
    bool initialize(vector<string> opts,cv::Mat & frame)
    {
        cout << "Initializing SG background subtractor with frame size of " << rows << " x " << cols << endl;
        // defaults
        learningRate = 0.1;
        numGaussians = 2;
        threshold = 0.4;

        // initialize settings from CLI arguments
        if (opts.size() > 1) numGaussians = stoi(string(opts[0]));
        if (opts.size() > 2) learningRate = stod(string(opts[1]));
        if (opts.size() > 3) threshold = stod(string(opts[2]));

        // initialize distributions  & weights
        // check memory
        pixels = new pixel [rows * cols];
        if(pixels == nullptr)
        {
            cerr << "Unable to allocate enough memory for pixels class." << endl;
            return false ;
        }

        // init each pixel
        for(int i=0;i<rows;++i)
        {
            for(int j=0;j<cols;++j)
            {

                pixels[i*cols + j].setPixelValue((float)((int)frame.at<unsigned char>(i,j)));

                if(pixels[i*cols + j].init(numGaussians,learningRate,threshold) == false)
                {
                    cerr << "Initialization error!" << endl;
                    return false;
                }
            }
        }
        // resize output frame
        outputFrame = cv::Mat(rows,cols,CV_32F);
        usage_controls();
        return true;
    }

    // process each incoming frame
    cv::Mat process(cv::Mat inputFrame)
    {

        // for each pixel of the input frame
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;++j)
            {
                // set the pixel intensity value
                pixels[i*cols + j].setPixelValue((float)((int)inputFrame.at<unsigned char>(i,j)));
                // set the parameters which the user may have changed
                pixels[i*cols + j].setMetaparameters(learningRate,threshold);
                // process the value and check the result
                if(pixels[i*cols+j].processValue())
                {
                    outputFrame.at<float>(i,j) = 0; // background
                }
                else
                {
                    outputFrame.at<float>(i,j) = 255; // foreground
                }
            }
        }
        // return the resulting frame
        return outputFrame;
    }



    // interactivity functions
    /**
     * @brief info Display current parmeters.
     */
    void info()
    {
        cout << "Thresh: " << threshold << " LR : " << learningRate  << endl;
    }
    /**
     * @brief decreaseLR Decrease learning rate
     */
    void decreaseLR(){learningRate -= 0.1; if(learningRate<0) learningRate=0; info(); }
    /**
     * @brief increaseLR Increase learning rate
     */
    void increaseLR(){learningRate += 0.1; if(learningRate>1) learningRate=1; info(); }
    /**
     * @brief increaseThreshold Increase the threshold value
     */
    void increaseThreshold(){ threshold+=0.1; if(threshold>1) threshold=1; info();}

    /**
     * @brief decreaseThreshold Decrease the threshold value
     */
    void decreaseThreshold(){ threshold-=0.1; if(threshold<0) threshold=0; info();}

    // members
    /// Array of all pixels in the image.
    pixel * pixels;
    /// Output frame mat container.
    cv::Mat outputFrame;
    /// Image rows
    int rows,
        /// Image columns
        cols,
        ///Gaussians per pixel.
        numGaussians;
    /// The learning rate per pixel
    double learningRate,
        ///The threshold per pixel
        threshold;
};
/// Method name
const string SGb::name="SG";
#endif // SGB

