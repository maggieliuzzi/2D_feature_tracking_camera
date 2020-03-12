/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <deque>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;


int main(int argc, const char *argv[])
{
    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // colour camera - left
    string imgFileType = ".png";
    int imgStartIndex = 0;  // first file index to load (assumes LiDAR and camera names have identical naming conventions)
    int imgEndIndex = 9;    // last file index to load
    int imgFillWidth = 4;   // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images held on the circular buffer simultaneously
    deque<DataFrame> dataBuffer;  // circular buffer: double-ended queue (std::deque) (constant time O(1) insertion, deletion from both ends)
    bool bVis = false;            // visualisation

	// looping over all images
    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        // LOADING GRAYSCALE IMAGE INTO BUFFER

        // assembling filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // loading image from file and converting it to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
      
        DataFrame frame;
        frame.cameraImg = imgGray;  
        dataBuffer.push_back(frame);  // loading frame containing gray image into queue
        if (dataBuffer.size() > dataBufferSize) 
        	dataBuffer.pop_front();
        assert(dataBuffer.size() <= dataBufferSize);


        // DETECTING IMAGE KEYPOINTS

        // extracting 2D keypoints from current image
        vector<cv::KeyPoint> keypoints;
      
      	// selecting detection algorithm to use
        string detectorType = "SIFT";  // "SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"  //fast

        double det_t = (double)cv::getTickCount();  // timing detection process

        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, imgGray, false);
        }
        else if (detectorType.compare("FAST")  == 0 ||
                 detectorType.compare("BRISK") == 0 ||
                 detectorType.compare("ORB")   == 0 ||
                 detectorType.compare("AKAZE") == 0 ||
                 detectorType.compare("SIFT")  == 0)  // modern detection algorithms
        {
            detKeypointsModern(keypoints, imgGray, detectorType, false);
        }
        else
        {
            throw invalid_argument(detectorType + " is not a valid detectorType");
        }

        det_t = ((double)cv::getTickCount() - det_t) / cv::getTickFrequency();


      	// Filtering out keypoints not on preceding vehicle (only for algorithm evaluation purposes)

        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);  // hardcoded estimated bounding box
        if (bFocusOnVehicle)
        {
            vector<cv::KeyPoint> filteredKeypoints;
            for (auto kp : keypoints) 
            {
                if (vehicleRect.contains(kp.pt)) 
                	filteredKeypoints.push_back(kp);
            }
            keypoints = filteredKeypoints;
            cout << "Note: only considering keypoints on preceding vehicle" << endl;
        }
        cout << detectorType << ": " << keypoints.size() << ", ";


        // limiting number of keypoints (for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            {   // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << "Note: keypoints have been limited" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;  // pushing current frame's keypoints into buffer (through its end)


        // EXTRACTING KEYPOINT DESCRIPTORS

        cv::Mat descriptors;
        // choosing descriptor extracting algorithm to use
        string descriptorType = "ORB";  // "BRISK", "BRIEF", "ORB", "FREAK", "AKAZE" (requires AKAZE detectors), "SIFT" (not compatible with ORB detectors)

        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);

        (dataBuffer.end() - 1)->descriptors = descriptors;  // adding current frame's descriptors to end of buffer


		// MATCHING KEYPOINT DESCRIPTORS BETWEEN TWO IMAGES
        if (dataBuffer.size() > 1)  // wait until at least two images have been processed
        {
            vector<cv::DMatch> matches;
          	// choosing keypoint descriptor matching algorithm to use
            string matcherType = "MAT_BF";  // "MAT_BF" (Brute Force) or "MAT_FLANN" (Approximate Nearest Neighbours)
            
          	// choosing appropriate descriptor type: "DES_BINARY" (Binary) or "DES_HOG" (Histogram of Gradients)
            /* BINARY descriptors include: BRISK, BRIEF, ORB, FREAK, and (A)KAZE. */
            /* HOG descriptors include: SIFT (plus SURF and GLOH, all patented). */
            string descriptorCategory {};
            if (descriptorType.compare("SIFT") == 0) 
            {
                descriptorCategory = "DES_HOG";
            }
            else 
            {
                descriptorCategory = "DES_BINARY";
            }

            // choosing selector type to use
            string selectorType = "SEL_KNN";  // "SEL_NN" (Nearest Neighbour) or "SEL_KNN" (K Nearest Neighbours)


            double desc_t = (double)cv::getTickCount();  // timing descriptor matching

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorCategory, matcherType, selectorType);
            
            desc_t = ((double)cv::getTickCount() - desc_t) / cv::getTickFrequency();
            cout << detectorType << " | " << descriptorType << ": " << matches.size() << ", ";


            (dataBuffer.end() - 1)->kptMatches = matches;  // storing current frame's keypoint matches on buffer

            // visualising keypoint matches between current and previous image
            bVis = true;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matched keypoints between previous and current frames";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image..." << endl;
                cv::waitKey(0);  // wait for key to be pressed on pop-up
            }
            bVis = false;
        }

    }  // eof loop over all images

    return 0;
}
