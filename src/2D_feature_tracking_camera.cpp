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

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    deque<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize
        
        // The dataBuffer on line 40 above has been changed from a std::vector to std::deque
        // A double-ended queue (std::deque) has constant time O(1) insertion
        // and deletion on both ends of the collection. -Dylan

        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push_back(frame);
        if (dataBuffer.size() > dataBufferSize) dataBuffer.pop_front();
        assert(dataBuffer.size() <= dataBufferSize);

        //// EOF STUDENT ASSIGNMENT
        // cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints;

        // string detectorType = "SHITOMASI";
        // string detectorType = "HARRIS";
        string detectorType = "FAST";
        // string detectorType = "BRISK";
        // string detectorType = "ORB";
        // string detectorType = "AKAZE";
        // string detectorType = "SIFT";

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
        
        // Start MP.9 detector timer
        double det_t = (double)cv::getTickCount();

        // Shi-Tomasi
        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, false);
        }

        // Harris
        else if (detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, imgGray, false);
        }

        // Modern detector types, including FAST, BRISK, ORB, AKAZE, and SIFT
        else if (detectorType.compare("FAST")  == 0 ||
                 detectorType.compare("BRISK") == 0 ||
                 detectorType.compare("ORB")   == 0 ||
                 detectorType.compare("AKAZE") == 0 ||
                 detectorType.compare("SIFT")  == 0)
        {
            detKeypointsModern(keypoints, imgGray, detectorType, false);
        }

        // Specified detectorType is unsupported
        else
        {
            throw invalid_argument(detectorType + " is not a valid detectorType");
        }

        // Task MP.9 detector timing
        det_t = ((double)cv::getTickCount() - det_t) / cv::getTickFrequency();
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // NOTICE: Please use this cropping method for dev evaluation only.
        // All keypoints should be considered in the upcoming final project.

        // Filter with a hardcoded bounding box to keep only keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
            vector<cv::KeyPoint> filteredKeypoints;
            for (auto kp : keypoints) {
                if (vehicleRect.contains(kp.pt)) filteredKeypoints.push_back(kp);
            }
            keypoints = filteredKeypoints;
        }

        // Task MP.7 Counting keypoints
        // cout << detectorType << ",";
        // cout << keypoints.size() << ",";
        // if (keypoints.size() > 0) {
        //     auto kp = keypoints[0];
        //     cout << kp.size << endl;
        // }
        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        // cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        // string descriptorType = "BRISK";
        // string descriptorType = "BRIEF";
        string descriptorType = "ORB";
        // string descriptorType = "FREAK";
        // string descriptorType = "AKAZE";  // Fails with all non-AKAZE detectors
        // string descriptorType = "SIFT";  // Fails with ORB detectors

        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        // cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */
            vector<cv::DMatch> matches;

            /* Select brute force (BF) or Fast Library for Approximate Nearest Neighbors (FLANN) */
            string matcherType = "MAT_BF";
            // string matcherType = "MAT_FLANN";
            
            /* For descriptor type, select binary (BINARY) or histogram of gradients (HOG) */
            /* BINARY descriptors include: BRISK, BRIEF, ORB, FREAK, and (A)KAZE. */
            /* HOG descriptors include: SIFT (and SURF and GLOH, all patented). */
            string descriptorCategory {};
            if (0 == descriptorType.compare("SIFT")) {
                descriptorCategory = "DES_HOG";
            }
            else {
                descriptorCategory = "DES_BINARY";
            }

            /* For selector type, choose nearest neighbors (NN) or k nearest neighbors (KNN) */
            // string selectorType = "SEL_NN";
            string selectorType = "SEL_KNN";

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            double desc_t = (double)cv::getTickCount();

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorCategory, matcherType, selectorType);
            
            desc_t = ((double)cv::getTickCount() - desc_t) / cv::getTickFrequency();

            cout << detectorType << ",";
            cout << descriptorType << ",";
            cout << matches.size() << ",";
            cout << 1000 * det_t / 1.0 << ",";
            cout << 1000 * desc_t / 1.0 << ",";
            cout << (1000 * det_t / 1.0) + (1000 * desc_t / 1.0) << endl;
            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            // cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            bVis = true;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

    } // eof loop over all images

    return 0;
}
