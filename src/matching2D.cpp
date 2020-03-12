#include <numeric>
#include "matching2D.hpp"

using namespace std;


void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    /* Detects keypoints in an image using the traditional SHI-TOMASI algorithm */
  
    // computing detector parameters based on image size
    int blockSize = 4;        //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0;  // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance);  // max. num. of keypoints
    double qualityLevel = 0.01;  // minimal accepted quality of image corners
    double k = 0.04;

    // applying corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // adding corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    // cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis)  // visualisation setting on
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in an image by adapting the Harris detector developed in a previous exercise
void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    /* Detecting keypoints in an image using the HARRIS algorithm */
  
    int blockSize = 2;     // A blockSize neighborhood is considered
    int apertureSize = 3;  // Aperture parameter for the Sobel operator
    assert(1 == apertureSize % 2);  // Aperture size must be odd
    int minResponse = 100; // Minimum value for a corner in the scaled (0...255) response matrix
    double k = 0.04;       // Harris parameter (see equation for details)
    double maxOverlap = 0.0;  // non-maximum suppression (NMS): maximum overlap between two features in %

    // detecting Harris corners and normalising output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    if (bVis)  // visualisation setting on
    {
        string windowName = "Harris Corner Detector Response Matrix";
        cv::namedWindow(windowName);
        cv::imshow(windowName, dst_norm_scaled);
        cv::waitKey(0);
    }

    // applying non-maximum suppression (NMS)
    for (size_t j = 0; j < dst_norm.rows; j++) {
        for (size_t i = 0; i < dst_norm.cols; i++) {
            int response = (int)dst_norm.at<float>(j, i);

            if (response < minResponse) continue;  // applying the minimum threshold for Harris cornerness response     
          
            cv::KeyPoint newKeyPoint;  // else, creating a tentative new keypoint
            newKeyPoint.pt = cv::Point2f(i, j);
            newKeyPoint.size = 2 * apertureSize;
            newKeyPoint.response = response;

            // performing non-maximum suppression (NMS) in local neighbourhood around the new keypoint
            bool bOverlap = false;
            for (auto it = keypoints.begin(); it != keypoints.end(); ++it)  // looping through all keypoints
            {
                double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                // testing if overlap exceeds the maximum percentage allowable
                if (kptOverlap > maxOverlap) {
                    bOverlap = true;
                    // if overlapping, testing if new response is the local maximum
                    if (newKeyPoint.response > (*it).response) {
                        *it = newKeyPoint;  // replacing the old keypoint
                        break;
                    }
                }
            }

            // else, if response threshold and not overlapping any other keypoint
            if (!bOverlap) {
                keypoints.push_back(newKeyPoint);  // add to keypoints list
            }
        }
    }

    if (bVis)  // visualisation setting on
    {
        string windowName = "Harris corner detection results";
        cv::namedWindow(windowName);
        cv::Mat visImage = dst_norm_scaled.clone();
        cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    /* Detects keypoints in an image using various modern keypoint detection algorithms */
    if (detectorType.compare("FAST") == 0) 
    {
        // int threshold_FAST = 150;
        auto fast = cv::FastFeatureDetector::create();
        fast->detect(img, keypoints);
    }
    else if (detectorType.compare("BRISK") == 0) 
    {
        // int threshold_BRISK = 200;
        auto brisk = cv::BRISK::create();
        brisk->detect(img, keypoints);
    }
    else if (detectorType.compare("ORB") == 0) 
    {
        auto orb = cv::ORB::create();
        orb->detect(img, keypoints);
    }
    else if (detectorType.compare("AKAZE") == 0) 
    {
        auto akaze = cv::AKAZE::create();
        akaze->detect(img, keypoints);
    }
    else if (detectorType.compare("SIFT") == 0) 
    {
        auto sift = cv::xfeatures2d::SIFT::create();
        sift->detect(img, keypoints);
    }
    else
    	throw invalid_argument(detectorType + " is not a valid detectorType");

    if (bVis)  // visualisation setting on
    {
        string windowName = detectorType + " keypoint detection results";
        cv::namedWindow(windowName);
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}


// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    /* Identifies/ matches keypoints using various keypoint descriptor extractors */
  
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {
        int threshold = 30;         // FAST/AGAST detection threshold score.
        int octaves = 3;            // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f;  // apply this scale to the pattern used for sampling the neighbourhood of a keypoint

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }
    else
    	throw invalid_argument(descriptorType + " is not a valid descriptorType");

    extractor->compute(img, keypoints, descriptors);
}


void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef, std::vector<cv::DMatch> &matches, std::string descriptorCategory, std::string matcherType, std::string selectorType)
{
    /* Finds best matches for keypoints in two camera images based on several matching methods */
    
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)  // Brute Force Matching
    {
        int normType;
        if (descriptorCategory.compare("DES_HOG") == 0)  // Histogram of Gradients Descriptor  // for SIFT
        	normType = cv::NORM_L2;
        else if (descriptorCategory.compare("DES_BINARY") == 0)  // Binary Descriptor
        	normType = cv::NORM_HAMMING;
        else 
        	throw invalid_argument(descriptorCategory + " is not a valid descriptorCategory");

        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)  // FLANN Matching
    {
        if (descriptorCategory.compare("DES_HOG") == 0)  // for SIFT
        	matcher = cv::FlannBasedMatcher::create();
        else if (descriptorCategory.compare("DES_BINARY") == 0)  // for binary descriptors
        {
        	const cv::Ptr<cv::flann::IndexParams>& indexParams = cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2);
            matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams);
        }
        else
        	throw invalid_argument(descriptorCategory + " is not a valid descriptorCategory");
    }
    else
    	throw invalid_argument(matcherType + " is not a valid matcherType");

  
    if (selectorType.compare("SEL_NN") == 0)  // Nearest Neighbour (best match)
    	matcher->match(descSource, descRef, matches);
    else if (selectorType.compare("SEL_KNN") == 0)  // K-Nearest Neighbours (k=2)
    {
        int k = 2;
        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, k);
        
        double minDescDistRatio = 0.8;  // filtering matches with descriptor distance ratio test (0.8 threshold)
        for (auto it : knn_matches) {
            if ( 2 == it.size() && (it[0].distance < minDescDistRatio * it[1].distance) ) {  // the returned knn_matches vector contains some nested vectors with size < 2
                matches.push_back(it[0]);
            }
        }
    }
    else
    	throw invalid_argument(selectorType + " is not a valid selectorType");
}
