# 2D feature tracking using camera

## Data buffer

Vector for dataBuffer objects whose size does not exceed a limit (e.g. 2 elements).

## Keypoint detection

Implementing detectors HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT and making them selectable by setting a string accordingly.

## Keypoint removal/ filtering

Removing all keypoints outside of a pre-defined rectangle (car right in front) and only using the keypoints within the rectangle for further processing.

## Keypoint descriptors

Implementing descriptors BRIEF, ORB, FREAK, AKAZE and SIFT and making them selectable by setting a string accordingly.

## Descriptor matching

Implementing FLANN matching as well as k-nearest neighbor selection. Both methods are selectable using the respective strings in the main function.

## Descriptor distance ratio

Using the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs second-best match to decide whether to keep an associated pair of keypoints or not.

## Performance Evaluation

For each implemented detector, counting the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size:

Counting the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8:

Logging the time it takes for keypoint detection and descriptor extraction:

Based on this data, the top 3 recommended detector / descriptor combinations for detecting keypoints on vehicles are:

