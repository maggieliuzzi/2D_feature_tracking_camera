# 2D feature tracking using camera

## Data buffer

Using a double-ended queue with a 2-element limit (Insertion time: O(1)).

## Keypoint detection

Implementing detectors HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT and making them selectable by setting a string accordingly.

## Keypoint removal/ filtering

Removing all keypoints outside of a pre-defined rectangle (estimated bounding box for preceding vehicle) and only using the keypoints within the rectangle for further processing.

## Keypoint descriptors

Implementing descriptors BRIEF, ORB, FREAK, AKAZE and SIFT and making them selectable by setting a string accordingly.

## Descriptor matching

Implementing FLANN matching as well as k-nearest neighbor selection. Both methods are selectable using the respective strings in the main function.

## Descriptor distance ratio

Using the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs second-best match to decide whether to keep an associated pair of keypoints or not.


## Performance Evaluation

For each implemented detector, counting the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size:

	SHI-TOMASI
    125, 118, 123, 120, 120, 113, 114, 123, 111, 112
    HARRIS
    17, 14, 19, 22, 26, 47, 18, 33, 27, 35
    FAST
    419, 427, 404, 423, 386, 414, 418, 406, 396, 401
    BRISK
    264, 282, 282, 277, 297, 279, 289, 272, 266, 254
    ORB
    92, 102, 106, 113, 109, 125, 130, 129, 127, 128
    AKAZE
    166, 157, 161, 155, 163, 164, 173, 175, 177, 179
    SIFT
    138, 132, 124, 138, 134, 140, 137, 148, 159, 137

Counting the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8:

	Found on results > det_desc_pair_results.txt

Logging the time it takes for keypoint detection and descriptor extraction:

	Found on results > det_desc_pair_results.txt

Based on this data, the top 3 recommended detector / descriptor combinations for detecting keypoints on vehicles are:

	FAST + ORB
    FAST + BRIEF
    FAST + BRISK or SIFT
