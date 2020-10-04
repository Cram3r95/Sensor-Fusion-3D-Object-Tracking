#include <numeric>
#include "matching2D.hpp"

using namespace std;

// ---- Detection

// Detect keypoints in image using the traditional Shi-Thomasi detector

void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency(); // End time
    cout << "Shi-Tomasi detection with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // Visualize results

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using the traditional Harris cournerness detector

void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize = 2;       // for every pixel, a blockSize x blockSize neighbourhood is considered
    int apertureSize = 3;    // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100;   // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;         // Harris parameter 

    // Detect corners using Harris detector and normalize output

    double t = (double)cv::getTickCount(); // Start time

    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);

    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Get preliminary keypoints based on response

    double max_overlap = 0.0; // maximum permissible overlap between two features in %

    for (size_t i = 0; i < dst_norm.rows; i++)
    {
        for (size_t j = 0; j < dst_norm.cols; j++)
        {
            int response = (int)dst_norm.at<float>(i,j); // dst_norm contains float elements

            if (response > minResponse)
            // This point can be considered as a preliminary keypoint due to its response
            {
                cv::KeyPoint new_keypoint;
                new_keypoint.pt = cv::Point2f(j,i); // First argument = x-axis = columns = j index
                                                    // Second argument = y-axis = rows = i index
                new_keypoint.size = 2*apertureSize;
                new_keypoint.response = response;

                // Perform Non-Maximum Supression (NMS) and get definitive keypoints

                bool overlap = false;

                for (auto it = keypoints.begin(); it != keypoints.end(); it++)
                {
                    // Compute overlap between current keypoint candidate and definitive keypoints

                    double keypoints_overlap = cv::KeyPoint::overlap(new_keypoint, *it);

                    if (keypoints_overlap > max_overlap) // If both keypoints overlap, either current
                    // keypoint candidate or "definitive" keypoint must be preserved based on its response
                    {
                        overlap = true;

                        if (new_keypoint.response > it->response)
                        {
                            *it = new_keypoint;
                            break;
                        }
                    }
                }

                if (!overlap)
                {
                    keypoints.push_back(new_keypoint);
                }
            }
        }
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency(); // End time
    cout << "Harris detection with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // Visualize results

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using modern keypoints detectors

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    cv::Ptr<cv::FastFeatureDetector> detector_fast;
    cv::Ptr<cv::FeatureDetector> detector;

    bool fast_flag = false;

    // FAST, BRISK, ORB, AKAZE, SIFT

    double t = (double)cv::getTickCount(); // Start time

    if (detectorType.compare("FAST") == 0)
    {
        detector_fast = cv::FastFeatureDetector::create();
        detector_fast->detect(img, keypoints);
        fast_flag = true;
    }
    else if(detectorType.compare("BRISK") == 0)
    {
        detector = cv::BRISK::create();
    }
    else if(detectorType.compare("ORB") == 0)
    {
        detector = cv::ORB::create();
    }
    else if(detectorType.compare("AKAZE") == 0)
    {
        detector = cv::AKAZE::create();
    }
    else if(detectorType.compare("SIFT") == 0)
    {
        detector = cv::xfeatures2d::SiftFeatureDetector::create();  
    }

    t = (double)cv::getTickCount(); // Start time

    if (!fast_flag)
    {
        detector->detect(img, keypoints);
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency(); // End time
    cout << detectorType << " detection with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // Visualize results

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Modern Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// ---- Description

// Use one of several types of state-of-art descriptors to uniquely identify keypoints

void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    cv::Ptr<cv::DescriptorExtractor> extractor;
    cv::Ptr<cv::Feature2D> extractor_akaze;

    bool akaze_descriptor = false;

    // select appropriate descriptor (BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT)

    // Note: For each branch of if-else structure, suitable parameters are written, which are indeed
    // by-default OpenCV image parameters. They are written just to observe them in the code, but
    // by default the class constructor will asign these values

    double t = (double)cv::getTickCount(); // Start time

    if (descriptorType.compare("BRISK") == 0)
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if(descriptorType.compare("BRIEF") == 0)
    {
        int bytes = 32;
        bool use_orientation = false;

        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, use_orientation);
    }
    else if(descriptorType.compare("ORB") == 0)
    {
        int features = 500;
        float scale_factor = 1.2f;
        int n_levels = 8;
        int threshold = 31;
        int first_level = 0;
        int WTA_K = 2;
        cv::ORB::ScoreType score_type = cv::ORB::HARRIS_SCORE; // in docOpeCV is int
        int patch_size = 31;
        int fast_threshold = 20;

        extractor = cv::ORB::create(features, scale_factor, n_levels, threshold, first_level, WTA_K, score_type, patch_size, fast_threshold);
    }
    else if(descriptorType.compare("FREAK") == 0)
    {
        bool orientation_normalized = true;
        bool scale_normalized = true;
        float pattern_scale = 22.0f;
        int n_octaves = 4;
        const std::vector<int> & selected_pairs = std::vector<int>();

        extractor = cv::xfeatures2d::FREAK::create(orientation_normalized, scale_normalized, pattern_scale, n_octaves, selected_pairs);
    }
    else if(descriptorType.compare("AKAZE") == 0)
    {
        cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB; // in docOpeCV is int
        int descriptor_size = 0;
        int descriptor_channels = 3;
        float threshold = 0.001f;
        int n_octaves = 4;
        int n_octave_layers = 4;
        cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2; // in docOpeCV is int

        extractor_akaze = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold, n_octaves, n_octave_layers);

        // Note that second argument is the mask (empty array in our case). With this method we get both keypoints and descriptors
        
        akaze_descriptor = true;
    }
    else if(descriptorType.compare("SIFT") == 0)
    {
        int n_features = 0;
        int n_octave_layers = 3;
        double contrast_threshold = 0.04;
        double edge_threshold = 10;
        double sigma = 1.6;

        extractor = cv::xfeatures2d::SiftDescriptorExtractor::create(n_features, n_octave_layers, contrast_threshold, edge_threshold, sigma);
    }

    // Perform feature description

    t = (double)cv::getTickCount(); // End time

    if (!akaze_descriptor)
    {
        extractor->compute(img, keypoints, descriptors);
    }
    else
    {
        //extractor_akaze->compute(img, keypoints, descriptors);

        std::vector<cv::KeyPoint> akaze_keypoints;
        extractor_akaze->detectAndCompute(img, cv::Mat(), akaze_keypoints, descriptors);
    }
    

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// ---- Matching

// Find best matches for keypoints in two camera images based on several matching methods

void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptor_type, std::string matcherType, std::string selectorType)
{
    // Configure matcher

    bool cross_check = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptor_type.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, cross_check);
        cout << "BF matching cross-check: " << cross_check << endl;
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F) // Source image descriptors type
        { // OpenCV bug workaround: Convert binary  descriptors (so, != CV_32F) to floating point due
        // to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
            // It is assumed that if the source image was analyzed using binary descriptors,
            // reference image was analyzed in the same way
        }

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

        cout << "FLANN matching" <<endl;
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
    // If Nearest Neighbour is chosen, it is pretty simple. However, two images might be totally
    // different, so even if all descriptors are completely different, we would choose the closest
    // one. Then, it does not make sense unless we are pretty sure that the images we are going
    // to compare belong to the same scene. Instead, using the combination kNN and descriptor
    // distance ratio test, despite its simplicity, help us to resolve ambiguities and mistmatches
    // in most cases

        vector<vector<cv::DMatch>> knn_matches;

        double t = (double)cv::getTickCount(); // Start time

        // Find the k best matches

        int k = 2;

        matcher->knnMatch(descSource, descRef, knn_matches, k);

        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency(); // End time

        cout << "KNN with k = " << k << " and n = " << knn_matches.size() << " matches in "<<1000*t/1.0<<" ms"<<endl;

        // Filter matches using descriptor distance ratio test

        double distance_threshold = 0.8; // If distance ratio (best_match / second_best_match) is
        // lower or equal than distance_threshold, push_back best match

        for (auto it = knn_matches.begin(); it != knn_matches.end(); it++)
        {
            if ((*it)[0].distance <= distance_threshold*(*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }

        cout << "# Keypoints removed: " << knn_matches.size() - matches.size() << endl;
    }

    cout << "Number of definitive matches: " << matches.size() << endl;
}
