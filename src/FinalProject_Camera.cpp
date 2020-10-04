
/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
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
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // image_02 = left camera in color,
    // since the first processing endured by these images is deep learning framework detection (YOLO),
    // so it is much more convenient to have these images in colour rather than grayscale (required 
    // for keypoints detection)
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 18;   // last file index to load
    int imgStepWidth = 1; // If 1, we are using every image (from StartIndex to EndIndex) 
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // object detection
    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // calibration data for camera and lidar
    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector
    
    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;
    
    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;
    
    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;    

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    double det_des_time = 0;

    string detector_algorithm = "FAST"; // SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
    string descriptor_algorithm = "FREAK"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth)
    {
        cout << "\n-----------------------" <<endl<<endl;

        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file 
        cv::Mat img = cv::imread(imgFullFilename);

        //cout << "Cols, rows: " << img.cols << " " << img.rows << endl;

        if (dataBuffer.size() == dataBufferSize) // Buffer is full. First element of the vector 
        // is removed and second element is now the first one
        {
            vector<DataFrame> aux_dataBuffer;
            aux_dataBuffer.push_back(dataBuffer[1]); 
            dataBuffer.clear();
            dataBuffer = aux_dataBuffer;
        }

        DataFrame frame;
        frame.cameraImg = img;
        dataBuffer.push_back(frame); // Note that here we are storing colour images, required
        // by deep learning based detection framework, albeit we will perform keypoints
        // extraction using grayscale images

        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT & CLASSIFY OBJECTS */

        float confThreshold = 0.15;
        float nmsThreshold = 0.4;
        bool bVis = false; 
        // Note that we only use the last image stored in the buffer to detect bounding boxes with YOLO
        detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
                      yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

        //cout<<"Number of BBs: "<<(dataBuffer.end() - 1)->boundingBoxes.size()<<endl;
        cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;

        /* CROP LIDAR POINTS */

        // load 3D Lidar points from file
        string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // remove Lidar points based on distance properties

        // minZ = -1.5, maxZ = -0.9 -> We focus on the tail gates of the surrounding vehicles
        // minX = 2.0, maxX = 20.0, maxY = 2.0 (both sides) -> We focus in front of us
        // minR = 0.1 -> Reflectivity ranges from 0.0 (no reflect.) to 1.0 (max. reflect.). We only
        // considers points that have a reflect. greater than minR

        // Note that the goal of this cropping is to compute the TTC (Time-to-Collision) and not to
        // display all the objects

        float minZ = -1.5, maxZ = -0.5, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
    
        (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

        cout << "#3 : CROP LIDAR POINTS done" << endl;

        /* CLUSTER LIDAR POINT CLOUD */

        // associate Lidar points with camera-based ROI
        float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
        clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

        // Visualize 3D objects
        bVis = false;
        if(bVis)
        {
            show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true);
        }
        bVis = false;

        cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;
        
        /*for (auto it = dataBuffer.begin(); it != dataBuffer.end(); it++)
        {
            cout <<"LiDAR points: "<<it->lidarPoints.size()<<endl<<endl;

            for (auto it1 = it->boundingBoxes.begin(); it1 != it->boundingBoxes.end(); it1++)
            {
                cout<<"BB LiDAR Points: "<<it1->lidarPoints.size()<<endl;
            }
        }*/

        //continue; // skips directly to the next image without processing what comes beneath

        /* DETECT IMAGE KEYPOINTS */

        // convert current image to grayscale
        cv::Mat imgGray;
        cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        double t_det = (double)cv::getTickCount();

        bVis = false;
        if (detector_algorithm.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, bVis);
        }
        else if (detector_algorithm.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, imgGray, bVis);
        }
        else // We assume that the other types are modern detectors
             // If the user does not introduce a right detector type name (FAST, BRISK, ORB,
             // AKAZE or SIFT), no keypoints will be detected
        {
            detKeypointsModern(keypoints, imgGray, detector_algorithm, bVis);
        }

        t_det = ((double)cv::getTickCount() - t_det) / cv::getTickFrequency(); // End detection time
        det_des_time += t_det;

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detector_algorithm.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << "NOTE: Keypoints have been limited!" << endl;
        }

        // Only keep keypoints on the preceding vehicle

        bool bFocusOnVehicle = true;

        if (bFocusOnVehicle)
        {
            vector<cv::KeyPoint> aux_keypoints = keypoints;
            keypoints.clear();

            for (auto it = aux_keypoints.begin(); it != aux_keypoints.end(); it++)
            {
                float x = (*it).pt.x;
                float y = (*it).pt.y;

                int tolerance = 150;

                float x_min = ((dataBuffer.end()-1)->cameraImg.cols)/2.0 - tolerance;
                float x_max = ((dataBuffer.end()-1)->cameraImg.cols)/2.0 + tolerance;

                if (x >= x_min && x <= x_max)
                {
                    keypoints.push_back(*it);
                }
            }

            cout << "Number of keypoints after cropping: " << keypoints.size() << endl;
        }

        // Push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;

        cout << "#5 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        cv::Mat descriptors;

        double t_des = (double)cv::getTickCount();

        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptor_algorithm);

        t_des = ((double)cv::getTickCount() - t_des) / cv::getTickFrequency(); // End detection time
        det_des_time += t_des;

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#6 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {
            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            string descriptor_type = "DES_BINARY"; // DES_BINARY, DES_HOG
            string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

            if (descriptor_algorithm.compare("SIFT") == 0)
            {
                descriptor_type = "DES_HOG";
            }

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptor_type, matcherType, selectorType);

            bVis = false;
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

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            //cout << "Number of matches: " << matches.size() << endl;
            
            cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;
            
            /* TRACK 3D OBJECT BOUNDING BOXES */

            //// STUDENT ASSIGNMENT
            //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
            
            map<int, int> bbBestMatches;
            matchBoundingBoxes(bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1)); // associate bounding boxes between current and previous frame using keypoint matches
            
            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end()-1)->bbMatches = bbBestMatches;

            cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;

            /*cout << "\nBest matches map: " << endl;
            cout << "Size: " << bbBestMatches.size() << endl;
            for (auto it = bbBestMatches.begin(); it != bbBestMatches.end(); it++)
            {
                cout << it->first << " " << it->second << endl;
            }*/

            cout<<endl;

            /* COMPUTE TTC ON OBJECT IN FRONT */

            // loop over all BB match pairs

            for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
            {
                //cout << it1->first << " " << it1->second << endl;

                // find bounding boxes associates with current match
                BoundingBox *prevBB, *currBB;

                //cout << "Number of bbs current image: "<<(dataBuffer.end() - 1)->boundingBoxes.size()<<endl;
                for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2)
                {
                    if (it1->second == it2->boxID) // check wether current match partner corresponds to this BB
                    {
                        currBB = &(*it2);
                    }
                }

                //cout << "Number of bbs previous image: "<<(dataBuffer.end() - 2)->boundingBoxes.size()<<endl;
                for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
                {
                    if (it1->first == it2->boxID) // check wether current match partner corresponds to this BB
                    {
                        prevBB = &(*it2);
                    }
                }

                //cout << "Size prev: " << prevBB->lidarPoints.size() << endl;
                //cout << "Size curr: " << currBB->lidarPoints.size() << endl;

                //continue;

                // compute TTC for current match
                if((currBB->lidarPoints.size() > 0) && (prevBB->lidarPoints.size() > 0)) // only compute TTC if we have Lidar points
                {
                    //// STUDENT ASSIGNMENT
                    //// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)

                    double ttcLidar; 

                    computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);
                    //// EOF STUDENT ASSIGNMENT
                    //continue;
                    //// STUDENT ASSIGNMENT
                    //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                    //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
                    
                    double ttcCamera;

                    clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches);                    
                    computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera);
                    //// EOF STUDENT ASSIGNMENT

                    cout << "\n###############" << endl<<endl;

                    cout << "LiDAR based TTC: " << ttcLidar << " s" << endl;
                    cout << "Camera based TTC: " << ttcCamera << " s" << endl;
                    cout << "Time took by keypoint detection and description: " << det_des_time << " s" <<endl;

                    bVis = false;
                    if (bVis)
                    {
                        cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                        showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
                        cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);
                        
                        char str[200];
                        sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                        putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));

                        string windowName = "Final Results : TTC";
                        cv::namedWindow(windowName, 4);
                        cv::imshow(windowName, visImg);
                        cout << "Press key to continue to next frame" << endl;
                        cv::waitKey(0);
                    }
                    bVis = false;

                } // eof TTC computation
            } // eof loop over all BB matches            

        }

    } // eof loop over all images

    return 0;
}
