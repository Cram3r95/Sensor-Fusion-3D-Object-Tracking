
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // Note: In terms of matching descriptors, "query" is used for previous frame and "train" for current frame

    for (auto it = kptMatches.begin(); it != kptMatches.end(); it++)
    {
        int curr_index = (*it).trainIdx;

        cv::Point curr_pt;
        curr_pt.x = kptsCurr[curr_index].pt.x;
        curr_pt.y = kptsCurr[curr_index].pt.y;

        if (boundingBox.roi.contains(curr_pt))
        {
            boundingBox.kptMatches.push_back(*it);
            boundingBox.keypoints.push_back(kptsCurr[curr_index]);
        }
    }
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // Compute distance ratios between all matched keypoints
    vector<double> distRatios; // Stores the distance ratios for all keypoints between curr. and prev. frame
    
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    {
        // Get current keypoint and its matched partner in the previous frame

        cv::KeyPoint outer_current_keypoint = kptsCurr.at(it1->trainIdx); // Current
        cv::KeyPoint outer_previous_keypoint = kptsPrev.at(it1->queryIdx); // Previous
           
        // Go from it1 memory address + 1 to the end
        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { 
            double min_dist = 100.0; // min. required distance

            // Get next keypoint and its matched partner in the previous frame

            cv::KeyPoint inner_current_keypoint = kptsCurr.at(it2->trainIdx); // Current
            cv::KeyPoint inner_previous_keypoint = kptsPrev.at(it2->queryIdx); // Previous
            
            // Compute distances and distance ratios

            // Euclidean distance between a point A and a point B for the current frame
            double distance_curr = cv::norm(outer_current_keypoint.pt - inner_current_keypoint.pt);
            // Euclidean distance between the point matched with A and the point matched with B for the previous frame
            double distance_prev = cv::norm(outer_previous_keypoint.pt - inner_previous_keypoint.pt);

            // std::numeric_limits<double>::epsilon() == Difference between 1.0 and the next value representable
            // by the floating point type T. For int, char and so on, it is 0, but for floating point types
            // (such as float, double and long double) it is a very very very small number

            if (distance_prev > std::numeric_limits<double>::epsilon() && distance_curr >= min_dist)
            { // avoid division by zero using epsilon

                double distRatio = distance_curr / distance_prev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // Only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN; 
        return;
    }

    // Compute median filter to get rid of outliers (in terms of matches)

    // 1. Sort the value from lower to greater

    sort(distRatios.begin(), distRatios.end());

    // 2. Get the median index (central element index) and round

    float median_index = floor(distRatios.size() / 2.0); 
    
    // 3. Get the median value

    double median_dist_ratio = distRatios.size() % 2 == 0 ? (distRatios[median_index-1] + distRatios[median_index]) / 2.0 : distRatios[median_index];

    double dT = 1 / frameRate;
    TTC = -dT / (1 - median_dist_ratio);
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{ 
    // Auxiliary variables

    double dT = 1/frameRate;
    double lane_width = 4.0;
    float factor = 8.0;

    bool clear_outliers = true;
    bool median = true;
    bool min = true;

    double median_prev = 0, median_curr = 0;
    double min_x_prev = 1e9, min_x_curr = 1e9;

    // 1. Sort the values from lower to greater

    std::vector<double> x_prev, x_curr;

    for (auto it_prev = lidarPointsPrev.begin(); it_prev != lidarPointsPrev.end(); it_prev++)
    {
        x_prev.push_back(it_prev->x);
    }

    for (auto it_curr = lidarPointsCurr.begin(); it_curr != lidarPointsCurr.end(); it_curr++)
    {
        x_curr.push_back(it_curr->x);
    }

    sort(x_prev.begin(),x_prev.end());
    sort(x_curr.begin(),x_curr.end());

    // 2. Clear n % of points

    if (clear_outliers)
    {
        float percentage = 0.02;
        int to_erase = floor(x_prev.size()*percentage);
        x_prev.erase(x_prev.begin(),x_prev.begin()+to_erase);
        x_prev.erase(x_prev.end()-to_erase,x_prev.end());

        to_erase = floor(x_curr.size()*percentage);
        x_curr.erase(x_curr.begin(),x_curr.begin()+to_erase);
        x_curr.erase(x_curr.end()-to_erase,x_curr.end());
    }  

    // 3.1. Get min x values for each vector and compute LiDAR-based TTC

    if (min)
    {
        for (auto it = x_prev.begin(); it != x_prev.end(); it++)
        {
            min_x_prev = min_x_prev > *it ? *it : min_x_prev;
        }

        for (auto it = x_curr.begin(); it != x_curr.end(); it++)
        {
            min_x_curr = min_x_curr > *it ? *it : min_x_curr;
        }

        TTC = min_x_curr * dT / (min_x_prev - min_x_curr);
        //cout<<"TTC with min: "<<TTC<<endl;
    }

    // 3.2. Get the median for each vector and compute LiDAR-based TTC

    if (median)
    {
        float median_prev_index = floor(x_prev.size() / 2.0);
        float median_curr_index = floor(x_curr.size() / 2.0);

        median_prev = x_prev.size() % 2 == 0 ? (x_prev[median_prev_index -1] + x_prev[median_prev_index])/2.0 : x_prev[median_prev_index];
        median_curr = x_curr.size() % 2 == 0 ? (x_curr[median_curr_index -1] + x_curr[median_curr_index])/2.0 : x_curr[median_curr_index];
   
        TTC = median_curr * dT / (median_prev - median_curr);
        //cout<<"TTC with median: "<<TTC<<endl;
    }   
}

// Match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)

void matchBoundingBoxes(std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    std::vector<cv::DMatch> matches = currFrame.kptMatches;

    int rows = prevFrame.boundingBoxes.size();
    int cols = currFrame.boundingBoxes.size();

    cv::Mat matches_prev_curr_bbs(rows, cols, CV_8UC1, cv::Scalar(0));

    for (auto it = matches.begin(); it != matches.end(); it++)
    {
        int prev_index = (*it).queryIdx;
        int curr_index = (*it).trainIdx;

        cv::Point prev_pt;
        prev_pt.x = prevFrame.keypoints[prev_index].pt.x;
        prev_pt.y = prevFrame.keypoints[prev_index].pt.y;

        cv::Point curr_pt;
        curr_pt.x = currFrame.keypoints[curr_index].pt.x;
        curr_pt.y = currFrame.keypoints[curr_index].pt.y;

        // Check if prev_pt was included in any previous_bb

        vector<vector<BoundingBox>::iterator> prev_enclosingBoxes;
        int row = 0;
        for (auto it1 = prevFrame.boundingBoxes.begin(); it1 != prevFrame.boundingBoxes.end(); it1++)
        {
            if ((*it1).roi.contains(prev_pt))
            {
                row = it1 - prevFrame.boundingBoxes.begin();
                prev_enclosingBoxes.push_back(it1);
            }
        }

        // Check if curr_pt was included in any current_bb

        vector<vector<BoundingBox>::iterator> curr_enclosingBoxes;
        int col = 0;
        for (auto it2 = currFrame.boundingBoxes.begin(); it2 != currFrame.boundingBoxes.end(); it2++)
        {
            if ((*it2).roi.contains(curr_pt))
            {
                col = it2 - currFrame.boundingBoxes.begin();
                curr_enclosingBoxes.push_back(it2);
            }
        }

        if ((prev_enclosingBoxes.size() == 1) && (curr_enclosingBoxes.size() == 1)) 
        {
            matches_prev_curr_bbs.at<unsigned char>(row,col)++;
        }
    }

    // Find best match for each previous_bb

    double min=0, max=0;
    cv::Point minLoc, maxLoc;

    for (size_t i = 0; i < matches_prev_curr_bbs.rows; i++)
    {
        cv::Mat row = matches_prev_curr_bbs.row(i);
        cv::minMaxLoc(row, &min, &max, &minLoc, &maxLoc);

        if (max > 0)
        {
            // maxLoc.y = previous_bb, maxLoc.y = current_bb
            bbBestMatches.insert(std::pair<int,int> (i,maxLoc.x)); // i = our current row = previous_bb
        }
    }

    std::map<int, int> aux_map = bbBestMatches;
    std::map<int, int>::iterator it, it2;

    for (it = aux_map.begin(); it != aux_map.end(); it++)
    {
        for (it2 = aux_map.begin(); it2 != aux_map.end(); it2++)
        {
            if (it->second == it2->second && it->first != it2->first)
            {
                matches_prev_curr_bbs.at<unsigned char>(it->first,it->second) >= matches_prev_curr_bbs.at<unsigned char>(it2->first,it2->second) ? bbBestMatches.erase(it2->first) : bbBestMatches.erase(it->first);
            }
        }
    }

    bool bVis = false;

    if (bVis)
    {
        string windowName = "Matches";
        cv::namedWindow(windowName, 1);
        cv::imshow(windowName, matches_prev_curr_bbs);
        cv::waitKey(0);
    }
}
