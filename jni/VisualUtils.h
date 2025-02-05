//
// Created by Lingyun Wang on 2021/8/9.
//

#ifndef MOTIONTRACKING_VISUALUTILS_H
#define MOTIONTRACKING_VISUALUTILS_H

#include <opencv2/core/core.hpp>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;


class VisualUtils {
public:
    // 画特征点的匹配
    static void drawMatch(int frameIdx, vector<cv::Point2f> pts1, vector<cv::Point2f> pts2,
        cv::Mat frame1, cv::Mat frame2, string postfix);

    // 画跟踪边界
    static void drawBound(int frameIdx, cv::Mat frame, vector<cv::Point2f> scene_corners,
        cv::Mat motion, string postfix);

    // 画3D立方体
    static void drawCube(int frameIdx, cv::Mat frame, vector<cv::Point2f> projPts, string postfix);

    // 画运动矢量
    static void drawMotion(int frameIdx, cv::Mat frame,
    vector<cv::Point2f> prev_keypoints1, vector<cv::Point2f> next_keypoints1,
    vector<cv::Point2f> prev_keypoints2, vector<cv::Point2f> next_keypoints2);
};



#endif //MOTIONTRACKING_VISUALUTILS_H
