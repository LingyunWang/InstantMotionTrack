//
// Created by Lingyun Wang on 2021/7/21.
//

#ifndef RECORDER_TRACKER_H
#define RECORDER_TRACKER_H
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <list>
#include <string>
using namespace std;

class TrackResult {
public:
    vector<cv::Point2f> corners;
    vector<cv::Point2f> projectionPts; // 立方体的投影点，临时使用
    cv::Mat rotation;
    cv::Mat translation;
};

class Tracker {
private:
    vector<cv::Point2f> trackedKeypoints;
    vector<float> trackedWeights;       //特征点权重
    cv::Mat prevFrame;

    uint mFrameIdx;
    // 目标区域
    vector<cv::Point2f> obj_corners;
    cv::Rect mObjRegion;
    // 跟踪区域
    vector<cv::Point2f> mTrackedCorners;
    cv::Rect mTrackedRegion;
    cv::Mat mLastMotion;
    double mAvgTimeCost;

public:
    Tracker();
    Tracker(cv::Mat frame, int x, int y, int w, int h);
    // 初始化跟踪目标
    bool initTrack(cv::Mat frame, int x, int y, int w, int h);
    // 是否跟踪成功
    bool continueTrack(cv::Mat frame, TrackResult& trackResult, bool isDownSample = true);
    // 记录内部状态 用于调试
    void saveStatus(string error, cv::Mat frame, cv::Mat frameDraw);
    // 恢复内部状态 用于调试
    void loadStatus(const string& dir, int frame);
};


#endif //RECORDER_TRACKER_H
