//
// Created by Lingyun Wang on 2021/7/21.
//

#include <vector>
#include <sstream>
#include <chrono>
#include <math.h>
#include <fstream>
#include "Tracker.h"
#include "VisualUtils.h"
#include "log.h"

using namespace std;
#define MIN_TRACK_NUM 10
#define RESERVE_NUM 100
#define MAX_FEATURE_NUM 150
#define DOWN_SAMPLE_RADIO 2

const int ERROR = 4;
const int WARN = 3;
const int INFO = 2;
const int TRACE = 1;
const int DEBUG = 0;
const int LEVEL = ERROR;

// 是否达到调试等级
bool isDebug(int level) {
    return level >= LEVEL;
}

Tracker::Tracker() {
}
Tracker::Tracker(cv::Mat frame, int x, int y, int w, int h) {
    initTrack(frame, x, y, w, h);
}

bool KeyCompartor(const cv::KeyPoint &v1, const cv::KeyPoint &v2) {
    return v1.response > v2.response;//降序排列
}

// 约束大小和边界
void regularRect(int frameWidth, int frameHeight, cv::Rect& targetRegion) {
    targetRegion.width = targetRegion.width < 30 ? 30 : targetRegion.width;
    targetRegion.height = targetRegion.height < 30 ? 30 : targetRegion.height;

    // constrain bound
    int xmin = targetRegion.x;
    int xmax = targetRegion.x + targetRegion.width;
    int ymin = targetRegion.y;
    int ymax = targetRegion.y + targetRegion.height;

    if (xmin < 0) {
        xmin = 0;
        xmax = targetRegion.width;
    }
    if (xmax > frameWidth) {
        xmax = frameWidth;
        xmin = frameWidth - targetRegion.width;
    }
    if (ymin < 0) {
        ymin = 0;
        ymax = targetRegion.height;
    }
    if (ymax > frameHeight) {
        ymax = frameHeight;
        ymin = frameHeight - targetRegion.height;
    }

    targetRegion.x = xmin;
    targetRegion.y = ymin;
    targetRegion.width = xmax - xmin;
    targetRegion.height = ymax - ymin;
}

bool Tracker::initTrack(cv::Mat frame, int x, int y, int w, int h) {
    // 降采样1/2
    cv::Mat frameScale;
    cv::resize(frame, frameScale, cv::Size(frame.cols/DOWN_SAMPLE_RADIO, frame.rows/DOWN_SAMPLE_RADIO));
    frame = frameScale;
    x = x / DOWN_SAMPLE_RADIO;
    y = y / DOWN_SAMPLE_RADIO;
    w = w / DOWN_SAMPLE_RADIO;
    h = h / DOWN_SAMPLE_RADIO;

    mFrameIdx = 0;
    prevFrame = frame;

    // 初始化目标区域
    mObjRegion = cv::Rect(x, y, w, h);
    regularRect(frame.cols, frame.rows, mObjRegion);
    obj_corners.clear();
    obj_corners.push_back(cv::Point2f(mObjRegion.x, mObjRegion.y));
    obj_corners.push_back(cv::Point2f(mObjRegion.x+mObjRegion.width, mObjRegion.y));
    obj_corners.push_back(cv::Point2f(mObjRegion.x+mObjRegion.width, mObjRegion.y+mObjRegion.height));
    obj_corners.push_back(cv::Point2f(mObjRegion.x, mObjRegion.y+mObjRegion.height));
    // 初始化跟踪区域
    mTrackedRegion = mObjRegion;
    mTrackedCorners = obj_corners;

    // 提取特征点
    trackedKeypoints.clear();
    // 跟踪区域提取 用goodFeatures
    vector<cv::Point2f> kps;
    cv::goodFeaturesToTrack(frame(mTrackedRegion), kps, 300, 0.01, 10);
    for ( auto kp:kps ) {
        if (trackedKeypoints.size() >= MAX_FEATURE_NUM) {
            break;
        }
        trackedKeypoints.push_back(cv::Point2f(kp.x + x, kp.y+y));
    }
    // 初始化权重
    for (int i = 0; i<trackedKeypoints.size(); ++i) {
        trackedWeights.push_back(1.0);
    }
    LOGV("initial key points %lu", trackedKeypoints.size());
    if (trackedKeypoints.size() < MIN_TRACK_NUM) {
        return false;
    }
    return true;
}

void getCornersRange(vector<cv::Point2f> corners, cv::Rect& range) {
    float xmin = corners[0].x;
    float xmax = corners[0].x;
    float ymin = corners[0].y;
    float ymax = corners[0].y;
    for (int i = 1; i < corners.size(); ++i) {
        xmin = corners[i].x < xmin ? corners[i].x : xmin;
        xmax = corners[i].x > xmax ? corners[i].x : xmax;
        ymin = corners[i].y < ymin ? corners[i].y : ymin;
        ymax = corners[i].y > ymax ? corners[i].y : ymax;
    }
    range = cv::Rect(xmin, ymin, xmax-xmin, ymax-ymin);
}

// 从相似变换更新目标的corners
vector<cv::Point2f> similarityTransform(cv::Mat transform, const vector<cv::Point2f>& corners) {
    float sx = transform.at<float>(0, 0);
    float sy = transform.at<float>(1, 1);
    float bx = transform.at<float>(0, 2);
    float by = transform.at<float>(1, 2);
    vector<cv::Point2f> newCorners;
    for (auto corner : corners) {
        newCorners.push_back(cv::Point2f(corner.x * sx + bx, corner.y * sy + by));
    }
    return newCorners;
}

// 从相似变换更新目标的矩形区域
bool updateTargetRegion(int frameWidth, int frameHeight, cv::Mat transform, vector<cv::Point2f> corners,
                        cv::Rect& targetRegion, vector<cv::Point2f>& newCorners) {
    bool ret = false;
    newCorners.clear();
    newCorners = similarityTransform(transform, corners);

    float xmin = newCorners[0].x;
    float xmax = newCorners[1].x;
    float ymin = newCorners[0].y;
    float ymax = newCorners[2].y;
    float width = xmax - xmin;
    float height = ymax - ymin;

    // 约束大小和边界
    // constrain size
    if (width < 30) {
        width = 30;
    }
    if (height < 30) {
        height = 30;
    }
    // constrain bound
    if (xmin < 0) {
        xmin = 0;
        xmax = width;
    }
    if (xmax > frameWidth) {
        xmax = frameWidth;
        xmin = frameWidth - width;
    }
    if (ymin < 0) {
        ymin = 0;
        ymax = height;
    }
    if (ymax > frameHeight) {
        ymax = frameHeight;
        ymin = frameHeight - height;
    }
    targetRegion.x = xmin;
    targetRegion.y = ymin;
    targetRegion.width = xmax - xmin;
    targetRegion.height = ymax - ymin;
    newCorners[0] = cv::Point2f(xmin, ymin);
    newCorners[1] = cv::Point2f(xmin+width, ymin);
    newCorners[2] = cv::Point2f(xmin+width, ymin+height);
    newCorners[3] = cv::Point2f(xmin, ymin+height);
    return ret;
}
// 从相似变换更新目标的矩形区域
bool updateTargetRegionByMotion(int frameWidth, int frameHeight, cv::Mat motion, vector<cv::Point2f> corners,
                        vector<cv::Point2f>& newCorners) {
    bool ret = false;
    newCorners.clear();
    for (auto corner : corners) {
        newCorners.push_back(cv::Point2f(corner.x + motion.at<float>(0, 0), corner.y + motion.at<float>(1, 0)));
    }

    float xmin = newCorners[0].x;
    float xmax = newCorners[1].x;
    float ymin = newCorners[0].y;
    float ymax = newCorners[2].y;
    float width = xmax - xmin;
    float height = ymax - ymin;

    // 约束大小和边界
    // constrain size
    if (width < 30) {
        width = 30;
    }
    if (height < 30) {
        height = 30;
    }
    // constrain bound
    if (xmin < 0) {
        xmin = 0;
        xmax = width;
    }
    if (xmax > frameWidth) {
        xmax = frameWidth;
        xmin = frameWidth - width;
    }
    if (ymin < 0) {
        ymin = 0;
        ymax = height;
    }
    if (ymax > frameHeight) {
        ymax = frameHeight;
        ymin = frameHeight - height;
    }
    newCorners[0] = cv::Point2f(xmin, ymin);
    newCorners[1] = cv::Point2f(xmin+width, ymin);
    newCorners[2] = cv::Point2f(xmin+width, ymin+height);
    newCorners[3] = cv::Point2f(xmin, ymin+height);
    return ret;
}
// 扩展区域用于光流的跟踪
void extendRegion(int frameWidth, int frameHeight, cv::Rect& srcRegion, cv::Rect& regionEx) {
    float xmin = srcRegion.x - srcRegion.width/5;
    float xmax = srcRegion.x + srcRegion.width * 1.2;
    float ymin = srcRegion.y - srcRegion.height/5;
    float ymax = srcRegion.y + srcRegion.height * 1.2;
    xmin = xmin < 0 ? 0 : xmin;
    xmax = xmax > frameWidth ? frameWidth : xmax;
    ymin = ymin < 0 ? 0 : ymin;
    ymax = ymax > frameHeight ? frameHeight : ymax;
    regionEx = cv::Rect(xmin, ymin, xmax-xmin, ymax-ymin);
}

// 用homography约束过滤外点
cv::Mat refineWithHomography(const vector<cv::Point2f>& prevPts, const vector<cv::Point2f>& nextPts,
                             vector<cv::Point2f>& prevRefinePts, vector<cv::Point2f>& nextRefinePts) {
    std::vector<unsigned char> inliersMask;
    cv::Mat homography = cv::findHomography(prevPts, nextPts, cv::FM_RANSAC, 4, inliersMask);
    for (int i = 0; i < inliersMask.size(); ++i) {
        if (inliersMask[i]) {
            prevRefinePts.push_back(prevPts[i]);
            nextRefinePts.push_back(nextPts[i]);
        }
    }
    return homography;
}

cv::Mat cacuSimilarityLS(const vector<cv::Point2f>& prevPts, const vector<cv::Point2f>& nextPts, const vector<float> weights) {
    if (prevPts.size() != nextPts.size()) {
        return cv::Mat();
    }
    float bx = 0;
    float by = 0;
    float sx = 0;
    float sy = 0;

    float bResSumX = 0;
    float bResSumY = 0;
    float weightSum = 0;
    for (int i = 0; i<prevPts.size(); ++i) {
        bResSumX += weights[i]*(nextPts[i].x - prevPts[i].x);
        bResSumY += weights[i]*(nextPts[i].y - prevPts[i].y);
        weightSum += weights[i];
    }
    bx = bResSumX / weightSum;
    by = bResSumY / weightSum;

    float sx_p1 = 0;
    float sx_p2 = 0;
    float sx_p3 = 0;
    float sy_p1 = 0;
    float sy_p2 = 0;
    float sy_p3 = 0;
    for (int i = 0; i< prevPts.size(); ++i) {
        sx_p1 += weights[i] * nextPts[i].x * prevPts[i].x;
        sx_p2 += weights[i] * bx * prevPts[i].x;
        sx_p3 += weights[i] * prevPts[i].x * prevPts[i].x;
        sy_p1 += weights[i] * nextPts[i].y * prevPts[i].y;
        sy_p2 += weights[i] * by * prevPts[i].y;
        sy_p3 += weights[i] * prevPts[i].y * prevPts[i].y;
    }
    sx = (sx_p1 - sx_p2)/sx_p3;
    sy = (sy_p1 - sy_p2)/sy_p3;
    cv::Mat ret = cv::Mat::eye(3, 3, CV_32F);
    ret.at<float>(0,0) = sx;
    ret.at<float>(1,1) = sy;
    ret.at<float>(0,2) = bx;
    ret.at<float>(1,2) = by;
    return ret;
}

void normalize(vector<float>& weights) {
    if (weights.size() == 0) {
        return;
    }
    float weightsSum = 0;
    for(int i=0; i < weights.size(); ++i) {
        weightsSum += weights[i];
    }
    if (weightsSum == 0) {
        return;
    }
    for(int i=0; i<weights.size(); ++i) {
        weights[i] *= (weights.size() / weightsSum);
    }
}

// 使用IRLS来过滤外点  返回平均运动矢量   weights是归一化的
cv::Mat refineWithIRLS(const vector<cv::Point2f>& prevPts, const vector<cv::Point2f>& nextPts, const vector<float> weights,
                       vector<cv::Point2f>& prevRefinePts, vector<cv::Point2f>& nextRefinePts, vector<float>& refineWeights) {
    prevRefinePts.clear();
    nextRefinePts.clear();
    refineWeights.clear();
    // debug keypoint motion
    /*LOGD("input");
    for (int i = 0; i<prevPts.size(); ++i) {
        LOGD("%d:(%f %f) %f", i, nextPts[i].x - prevPts[i].x, nextPts[i].y - prevPts[i].y, weights[i]);
    }*/
    float sum = 0;
    for (auto w : weights) {
        sum += w;
    }

    float meanVx = 0;
    float meanVy = 0;
    float meanVLength = 0;
    vector<float> newWeights = weights;
    for (int k = 0; k < 20; ++k) {
        // 权重归一化
        normalize(newWeights);
        float prevMeanVx = meanVx;
        float prevMeanVy = meanVy;
        meanVx = 0;
        meanVy = 0;
        // 计算平均速度。
        for (int i = 0; i<prevPts.size(); ++i) {
            meanVx += (nextPts[i].x - prevPts[i].x) * newWeights[i];
            meanVy += (nextPts[i].y - prevPts[i].y) * newWeights[i];
        }
        meanVx /= newWeights.size();
        meanVy /= newWeights.size();
        // 迭代以后的平均速度改变不大，停止迭代。
        if (pow(meanVx - prevMeanVx, 2) + pow(meanVy - prevMeanVy, 2) < 0.0001) {
            break;
        }
        // 通过与平均速度的相似度更新权重
        meanVLength = sqrt(pow(meanVx, 2) + pow(meanVy, 2));
        for(int i = 0; i<prevPts.size(); ++i) {
            // L2距离
            float distance = sqrt(pow(nextPts[i].x - prevPts[i].x - meanVx, 2) + pow(nextPts[i].y - prevPts[i].y - meanVy, 2));
            newWeights[i] = 1 / (1 + distance / (0.5 + meanVLength));
        }
    }
    //LOGD("motion %f %f", meanVx, meanVy);
    // 过滤外点
    meanVLength = sqrt(pow(meanVx, 2) + pow(meanVy, 2));
    for (int i = 0; i < prevPts.size(); ++i) {
        // L2距离
        float distance = sqrt(pow(nextPts[i].x - prevPts[i].x - meanVx, 2) + pow(nextPts[i].y - prevPts[i].y - meanVy, 2));
        //LOGD("%d: %f (%f %f) %f", i, 1/(1+distance/(0.2+meanVLength)), nextPts[i].x - prevPts[i].x, nextPts[i].y - prevPts[i].y, newWeights[i]);
        // 相似度
        if (1/(1+distance/(0.5+meanVLength)) > 0.6) {
            prevRefinePts.push_back(prevPts[i]);
            nextRefinePts.push_back(nextPts[i]);
            refineWeights.push_back(newWeights[i]);
        }
    }
    cv::Mat vector(2, 1, CV_32F);
    vector.at<float>(0, 0) = meanVx;
    vector.at<float>(1, 0) = meanVy;
    return vector;
}

// 是否跟踪成功
bool Tracker::continueTrack(cv::Mat frame, TrackResult& trackResult, bool isDownSample) {
    mFrameIdx += 1;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    // 降采样 DOWN_SAMPLE_RADIO
    if (isDownSample) {
        cv::Mat frameScale;
        cv::resize(frame, frameScale, cv::Size(frame.cols / DOWN_SAMPLE_RADIO, frame.rows / DOWN_SAMPLE_RADIO), 0, 0, cv::INTER_NEAREST);
        frame = frameScale;
    }
    chrono::steady_clock::time_point t11 = chrono::steady_clock::now();
    LOGV("Resize use time %f seconds", chrono::duration_cast<chrono::duration<double>>(t11 - t1).count());

    // 间隔一帧运动估计更新
    if (mFrameIdx % 2 == 0) {
        vector<cv::Point2f> newCorners;
        updateTargetRegionByMotion(frame.cols, frame.rows, mLastMotion, mTrackedCorners, newCorners);
        // 跟踪区域的点需要还原到原有分辨率下面
        for (auto sceneCorner : newCorners) {
            trackResult.corners.push_back(cv::Point2f(sceneCorner.x * DOWN_SAMPLE_RADIO, sceneCorner.y * DOWN_SAMPLE_RADIO));
        }
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        double frameTime = chrono::duration_cast<chrono::duration<double>>(t2 - t1).count();
        mAvgTimeCost = (mAvgTimeCost * (mFrameIdx - 1) + frameTime) / mFrameIdx;
        LOGV("Total use time %f %f seconds", frameTime, mAvgTimeCost);
        return true;
    }

    // 可视化的图
    cv::Mat frameDraw;
    cv::cvtColor(frame, frameDraw, cv::COLOR_GRAY2BGR);

    // 跟踪的点越来越少，重新提取特征点。
    if (trackedKeypoints.size() < RESERVE_NUM) {
        // 截取上一帧跟踪区域提取特征点。
        vector<cv::Point2f> kps;
        //frame(mTrackedRegion)
        cv::goodFeaturesToTrack(prevFrame(mTrackedRegion), kps, 500, 0.01, 10);
        for ( auto kp:kps ) {
            if (trackedKeypoints.size() >= MAX_FEATURE_NUM) {
                break;
            }
            cv::Point2f ptInFrame(kp.x + mTrackedRegion.x , kp.y + mTrackedRegion.y);
            trackedKeypoints.push_back(ptInFrame);
        }
        LOGV("add keypoint %lu", trackedKeypoints.size());
        // 为新增点初始化权重, 添加前归一化权重
        int addSize = trackedKeypoints.size() - trackedWeights.size();
        normalize(trackedWeights);
        for (int i = 0; i < addSize; ++i) {
            trackedWeights.push_back(1);
        }
        if (trackedKeypoints.size() < MIN_TRACK_NUM) {
            LOGV("detect keypoints too less");
            if (isDebug(WARN)) {
                saveStatus("detect keypoints too less", frame, cv::Mat());
            }
            return false;
        }
    }
    chrono::steady_clock::time_point t12 = chrono::steady_clock::now();
    LOGV("Feature extract use time %f seconds", chrono::duration_cast<chrono::duration<double>>(t12 - t11).count());

    // 光流匹配
    chrono::steady_clock::time_point t21 = chrono::steady_clock::now();
    cv::Rect targetRegionEx;    //光流匹配区域
    extendRegion(frame.cols, frame.rows, mTrackedRegion, targetRegionEx);
    vector<cv::Point2f> prev_keypoints;
    for (auto pt : trackedKeypoints) {
        prev_keypoints.push_back(cv::Point2f(pt.x - targetRegionEx.x, pt.y - targetRegionEx.y));
    }
    vector<cv::Point2f> next_keypoints;
    cv::Mat sourceSubFrame = prevFrame(targetRegionEx);
    cv::Mat targetSubFrame = frame(targetRegionEx);
    vector<unsigned char> status;
    vector<float> error;
    cv::calcOpticalFlowPyrLK(sourceSubFrame, targetSubFrame, prev_keypoints, next_keypoints, status, error, cv::Size(21,21), 2);

    vector<cv::Point2f> prev_keypoints2;
    vector<cv::Point2f> next_keypoints2;
    vector<float> weights2;
    for (int i = 0; i < status.size(); ++i) {
        if (status[i] == 1) {
            prev_keypoints2.push_back(cv::Point2f(prev_keypoints[i].x + targetRegionEx.x, prev_keypoints[i].y + targetRegionEx.y));
            next_keypoints2.push_back(cv::Point2f(next_keypoints[i].x + targetRegionEx.x, next_keypoints[i].y + targetRegionEx.y));
            weights2.push_back(trackedWeights[i]);
        }
    }
    // 画当前的匹配
    if (isDebug(DEBUG)) {
        VisualUtils::drawMatch(mFrameIdx, prev_keypoints2, next_keypoints2, prevFrame, frame, "opticalflow");
    }
    LOGV("optical flow match num = %lu", prev_keypoints2.size());
    if (next_keypoints2.size() < MIN_TRACK_NUM) {
        LOGV("optical flow points too less");
        // 画点的位移
        if (isDebug(WARN)) {
            VisualUtils::drawMotion(mFrameIdx, frameDraw, prev_keypoints, next_keypoints, prev_keypoints2, prev_keypoints2);
            saveStatus("optical flow points too less", frame, frameDraw);
        }
        return false;
    }
    // time cost
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t21);
    LOGV("optical flow use time：%f seconds", time_used.count());

    // IRLS refine
    vector<cv::Point2f> prev_keypoints3;
    vector<cv::Point2f> next_keypoints3;
    vector<float> weights3;
    cv::Mat motion = refineWithIRLS(prev_keypoints2, next_keypoints2, weights2, prev_keypoints3, next_keypoints3, weights3);
    mLastMotion = motion;
    // 画点的位移
    if (isDebug(TRACE)) {
        VisualUtils::drawMotion(mFrameIdx, frameDraw, prev_keypoints2, next_keypoints2, prev_keypoints3, next_keypoints3);
    }
    LOGV("IRLS match %lu", next_keypoints3.size());
    if (next_keypoints3.size() < MIN_TRACK_NUM) {
        LOGV("IRLS match too less");
        if (isDebug(WARN)) {
            VisualUtils::drawMotion(mFrameIdx, frameDraw, prev_keypoints2, next_keypoints2, prev_keypoints3, next_keypoints3);
            saveStatus("IRLS match too less", frame, frameDraw);
        }
        return false;
    }

    // 使用相似变换更新目标区域
    cv::Mat similarityMat = cacuSimilarityLS(prev_keypoints3, next_keypoints3, weights3);
    if (similarityMat.data == nullptr) {
        LOGV("find similarity mat null");
        if (isDebug(WARN)) {
            saveStatus("find similarity mat null", frame, frameDraw);
        }
        return false;
    }
    // 更新跟踪区域 更新跟踪corners
    vector<cv::Point2f> newCorners;
    updateTargetRegion(frame.cols, frame.rows, similarityMat, mTrackedCorners, mTrackedRegion, newCorners);
    mTrackedCorners = newCorners;

    // draw bound
    if (isDebug(TRACE)) {
        VisualUtils::drawBound(mFrameIdx, frameDraw, mTrackedCorners, motion, "scale");
    }
    // time cost
    chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t3 - t2);
    LOGV("refine use time：%f seconds", time_used.count());

    // 计算位姿
    cv::Mat r, t;
    vector<cv::Point3f> objPoints;
    objPoints.push_back(cv::Point3f(-1, 1, 0));
    objPoints.push_back(cv::Point3f(1, 1, 0));
    objPoints.push_back(cv::Point3f(1, -1, 0));
    objPoints.push_back(cv::Point3f(-1, -1, 0));
    cv::Mat K = ( cv::Mat_<double> ( 3,3 ) << 3030.351473360361/DOWN_SAMPLE_RADIO, 0, 994.3512051559228/DOWN_SAMPLE_RADIO, 0, 3086.405873208551/DOWN_SAMPLE_RADIO, 512.3981410434634/DOWN_SAMPLE_RADIO, 0, 0, 1 );
    cv::solvePnP(objPoints, mTrackedCorners, K, cv::Mat(), r, t, false);
    cv::Mat rotM;
    cv::Rodrigues(r,rotM);

    // time cost
    chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t4 - t3);
    LOGV("PnP use time：%f seconds", time_used.count());
    double frameTime = chrono::duration_cast<chrono::duration<double>>(t4 - t1).count();
    mAvgTimeCost = (mAvgTimeCost * (mFrameIdx - 1) + frameTime) / mFrameIdx;
    LOGV("Total use time %f %f seconds", frameTime, mAvgTimeCost);

    // update reserved frames
    prevFrame = frame;
    trackedKeypoints = next_keypoints3;
    trackedWeights = weights3;

    trackResult.corners.clear();
    // 跟踪区域的点需要还原到原有分辨率下面
    for (auto sceneCorner : mTrackedCorners) {
        trackResult.corners.push_back(cv::Point2f(sceneCorner.x * DOWN_SAMPLE_RADIO, sceneCorner.y * DOWN_SAMPLE_RADIO));
    }
    //  虚拟立方体AR效果
    /*vector<cv::Mat> objPts;  // 虚拟立方体的8个点
    vector<cv::Point2f> objProjPts;   // 虚拟立方体的8个投影点
    objPts.push_back((cv::Mat_<double>(3, 1) << -1.0/2, -1.0/2, 0));
    objPts.push_back((cv::Mat_<double>(3, 1) << 1.0/2, -1.0/2, 0));
    objPts.push_back((cv::Mat_<double>(3, 1) << 1.0/2, 1.0/2, 0));
    objPts.push_back((cv::Mat_<double>(3, 1) << -1.0/2, 1.0/2, 0));
    objPts.push_back((cv::Mat_<double>(3, 1) << -1.0/2, -1.0/2, 1.0));
    objPts.push_back((cv::Mat_<double>(3, 1) << 1.0/2, -1.0/2, 1.0));
    objPts.push_back((cv::Mat_<double>(3, 1) << 1.0/2, 1.0/2, 1.0));
    objPts.push_back((cv::Mat_<double>(3, 1) << -1.0/2, 1.0/2, 1.0));
    for (int i =0; i<objPts.size(); ++i) {
        cv::Mat pt_projection = K * (rotM * objPts[i] + t);
        objProjPts.push_back(cv::Point2f(pt_projection.at<double>(0)/pt_projection.at<double>(2), pt_projection.at<double>(1)/pt_projection.at<double>(2)));
    }
    if (isDebug(TRACE)) {
        VisualUtils::drawCube(mFrameIdx, frameDraw, objProjPts, "scale");
    }*/
    /*trackResult.projectionPts.clear();
    for (auto objProjPt : objProjPts) {
        trackResult.projectionPts.push_back(cv::Point2f(objProjPt.x * DOWN_SAMPLE_RADIO, objProjPt.y * DOWN_SAMPLE_RADIO));
    }*/
    trackResult.rotation = r;
    trackResult.translation = t;
    return true;
}

// 记录内部状态 用于调试
void Tracker::saveStatus(string error, cv::Mat frame, cv::Mat frameDraw) {
    ostringstream ostr;
    ostr << "/sdcard/status/" << mFrameIdx <<".txt";
    ofstream outfile(ostr.str());
    // record trackedKeypoints
    for (auto pt : trackedKeypoints) {
        outfile << pt.x << " " << pt.y << " ";
    }
    outfile << endl;
    // record trackedWeights
    for (auto w : trackedWeights) {
        outfile << w << " ";
    }
    outfile << endl;
    // record mTrackedCorners
    for (auto pt : mTrackedCorners){
        outfile << pt.x << " " << pt.y << " ";
    }
    outfile << endl;
    // record mTrackedRegion
    outfile << mTrackedRegion.x << " " << mTrackedRegion.y << " " << mTrackedRegion.width << " " << mTrackedRegion.height << endl;
    // record error
    outfile << error << endl;
    outfile.close();

    ostr.str("");
    ostr << "/sdcard/status/" << mFrameIdx <<".jpg";
    cv::imwrite(ostr.str(), frame);
    if (frameDraw.data != nullptr) {
        ostr.str("");
        ostr << "/sdcard/status/" << mFrameIdx <<"lost.jpg";
        cv::imwrite(ostr.str(), frame);
    }
}

// 恢复内部状态 用于调试
void Tracker::loadStatus(const string& dir, int frame) {
    mFrameIdx = frame - 1;
    ostringstream ostr;
    ostr << dir << frame <<".txt";
    ifstream readFile(ostr.str());
    string tmp;
    // read tracked points
    getline(readFile, tmp);
    istringstream istr1(tmp);
    vector<float> vals;
    float val;
    while(istr1 >> val) {
        vals.push_back(val);
    }
    trackedKeypoints.clear();
    for (int i = 0; i < vals.size()/2; ++i) {
        trackedKeypoints.push_back(cv::Point2f(vals[i * 2], vals[i * 2 + 1]));
    }

    // read tracked poionts weight
    getline(readFile, tmp);
    istringstream istr2(tmp);
    trackedWeights.clear();
    while(istr1 >> val) {
        trackedWeights.push_back(val);
    }

    // read tracked corners
    getline(readFile, tmp);
    istringstream istr3(tmp);
    vals.clear();
    while(istr3 >> val) {
        vals.push_back(val);
    }
    mTrackedCorners.clear();
    for (int i = 0; i < vals.size()/2; ++i) {
        mTrackedCorners.push_back(cv::Point2f(vals[i * 2], vals[i * 2 + 1]));
    }

    //read tracked region
    getline(readFile, tmp);
    istringstream istr4(tmp);
    vals.clear();
    while(istr3 >> val) {
        vals.push_back(val);
    }
    mTrackedRegion.x = vals[0];
    mTrackedRegion.y = vals[1];
    mTrackedRegion.width = vals[2];
    mTrackedRegion.height = vals[3];
}