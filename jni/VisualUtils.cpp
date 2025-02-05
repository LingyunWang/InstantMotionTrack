//
// Created by Lingyun Wang on 2021/8/9.
//

#include "VisualUtils.h"
#include <sstream>


// 画特征点的匹配
void VisualUtils::drawMatch(int frameIdx, vector<cv::Point2f> pts1, vector<cv::Point2f> pts2,
    cv::Mat frame1, cv::Mat frame2, string postfix) {
    ostringstream ostr;
    ostr << "/sdcard/match/"<<frameIdx << postfix <<".jpg";
    cv::Mat img_match;
    vector<cv::KeyPoint> prevPts;
    vector<cv::KeyPoint> nextPts;
    vector<cv::DMatch> matches;
    for(int i = 0; i < pts1.size(); ++i) {
        cv::KeyPoint pt;
        pt.pt = pts1[i];
        prevPts.push_back(pt);
        pt.pt = pts2[i];
        nextPts.push_back(pt);
        matches.push_back(cv::DMatch(i, i, 1));
    }
    cv::drawMatches (frame1, prevPts, frame2, nextPts, matches, img_match );
    cv::imwrite(ostr.str(), img_match);
}

// 画跟踪边界
void VisualUtils::drawBound(int frameIdx, cv::Mat frame, vector<cv::Point2f> scene_corners, cv::Mat motion, string postfix) {
    line(frame,scene_corners[0],scene_corners[1],CV_RGB(255,0,0),1);
    line(frame,scene_corners[1],scene_corners[2],CV_RGB(255,0,0),1);
    line(frame,scene_corners[2],scene_corners[3],CV_RGB(255,0,0),1);
    line(frame,scene_corners[3],scene_corners[0],CV_RGB(255,0,0),1);
    // draw motion
    for (int i =0; i<4; ++i) {
        line(frame, scene_corners[i], cv::Point2f(scene_corners[i].x-motion.at<float>(0,0), scene_corners[i].y-motion.at<float>(1,0)),
            CV_RGB(0,0,255), 2);
    }
    ostringstream ostr;
    ostr << "/sdcard/out/" << frameIdx << postfix <<".jpg";
    cv::imwrite(ostr.str(), frame);
}

// 画立方体
void VisualUtils::drawCube(int frameIdx, cv::Mat frame, vector<cv::Point2f> projPts, string postfix) {
    line(frame,projPts[0],projPts[1],CV_RGB(198, 32, 212),1);
    line(frame,projPts[1],projPts[2],CV_RGB(198, 32, 212),1);
    line(frame,projPts[2],projPts[3],CV_RGB(198, 32, 212),1);
    line(frame,projPts[3],projPts[0],CV_RGB(198, 32, 212),1);
    line(frame,projPts[0],projPts[4],CV_RGB(128, 64, 176),1);
    line(frame,projPts[1],projPts[5],CV_RGB(128, 64, 176),1);
    line(frame,projPts[2],projPts[6],CV_RGB(128, 64, 176),1);
    line(frame,projPts[3],projPts[7],CV_RGB(128, 64, 176),1);
    line(frame,projPts[4],projPts[5],CV_RGB(108, 202, 76),1);
    line(frame,projPts[5],projPts[6],CV_RGB(108, 202, 76),1);
    line(frame,projPts[6],projPts[7],CV_RGB(108, 202, 76),1);
    line(frame,projPts[7],projPts[4],CV_RGB(108, 202, 76),1);
    ostringstream ostr;
    ostr << "/sdcard/out/" << frameIdx << postfix <<".jpg";
    cv::imwrite(ostr.str(), frame);
}

// 画运动矢量
void VisualUtils::drawMotion(int frameIdx, cv::Mat frame,
    vector<cv::Point2f> prev_keypoints1, vector<cv::Point2f> next_keypoints1,
    vector<cv::Point2f> prev_keypoints2, vector<cv::Point2f> next_keypoints2) {
    for (int i = 0; i < prev_keypoints1.size(); ++i) {
        line(frame, prev_keypoints1[i], next_keypoints1[i], CV_RGB(255,0,0), 1);
    }
    for (int i = 0; i < prev_keypoints2.size(); ++i) {
        line(frame, prev_keypoints2[i], next_keypoints2[i], CV_RGB(0,255,0), 1);
    }
}
