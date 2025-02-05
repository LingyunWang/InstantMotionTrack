//
//  main.cpp
//  MotionTracking
//
//  Created by Lingyun Wang on 2021/7/20.
//

#include "main.hpp"
#include "Tracker.h"
#include <sstream>
#include <iostream>

using namespace std;
void runSequence() {
    cv::Mat frame0 = cv::imread("/sdcard/dump/1168.jpg");
    cv::Mat gray;
    cv::cvtColor(frame0, gray, cv::COLOR_BGR2GRAY);
    Tracker tracker = Tracker(gray, 600, 824, 200, 200);  //face 700, 350, 460, 460  //shoulder 410, 670, 300, 400 //small 200, 395, 153, 102
    TrackResult result;
    for(int i = 1169; i < 1662; ++i) {
        ostringstream ss;
        ss << "/sdcard/dump/"<<i<<".jpg";
        cv::Mat frame = cv::imread(ss.str(), 0);
        cout << ss.str() << endl;
        if (!tracker.continueTrack(frame, result)){
            cout << "track lost" << endl;
            break;
        }
        cout << endl;
    }
}

void debugLost() {
    Tracker tracker;

    tracker.loadStatus("/sdcard/status", 92);
    ostringstream ss;
    ss << "/sdcard/status/"<<92<<".jpg";
    cv::Mat frame = cv::imread(ss.str(), 0);
    TrackResult result;
    tracker.continueTrack(frame, result, false);
}

int main() {
    runSequence();
    return 0;
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
using namespace cv;
int calibrate()
{
	//读取每一幅图像，从中提取角点，然后对角点进行亚像素精确化

	cout << "开始提取角点...";
	int image_count = 0; /*图像数量*/
	Size image_size; /*图像尺寸*/
	Size board_size = Size(6, 9);  //标定板每行列角点数

	vector<Point2f> image_points_buf; /*缓存每幅图像上检测到的角点*/
	vector<vector<Point2f>> image_point_seq;  /*保存所有检测到的角点*/

	string filename;
	for (int i =0; i<15; ++i)
	{
	    ostringstream ostr;
	    ostr << "/sdcard/cali/"<<i<<".jpg";
	    filename = ostr.str();
		image_count++;
		// 用于观察检验输出
		cout << "image_count = " << image_count << endl;
		Mat grayImg = imread(filename,IMREAD_GRAYSCALE);  //读入灰度图

		if (image_count == 1)
		{
			// 读入第一张图片时获取图像的宽高信息
			image_size.width = grayImg.cols;
			image_size.height = grayImg.rows;

			cout << "image_size.width = " << image_size.width << endl;
			cout << "image_size.height = " << image_size.height << endl;
		}
		if (0 == findChessboardCorners(grayImg, board_size, image_points_buf)) /*输入图像必须是8位灰度或彩色图像*/
		{
			cout << "can not find chessboard corners!\n";  //没有找到全部角点
			exit(1);
		} else {
			/*亚像素精确化*/
			find4QuadCornerSubpix(grayImg, image_points_buf, Size(11, 11));  //粗提取角点 精确化
			/*保存亚像素点*/
			image_point_seq.push_back(image_points_buf);
			/*绘制出检测到的角点(彩色)*/
			//Mat rgbImg;
            //cvtColor(grayImg, rgbImg, COLOR_GRAY2BGR);
			//drawChessboardCorners(rgbImg, board_size, image_points_buf, true);
		}
	}
	int total = image_point_seq.size();  //读取的总的图像数
	cout << "total = " << total << endl;
	int CornerNum = board_size.width * board_size.height;  /*总的角点数*/

	for (int i = 0; i < total; i++)
	{
		if (0 == i % CornerNum)
		{
			int j = -1;
			j = i / CornerNum;
			int k = j + 1;
			cout << "--> 第 " << k << "图片的数据 -->" << endl;
		}
		if (0 == i % 3)
		{
			cout << endl;
		}
		else
		{
			cout.width(10);
		}
		//输出所有图像中第一个角点坐标
		cout << "-->" << image_point_seq[i][0].x;
		cout << "-->" << image_point_seq[i][0].y;
	}
	cout << "角点提取完成！\n";

	//摄像机标定
	cout << "开始标定..." << endl;
	Size square_size = Size(40, 40); /*棋盘格大小40mm*/

	vector<vector<Point3f>>  object_points; /*保存棋盘上的坐标点*/
	vector<int>  point_counts; //每幅图像中的角点的数量

	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));  /*摄像机内参数矩阵*/
	Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));  /*摄像机的5个畸变系数：k1,k2,p1,p2,k3*/

	vector<Mat> tvecsMat; /*存储每幅图像的旋转向量*/
	vector<Mat> rvecsMat; /*存储每幅图像的平移向量*/


	/*设置棋盘点的三维坐标*/
	for (int t = 0; t < image_count; t++)
	{
		vector<Point3f> tempPointSet;
		for (int i = 0; i < board_size.height; i++)   //一定从行开始（刚开始没注意写错了，找了半天）
		{
			for (int j = 0; j < board_size.width; j++)
			{
				Point3f realPoint;
				//注意x,y对应关系
				realPoint.x = j * square_size.width;  //注意棋盘格默认坐标系的位置
				realPoint.y = i * square_size.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);  /*添加一幅图像的所有角点坐标*/
			}
		}
		object_points.push_back(tempPointSet);
	}
	/*初始化每幅图像中的角点数量，假定每幅图像中都可以看到完整的标定板*/
	for (int i = 0; i < image_count; i++)
	{
		point_counts.push_back(board_size.width * board_size.height);
	}
	/*calibrate camera*/
	calibrateCamera(object_points, image_point_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);

	cout << "标定完成！\n";

	//对标定结果进行评价  可以写成一个单独的函数
	cout << "开始评价标定结果.........\n";
	double total_err = 0.0;  //所有图像的平均误差的总和
	double err = 0.0;    /*每幅图像的平均误差*/
	vector<Point2f> image_points2; /*保存重新计算得到的投影点*/

	cout << "\t每幅图像的标定误差：\n";

	for (int i = 0; i < image_count; i++)
	{
		vector<Point3f> tempPointSet = object_points[i];
		/*通过得到摄像机内外参数、对空间的三维点进行重新投影计算，得到新的投影点*/
		projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points2);
		/*计算新的投影点和旧的投影点之间的误差*/
		vector<Point2f> tempImagePoint = image_point_seq[i];
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(),CV_32FC2); //存储未矫正角点的坐标值
		Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);  //存储重新投影计算得到的投影角点坐标值

		for (int j = 0; j < tempImagePoint.size(); j++)
		{
			image_points2Mat.at<Vec2f>(0, j) = Vec2f(image_points2[j].x, image_points2[j].y);
			tempImagePointMat.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err = norm(image_points2Mat, tempImagePointMat, NORM_L2); /*矩阵范数运算，用来表征矩阵变化的大小*/
		total_err += (err /= point_counts[i]);
		cout << "第" << i + 1 << "幅图像平均误差： " << err << " 像素" << endl;
	}
	cout << "总体平均误差: " << total_err / image_count << "像素" << endl;
	cout << "评价完成" << endl;

	/*保存标定结果*/
	cout << "开始保存标定结果...." << endl;
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /*保存每幅图像的旋转矩阵*/

	cout << "相机内参数矩阵: " << endl;
	cout << cameraMatrix << endl << endl;

	cout << "畸变系数：\n";
	cout << distCoeffs << endl << endl << endl;

	for (int i = 0; i < image_count; i++)
	{
		cout << "第" << i + 1 << "幅图像的旋转向量：" << endl;
		cout << rvecsMat[i] << endl;
		/*将旋转向量对应的转换为旋转矩阵*/
		Rodrigues(rvecsMat[i], rotation_matrix);
		cout << "第" << i + 1 << "幅图像的旋转矩阵： " << endl;
		cout << rotation_matrix << endl;
		cout << "第" << i + 1 << "幅图像的平移矩阵： " << endl;
		cout << tvecsMat[i] << endl << endl;
	}
	cout << "保存完成" << endl;

	//显示矫正图像
	//Mat src,dst, map1, map2;
	//initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),Mat(), image_size,CV_16SC2,map1,map2); /*计算一次数据，多次使用*/
	//fin.clear();/*清除读到文件尾标志位*/
	//fin.seekg(0, ios::beg);/*定位到文件开头*/

	/*while (getline(fin, filename))
	{
		src = imread(filename, IMREAD_GRAYSCALE);
		remap(src, dst, map1, map2,INTER_LINEAR);
	}*/

	return 0;
} //https://blog.csdn.net/hellohake/article/details/104642687
#include "ORBextractor.h"
using namespace ORB_SLAM2;
// 特征提取
//cv::Ptr<cv::FastFeatureDetector> featureDetector;
//ORBextractor* mpORBextractor;
//featureDetector = cv::FastFeatureDetector::create();
//mpORBextractor = new ORBextractor(2000,1.2,8,20,10);

/*vector<cv::KeyPoint> kps;
    featureDetector->detect(frame(mTrackedRegion), kps);
    sort(kps.begin(), kps.end(), KeyCompartor);
    //cout << "before extract" << endl;
    //cv::Mat descriptors;
    //(*mpORBextractor)(targetRegion, cv::Mat(), kps, descriptors);
    // 选择质量最好在目标范围内的前300个特征点
    for ( auto kp:kps ) {
        if (trackedKeypoints.size() >= 300) {
            break;
        }
        cv::Point2f ptInFrame(kp.pt.x + x, kp.pt.y + y);
        trackedKeypoints.push_back(ptInFrame);
    }*/

    // 2 整帧提取
        /*cv::Mat targetRegion = frame(cv::Rect(x, y, w, h));
        vector<cv::KeyPoint> kps;
        cv::Mat descriptors;
        cout << "before extract" << endl;
        (*mpORBextractor)(frame, cv::Mat(), kps, descriptors);
        cout << "after extract" << endl;

        // 选择质量最好在目标范围内的前300个特征点
        trackedKeypoints.clear();
        sort(kps.begin(), kps.end(), KeyCompartor);
        for ( auto kp:kps ) {
            if (trackedKeypoints.size() >= 300) {
                break;
            }
            if (kp.pt.x >= x && kp.pt.x <= x+w && kp.pt.y >= y && kp.pt.y <= y+h) {
                cv::Point2f ptInFrame(kp.pt.x, kp.pt.y);
                trackedKeypoints.push_back(ptInFrame);
            }
        }*/

    /*vector<cv::KeyPoint> kps;
    featureDetector->detect(targetRegionFrame, kps);
    sort(kps.begin(), kps.end(), KeyCompartor);
    //cv::Mat descriptors;
    //(*mpORBextractor)(targetRegionFrame, cv::Mat(), kps, descriptors);
    // 选择质量最好在目标范围内的前300个特征点
    for ( auto kp:kps ) {
        if (trackedKeypoints.size() >= 300) {
            break;
        }
        cv::Point2f ptInFrame(kp.pt.x + mTrackedRegion.x , kp.pt.y + mTrackedRegion.y);
        //1 homo没有更新，不能用2 warp回去的
        //trackedKeypoints.push_back(ptInFrame);
        //2 当前帧特征点warp到初始帧，判断是否在目标矩形内  ToDo 判断点是否在多边形内
        vector<cv::Point2f> prevPts;
        prevPts.push_back(ptInFrame);
        vector<cv::Point2f> warpPts;
        cv::perspectiveTransform(prevPts, warpPts, mTrackHomo.inv());
        if (warpPts[0].x >= mObjRegion.x && warpPts[0].x <= mObjRegion.x + mObjRegion.width
            && warpPts[0].y >= mObjRegion.y && warpPts[0].y <= mObjRegion.y + mObjRegion.height) {
            trackedKeypoints.push_back(ptInFrame);
        }
    }*/

    // 在整帧上提取特征点。
    /*vector<cv::KeyPoint> kps;
    cv::Mat descriptors;
    (*mpORBextractor)(frame, cv::Mat(), kps, descriptors);
    // 选择质量最好在目标范围内的前300个特征点
    sort(kps.begin(), kps.end(), KeyCompartor);
    for ( auto kp:kps ) {
        if (trackedKeypoints.size() >= 300) {
            break;
        }
        cv::Point2f ptInFrame(kp.pt.x, kp.pt.y);
        // 1 判断是否在trackedRegion
        if (ptInFrame.x >= mTrackedRegion.x && ptInFrame.x <= mTrackedRegion.x + mTrackedRegion.width
            && ptInFrame.y >= mTrackedRegion.y && ptInFrame.y <= mTrackedRegion.y + mTrackedRegion.height) {
            trackedKeypoints.push_back(ptInFrame);
        }
        // 2 当前帧特征点warp到初始帧，判断是否在目标区域内
        vector<cv::Point2f> prevPts;
        prevPts.push_back(ptInFrame);
        vector<cv::Point2f> warpPts;
        cv::perspectiveTransform(prevPts, warpPts, mTrackHomo.inv());
        // 只取区域内的点
        if (warpPts[0].x >= mObjRegion.x && warpPts[0].x <= mObjRegion.x + mObjRegion.width
            && warpPts[0].y >= mObjRegion.y && warpPts[0].y <= mObjRegion.y + mObjRegion.height) {
            trackedKeypoints.push_back(ptInFrame);
        }
    }*/

// 用仿射变换约束过滤外点
/*cv::Mat refineWithAffine(const vector<cv::Point2f>& prevPts, const vector<cv::Point2f>& nextPts,
                             vector<cv::Point2f>& prevRefinePts, vector<cv::Point2f>& nextRefinePts) {
    AffineParamEstimator affineEstimator(2);
    std::vector<double> parameters;
    std::vector<PointPair> inputData;
    for (int i = 0; i < prevPts.size(); ++i) {
        inputData.push_back(PointPair(prevPts[i], nextPts[i]));
    }
    Ransac<PointPair, double>::compute(parameters, &affineEstimator, inputData, 3, 0.995, 0.3);
    // ToDo 返回内点外点

    cv::Mat affine = cv::Mat(parameters);
    return affine.reshape(1, 2).clone();  // 2 * 3
}*/