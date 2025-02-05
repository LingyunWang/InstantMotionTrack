#include <math.h>
#include "AffineParamEstimator.h"

AffineParamEstimator::AffineParamEstimator(double delta) : m_deltaSquared(delta*delta) {}
/*****************************************************************************/
/*
 * Compute the affine parameters
 */
void AffineParamEstimator::estimate(std::vector<PointPair *> &data, std::vector<double> &parameters) {
	parameters.clear();
	if(data.size()<3)
		return;
    std::vector<cv::Point2f> srcPts;
    std::vector<cv::Point2f> dstPts;
    for (auto ptPair : data) {
        srcPts.push_back(ptPair->from);
        dstPts.push_back(ptPair->to);
    }
    cv::Mat affine = cv::getAffineTransform(srcPts, dstPts);
	parameters = (std::vector<double>)(affine.reshape(1, 1));
	/*for (int i = 0; i < affine.rows; ++i) {
	    for (int j=0; j < affine.cols; ++j) {
	        parameters.push_back(affine.at<double>(i, j));
	    }
	}*/
}
/*****************************************************************************/
/*
 * Compute the affine parameters
 */
void AffineParamEstimator::leastSquaresEstimate(std::vector<PointPair *> &data, std::vector<double> &parameters)
{
    cv::Mat v1 = cv::Mat::ones(3, data.size(), CV_64F);
	cv::Mat v2 = cv::Mat::ones(2, data.size(), CV_64F);
	for (int i = 0; i<data.size(); ++i) {
	    v1.at<double>(0, i) = data[i]->from.x;
	    v1.at<double>(1, i) = data[i]->from.y;
	    v2.at<double>(0, i) = data[i]->to.x;
        v2.at<double>(1, i) = data[i]->to.y;
	}

    cv::Mat hessian = v1 * v1.t();
    cv::Mat m = v2 * v1.t();
    cv::Mat affine = m * hessian.inv();
    parameters = (std::vector<double>)(affine.reshape(1, 1));
}
/*****************************************************************************/
/*
 * Given the line parameters  [n_x,n_y,a_x,a_y] check if
 * [n_x, n_y] dot [data.x-a_x, data.y-a_y] < m_delta
 */
bool AffineParamEstimator::agree(std::vector<double> &parameters, PointPair &data) {
	cv::Mat affine = cv::Mat(parameters);
	affine.reshape(1, 2);  // 2 * 3
	cv::Point2f from = data.from;
	cv::Mat fromPt = cv::Mat::ones(3, 1, CV_64F);
	fromPt.at<double>(0, 0) = from.x;
	fromPt.at<double>(1, 0) = from.y;
	cv::Mat warpPt = affine * fromPt;
	cv::Point2f to = data.to;
	double signedDistance = pow(warpPt.at<double>(0, 0) - to.x, 2) + pow(warpPt.at<double>(1, 0) - to.y, 2);
	return ((signedDistance*signedDistance) < m_deltaSquared);
}
/*****************************************************************************/
void AffineParamEstimator::debugTest(std::ostream &out)
{
	/*std::vector<double> lineParameters;
	LineParamEstimator lpEstimator(0.5);
	std::vector<Point2D *> pointData;

	pointData.push_back(new Point2D(7,7));
	pointData.push_back(new Point2D(-1,-1));
	lpEstimator.estimate(pointData,lineParameters);
	out<<"[n_x,n_y,a_x,a_y] [ "<<lineParameters[0]<<", "<<lineParameters[1]<<", ";
	out<<lineParameters[2]<<", "<<lineParameters[3]<<" ]"<<std::endl;
	lpEstimator.leastSquaresEstimate(pointData,lineParameters);
	out<<"[n_x,n_y,a_x,a_y] [ "<<lineParameters[0]<<", "<<lineParameters[1]<<", ";
	out<<lineParameters[2]<<", "<<lineParameters[3]<<" ]"<<std::endl;
	delete pointData[0];
	delete pointData[1];
	pointData.clear();

	
	pointData.push_back(new Point2D(6,12));
	pointData.push_back(new Point2D(6,6));
	lpEstimator.estimate(pointData,lineParameters);
	out<<"[n_x,n_y,a_x,a_y] [ "<<lineParameters[0]<<", "<<lineParameters[1]<<", ";
	out<<lineParameters[2]<<", "<<lineParameters[3]<<" ]"<<std::endl;
	lpEstimator.leastSquaresEstimate(pointData,lineParameters);
	out<<"[n_x,n_y,a_x,a_y] [ "<<lineParameters[0]<<", "<<lineParameters[1]<<", ";
	out<<lineParameters[2]<<", "<<lineParameters[3]<<" ]"<<std::endl;
	delete pointData[0];
	delete pointData[1];
	pointData.clear();


	pointData.push_back(new Point2D(7,9));
	pointData.push_back(new Point2D(10,9));
	lpEstimator.estimate(pointData,lineParameters);
	out<<"[n_x,n_y,a_x,a_y] [ "<<lineParameters[0]<<", "<<lineParameters[1]<<", ";
	out<<lineParameters[2]<<", "<<lineParameters[3]<<" ]"<<std::endl;
	lpEstimator.leastSquaresEstimate(pointData,lineParameters);
	out<<"[n_x,n_y,a_x,a_y] [ "<<lineParameters[0]<<", "<<lineParameters[1]<<", ";
	out<<lineParameters[2]<<", "<<lineParameters[3]<<" ]"<<std::endl;
	delete pointData[0];
	delete pointData[1];
	pointData.clear();*/
}
