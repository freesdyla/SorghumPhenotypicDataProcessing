#ifndef __STEREO_MATCHING_H
#define __STEREO_MATCHING_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
//#include <pcl/visualization/cloud_viewer.h>
#include <vector>
#include <iostream>

class StereoMatching
{
public:
	bool checkStereoPairValid(std::vector<cv::Mat>& stereoPair);
	bool censusStereo(std::vector<cv::Mat>& stereoPair, cv::Mat& disparity);
	bool BMStereo(std::vector<cv::Mat>& stereoPair, cv::Mat& disparity);
	bool SGBMStereo(std::vector<cv::Mat>& stereoPair, int numDisp_x16, bool display, cv::Mat& disparity);
	bool scaleStereoPairQMatrix(std::vector<cv::Mat>& stereoPair, const cv::Mat& Q, double scale, std::vector<cv::Mat>& outStereoPair, cv::Mat& outQ);

};


#endif
