#ifndef __HEIGHT_DETECTION_H
#define __HEIGHT_DETECTION_H

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <math.h>
#include "Names.h"



class HeightDetection
{
	public:

		HeightDetection();

		static void mouseCallback(int event, int x, int y, int flags, void *param);

		void CallBackFunc(int event, int x, int y, int flags, void* userdata);

		bool selectTwoPointsOnRoot(cv::Mat img, std::vector<cv::Point>& twoPoints);

		bool getSoilPlane(int camera, cv::Mat img, cv::Mat pointCloud, double minZ, double maxZ, cv::Vec6f& soilPlane);

		bool getRootBaseLine(cv::Mat img, const cv::Mat& pointCloud, float minZ, float maxZ, std::vector<cv::Point> twoEndPoints, cv::Vec6f&  rootLine);

};


#endif