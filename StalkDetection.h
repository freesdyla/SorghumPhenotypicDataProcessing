#ifndef __STALK_DETECTION_H
#define __STALK_DETECTION_H

#include "Names.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
//#define _USE_MATH_DEFINES
#include <math.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

class StalkDetection
{
public:

	struct Line
	{
		Vec3f coefficients;
		float angle;
		float distance;
	};

	bool NIR = false;

	const float winR = 30.0;
	const int winRInt = winR;
	const int minWidth = 4;

	const float minLen = 30.0f;
	const float maxAngle = 20.0f;
	const float INF = 1e10f;
	const float maxConnectAngle = 15.0f;


	static void mouseCallback(int event, int x, int y, int flags, void *param);



	float lineAngle(Vec4f line);
	float lineSegLen(Vec4f line);
	bool lineSegValid(Vec4f line);


	Vec2f unitNormalLine(Vec4f line);

	Vec3f lineCoefficients(Vec4f line);

	Vec2i closestEndPointsOfTwoLS(Vec4f l1, Vec4f l2);

	bool checkTwoLineSegsRelative(Vec4f l1, Vec4f l2);

	Mat LSD(String path);

	float minXLineSegment(Vec4f line);

	float pointLineDistance(Vec3f lineCo, float x, float y);

	void sortLineSegmentsX(vector<Vec4f>& lines_in);

	Mat equalizeIntensity(const Mat& inputImage);

	bool searchNearestWidth(Mat widthMap, Point startPoint, int radius, Point& outPoint);

	float refineWidth(Mat widthMap, Point startPoint, Point& outPoint);

	void CallBackFunc(int event, int x, int y, int flags, void* userdata);

	Mat detectStalkWidth(Mat grad64, int maxWidth, uchar scale);

	int run(int cameraType, cv::Mat img, double scaleFactor, cv::Mat& markedImg);

};

#endif
