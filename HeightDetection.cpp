#include "HeightDetection.h"

int numPointSelected =0;
std::vector<cv::Point> rootPointVec;


HeightDetection::HeightDetection()
{
	rootPointVec.clear();
}

void HeightDetection::mouseCallback(int event, int x, int y, int flags, void *param)
{

	HeightDetection *self = static_cast<HeightDetection*>(param);
	self->CallBackFunc(event, x, y, flags, param);
}

void HeightDetection::CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == cv::EVENT_LBUTTONDOWN)
	{

		if (numPointSelected < 2)
		{
			numPointSelected++;
			std::cout << "Root point " << std::to_string(numPointSelected)<<" x:" << x << " y : " << y << std::endl;
			rootPointVec.push_back(cv::Point(x, y));

			if (numPointSelected == 2)
				std::cout << "Press space to continue" << std::endl;
			
		}
	}
}


bool HeightDetection::getRootBaseLine(cv::Mat img, const cv::Mat& pointCloud, float minZ, float maxZ, std::vector<cv::Point> twoEndPoints, cv::Vec6f&  rootLine)
{
	cv::Vec4i line(twoEndPoints[0].x, twoEndPoints[0].y, twoEndPoints[1].x, twoEndPoints[1].y);
	
	int largerRangeAxis = abs(line[0]-line[2]) > abs(line[1] - line[3]) ? 0 : 1;
	
	float range = fabs((float)line[largerRangeAxis + 2] - line[largerRangeAxis]);

	std::vector<cv::Point3f> lineSamplePointVec;

	
	if (largerRangeAxis == 0)   // x direction
	{
		float stepY = (line[3] - line[1]) / range;

		int cnt = 0;

		int sign = line[2] - line[0] > 0 ? 1 : -1;

		int x = line[0];

		while (true)
		{
			if (x == line[2])
				break;

			int y = (int)roundf(line[1] + cnt*stepY);

			if (pointCloud.at<cv::Vec3f>(y, x).val[2] >= minZ && pointCloud.at<cv::Vec3f>(y, x).val[2] <= maxZ)
				lineSamplePointVec.push_back(cv::Point3f(pointCloud.at<cv::Vec3f>(y, x)));

			cnt++;

			x += sign;
		}

		/*for (int x = line[0]; x <= line[2]; x++)
		{
			int y = (int)roundf(line[1] + cnt*stepY);

			if (pointCloud.at<cv::Vec3f>(y, x).val[2] >= minZ && pointCloud.at<cv::Vec3f>(y, x).val[2] <= maxZ)
				lineSamplePointVec.push_back(cv::Point3f(pointCloud.at<cv::Vec3f>(y, x)));
			
			cnt++;
		}*/
	}
	else       // y direction	
	{
		float stepX = (line[2] - line[0]) / range;

		int cnt = 0;

		int sign = line[3] - line[1] > 0 ? 1 : -1;

		int y = line[1];

		while (true)
		{
			if (y == line[3])
				break;

			int x = (int)roundf(line[0] + cnt*stepX);

			if (pointCloud.at<cv::Vec3f>(y, x).val[2] >= minZ && pointCloud.at<cv::Vec3f>(y, x).val[2] <= maxZ)
				lineSamplePointVec.push_back(cv::Point3f(pointCloud.at<cv::Vec3f>(y, x)));

			cnt++;

			y += sign;
		}

	/*	for (int y = line[1]; y <= line[3]; y++)
		{
			int x = (int)roundf(line[0] + cnt*stepX);

			if (pointCloud.at<cv::Vec3f>(y, x).val[2]>=minZ && pointCloud.at<cv::Vec3f>(y, x).val[2]<=maxZ)
				lineSamplePointVec.push_back(cv::Point3f(pointCloud.at<cv::Vec3f>(y, x)));


			cnt++;
		}*/
	}


	cv::fitLine(lineSamplePointVec, rootLine, cv::DIST_HUBER, 0, 0.01, 0.01);

	return true;
}

bool HeightDetection::selectTwoPointsOnRoot(cv::Mat img, std::vector<cv::Point>& twoPoints)
{
	if (img.data == NULL)
	{
		std::cout << "Image empty" << std::endl;
		return false;
	}

	numPointSelected = 0;
	cv::Mat rotatedImg;
	cv::transpose(img, rotatedImg);
	cv::flip(rotatedImg, rotatedImg, 0);
	cv::imshow("Select 2 Root Points", rotatedImg);
	cv::setMouseCallback("Select 2 Root Points", HeightDetection::mouseCallback, NULL);

	cv::waitKey(0);

	cv::Vec4f line;

	cv::destroyWindow("Select 2 Root Points");

	cv::line(rotatedImg, rootPointVec[0], rootPointVec[1], cv::Scalar(0, 255, 255), 1);
	
	//cv::imshow("Press space to continue", rotatedImg);

	//cv::waitKey(0);

	//cv::destroyWindow("Press space to continue");

	for (int i = 0; i < rootPointVec.size(); i++)
	{
		int tmp = rootPointVec[i].y;
		rootPointVec[i].y = rootPointVec[i].x;
		rootPointVec[i].x = img.cols - tmp;
	}

	cv::line(img, rootPointVec[0], rootPointVec[1], cv::Scalar(0, 255, 255), 1);

	twoPoints.push_back(rootPointVec[0]);
	twoPoints.push_back(rootPointVec[1]);

	rootPointVec.clear();

	return true;
}

bool HeightDetection::getSoilPlane(int camera, cv::Mat img, cv::Mat pointCloud, double minZ, double maxZ, cv::Vec6f& soilPlane)
{
	if (camera == PG)
	{
		cv::Mat rotatedImg;
		cv::transpose(img, rotatedImg);
		cv::flip(rotatedImg, rotatedImg, 0);
		cv::imshow("Select 2 Root Points", rotatedImg);
		cv::setMouseCallback("Select 2 Root Points", HeightDetection::mouseCallback, NULL);
		cv::waitKey(0);

		cv::Point p1(pointCloud.cols-644, 86);
		cv::Point p2(pointCloud.cols-596, 464);
		cv::Point p0 = (p1 + p2)/2;
		// prevent colinear
		p0.x += 50;

		std::vector<cv::Point> points;
		points.push_back(p1);
		points.push_back(p2);

		cv::Vec6f rootLine;
		getRootBaseLine(img, pointCloud, minZ, maxZ, points, rootLine);


		points[0].x += 45;
		points[1].x += 45;
		
		cv::Vec6f rootLine1;
		getRootBaseLine(img, pointCloud, minZ, maxZ, points, rootLine1);

		cv::Vec3f P1(rootLine[0], rootLine[1], rootLine[2]);
		cv::Vec3f P2(rootLine[3] - rootLine1[3], rootLine[4] - rootLine1[4], rootLine[5] - rootLine1[5]);

		cv::Vec3f normal = (P2).cross(P1);
		
		double len = cv::norm(normal);

		if (len > 1e-10)
		{
			normal /= len;

			soilPlane = rootLine;
			soilPlane[0] = normal[0];
			soilPlane[1] = normal[1];
			soilPlane[2] = normal[2];
		}
		else
		{
			return false;
		}

	}
	else if (camera == CANON)
	{

	}

	return false;
}