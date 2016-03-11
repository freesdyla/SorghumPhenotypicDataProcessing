#include "StalkDetection.h"


float rows, cols;
int rowsi, colsi;

Mat e;
Mat widthMap;
Mat widthBinary;


int labelCount=2;
vector<Point> selectedCenterPoints;
vector<Point> finalCenterPoints;
vector<Vec4f> potentialWidth;
vector<Vec4f> finalWidth;
int stage = 0;


void StalkDetection::mouseCallback(int event, int x, int y, int flags, void *param)
{

		StalkDetection *self = static_cast<StalkDetection*>(param);
		self->CallBackFunc(event, x, y, flags, param);
}


float StalkDetection::lineAngle(Vec4f line)
{
	if (fabs(line[0] - line[2]) < 1e-5)
		return 90.0;

	return atanf((line[3] - line[1]) / (line[2] - line[0]))/M_PI*180.0f;
}

float StalkDetection::lineSegLen(Vec4f line)
{
	return sqrtf(powf(line[0] - line[2], 2.0) + powf(line[1] - line[3], 2.0));
}

bool StalkDetection::lineSegValid(Vec4f line)
{
	// too short
	if (lineSegLen(line) < minLen)
		return false;

	if (fabs(lineAngle(line)) > maxAngle)
		return false;

	if (line[0] < winR || line[1] < winR || line[2] < winR || line[3] < winR)
		return false;

	if (line[0] > cols - winR || line[1] > rows - winR || line[2] > cols - winR || line[3] > rows - winR)
		return false;

	return true;
}

Vec2f StalkDetection::unitNormalLine(Vec4f line)
{
	// line coefficients
	float a = line[1] - line[3];
	float b = line[2] - line[0];

	float norm = sqrtf(a*a + b*b);

	Vec2f unitNormal(a/norm, b/norm);

	return unitNormal;
}

Vec3f StalkDetection::lineCoefficients(Vec4f line)
{
	// line coefficients
	float a = line[1] - line[3];
	float b = line[2] - line[0];
	float c = line[0] * line[3] - line[1] * line[2];

	/*if (a < 0.0f)
	{
		a *= -1.f;
		b *= -1.f;
		c *= -1.f;
	}*/

	//float norm = sqrtf(a*a + b*b);
	
	return Vec3f(a, b, c);
}

// return the closest end points from two line segments
Vec2i StalkDetection::closestEndPointsOfTwoLS(Vec4f l1, Vec4f l2)
{
	int i1 = l1[0] > l1[2] ? 0 : 2;

	int i2 = l2[0] < l2[2] ? 0 : 2;

	return Vec2i(i1, i2);
}


// check if l2 is on the right side of l1
bool StalkDetection::checkTwoLineSegsRelative(Vec4f l1, Vec4f l2)
{
	for (int i = 0; i < 3; i += 2)
	{
		for (int j = 0; j < 3; j += 2)
		{
			if (l2[j] < l1[i])
				return false;
		}
	}

	int i1 = l1[0] > l1[2] ? 0 : 2;

	int i2 = l2[0] < l2[2] ? 0 : 2;

	float dist = sqrtf(powf(l1[i1] - l2[i2], 2.0f) + powf(l1[i1 + 1] - l2[i2 + 1], 2.0f));

	if (dist > 300.0f)
		return false;

	return true;
}

Mat StalkDetection::LSD(String path)
{
	Mat img = imread(path, 0);
	Mat imgc;
	cvtColor(img, imgc, CV_BayerBG2BGR);
	cvtColor(imgc, img, CV_BGR2GRAY);

	Mat image;
	equalizeHist(img, image);

	// calculate gradient
	Mat grad;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	int delta = 0;
	int ddepth = CV_16S;

	/// Gradient X
	//Scharr(image, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT);
	Sobel(image, grad_x, ddepth, 1, 0, 3, 1., delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	/// Gradient Y
	//Scharr(image, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);
	Sobel(image, grad_y, ddepth, 0, 1, 3, 1., delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);

	vector<Vec4f> lines_std;
	// Detect the lines
	ls->detect(image, lines_std);

	//remove invalid line segments
	vector<Vec4f> lines;
	for (int i = 0; i < lines_std.size(); i++)
	{
		if (lineSegValid(lines_std[i]))
			lines.push_back(lines_std[i]);
	}

	// Show found lines
	Mat drawnLines(grad);
	ls->drawSegments(drawnLines, lines);

	return drawnLines;
}


// find the min x of each line segment
float StalkDetection::minXLineSegment(Vec4f line)
{
	// p1-x < p2-x
	if (line[0] <= line[2])
		return line[0];
	else
		return line[2];
}

float StalkDetection::pointLineDistance(Vec3f lineCo, float x, float y)
{
	return fabs(lineCo[0] * x + lineCo[1] * y + lineCo[2]) / sqrtf(powf(lineCo[0],2.0f) + powf(lineCo[1],2.0f));
}


// sort according to x in place
void StalkDetection::sortLineSegmentsX(vector<Vec4f>& lines_in)
{
	// find the min x of each line segment
	vector<float> min_xs;
	for (int i = 0; i < lines_in.size(); i++)
		min_xs.push_back(minXLineSegment(lines_in[i]));

	// insertion sort
	for (int i = 1; i < lines_in.size() - 1; i++)
	{
		int j = i;
		while (j>0 && min_xs[j - 1] > min_xs[j])
		{
			//swap x
			float tmp = min_xs[j - 1];
			min_xs[j - 1] = min_xs[j];
			min_xs[j] = tmp;

			// swap line segments
			Vec4f temp(lines_in[j - 1]);
			lines_in[j - 1] = lines_in[j];
			lines_in[j] = temp;

			j--;
		}
	}
}

Mat StalkDetection::equalizeIntensity(const Mat& inputImage)
{
	if (inputImage.channels() >= 3)
	{
		vector<Mat> channels;
		split(inputImage, channels);

		equalizeHist(channels[0], channels[0]);
		equalizeHist(channels[1], channels[1]);
		equalizeHist(channels[2], channels[2]);

		Mat result;
		merge(channels, result);

		return result;
	}

	return Mat();
}

int var1 = 1;
Point pos;

bool StalkDetection::searchNearestWidth(Mat widthMap, Point startPoint, int radius, Point& outPoint)
{
	float distance = 1e10f;
	

	for (int h = -radius; h <= radius; h++)
	{
		for (int w = -radius; w <= radius; w++)
		{
			Point curP(startPoint.x + w, startPoint.y + h);

			if (curP.x < 0 || curP.x >= widthMap.cols || curP.y < 0 || curP.y >= widthMap.rows)
				continue;

			if (widthMap.at<uchar>(curP.y, curP.x) != 0)
			{
				float dis = w*w+h*h;

				if (dis < distance)
				{
					distance = dis;
					outPoint = curP;
				}

			}
		}
	}

	return distance != 1e10f;
}

float StalkDetection::refineWidth(Mat widthMap, Point startPoint, Point& outPoint)
{
	int x = startPoint.x;
	int y = startPoint.y-1;

	vector<int> widthVector;
	widthVector.push_back(widthMap.at<uchar>(startPoint));

	

	// search up
	while (true)
	{
		if (y < 0)
			break;

		uchar width = widthMap.at<uchar>(y, x);

		if (width == 0)
			break;

		widthVector.push_back(width);

		y--;
	}

	int ymin = y;

	//search down
	y = startPoint.y + 1;
	while (true)
	{
		if (y >= widthMap.rows)
			break;

		uchar width = widthMap.at<uchar>(y, x);

		if (width == 0)
			break;

		widthVector.push_back(width);

		y++;
	}

	outPoint.x = x;
	outPoint.y = (int)roundf((ymin + y) / 2.0f);

	std::sort(widthVector.begin(), widthVector.end());

	if (widthVector.size() == 1)
	{
		return widthVector[0];
	}
	else
	{
		int idx = widthVector.size() / 2;

		if (widthVector.size() % 2 == 0)
		{
			return (widthVector[idx] + widthVector[idx - 1])*0.5f;
		}
		else
		{
			return widthVector[idx];
		}
	}
}


void StalkDetection::CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		pos.x = x;
		pos.y = y;
		var1 = 0;

		if (stage == 0)
		{
			float width = (int)widthMap.at<uchar>(y, x) / 10;
			//cout <<"x:"<< x << " y:" << y << " width:"<<width<<endl;

			Point p;

			int winR = 20;

			if (searchNearestWidth(widthMap, pos, winR, p))
			{
				width = (int)widthMap.at<uchar>(p) / 10;
				//cout << "x:" << p.x << " y:" << p.y << " w:" << width << endl;
			}
			else
				cout << "Not found" << endl;

			Point refinedPoint;
			width = refineWidth(widthMap, p, refinedPoint) / 10.f;
			//cout << "x:" << p.x << " y:" << p.y << " refined width:" << width << endl;

			Rect rect;
			floodFill(widthBinary, p, Scalar(labelCount), &rect, cv::Scalar(0), cv::Scalar(0), 8);

			std::vector<Point> CC;

			int minX = 1000000;
			int maxX = -1;

			for (int i = rect.y; i < (rect.y + rect.height); i++)
			{
				for (int j = rect.x; j < (rect.x + rect.width); j++)
				{
					if ((int)widthBinary.at<float>(i, j) != labelCount)
						continue;

					CC.push_back(Point(j, i));

					minX = j < minX ? j : minX;
					maxX = j > maxX ? j : maxX;
				}
			}

			Vec4f line;

			cv::fitLine(CC, line, CV_DIST_L2, 0, 0.01, 0.01);

			Vec2f normal(-line[1], line[0]);

			//float len = (maxX-minX)*0.5f;

			//Point p0((int)(roundf(line[2] + (float)width*normal[0] * 0.5f - len*line[0])), (int)(roundf(line[3] + (float)width*normal[1] * 0.5f - len*line[1])));
			//Point p1((int)(roundf(line[2] + (float)width*normal[0] * 0.5f + len*line[0])), (int)(roundf(line[3] + (float)width*normal[1] * 0.5f + len*line[1])));

			//Point p2((int)(roundf(line[2] - (float)width*normal[0] * 0.5f - len*line[0])), (int)(roundf(line[3] - (float)width*normal[1] * 0.5f - len*line[1])));
			//Point p3((int)(roundf(line[2] - (float)width*normal[0] * 0.5f + len*line[0])), (int)(roundf(line[3] - (float)width*normal[1] * 0.5f + len*line[1])));


			//cv::line(e, p0, p1, Scalar(0, 255, 255));
			//cv::line(e, p2, p3, Scalar(0, 255, 255));

			Point p4((int)(roundf(refinedPoint.x + (float)width*normal[0] * 0.5f)), (int)(roundf(refinedPoint.y + (float)width*normal[1] * 0.5f)));

			Point p5((int)(roundf(refinedPoint.x - (float)width*normal[0] * 0.5f)), (int)(roundf(refinedPoint.y - (float)width*normal[1] * 0.5f)));

			//circle(e, p4, 1, Scalar(0, 255, 255));
			//circle(e, p5, 1, Scalar(0, 255, 255));
			circle(e, refinedPoint, 1, Scalar(0, 255, 255));

			cv::line(e, p4, p5, Scalar(255, 0, 255));

			//circle(e, refinedPoint, (int)(roundf((float)widthMap.at<uchar>(refinedPoint) / 20.f)), Scalar(0, 255, 255));

			selectedCenterPoints.push_back(refinedPoint);

			imshow("width", e);

			labelCount++;
		}
		else if (stage == 1)
		{

		}
	}
	else if (event == EVENT_RBUTTONDOWN)
	{
		pos.x = x;
		pos.y = y;
		var1 = 0;

		if (stage == 0)
		{

			float minDist = 1e10f;
			int idx = -1;

			// find nearest center point
			for (int i = 0; i < selectedCenterPoints.size(); i++)
			{
				float dis = powf(selectedCenterPoints[i].x - pos.x, 2.0f) + powf(selectedCenterPoints[i].y - pos.y, 2.0f);

				if (dis < minDist)
				{
					minDist = dis;
					idx = i;
				}
			}

			if (idx != -1)
			{
				finalCenterPoints.push_back(selectedCenterPoints[idx]);

				circle(e, selectedCenterPoints[idx], (int)roundf((float)widthMap.at<uchar>(selectedCenterPoints[idx]) / 20.f), Scalar(0, 0, 255));

				cout << "x:" << selectedCenterPoints[idx].x << " y:" << selectedCenterPoints[idx].y << " refined width:" << widthMap.at<uchar>(selectedCenterPoints[idx])/10 << endl;

				imshow("width", e);
			}
		}
		else if (stage == 1)
		{
			float minDist = 1e10f;
			int idx = -1;

			// find nearest center point
			for (int i = 0; i < potentialWidth.size(); i++)
			{
				float dis = powf(potentialWidth[i][0] - pos.x, 2.0f) + powf(potentialWidth[i][1] - pos.y, 2.0f);

				if (dis < minDist)
				{
					minDist = dis;
					idx = i;
				}
			}

			if (idx != -1)
			{
				finalWidth.push_back(potentialWidth[idx]);

				Point center((int)roundf(0.5f*(potentialWidth[idx][0] + potentialWidth[idx][2])), (int)roundf(0.5f*(potentialWidth[idx][1] + potentialWidth[idx][3])));

				float width = sqrtf(powf(potentialWidth[idx][0] - potentialWidth[idx][2], 2.0f)+ powf(potentialWidth[idx][1] - potentialWidth[idx][3], 2.0f));

				circle(e, center, (int)roundf(width / 2.f), Scalar(0, 0, 255));

				cout << "x:" << center.x << " y:" << center.y << " width:" << width << endl;

				imshow("width", e);
			}
		}
	}
}

Mat StalkDetection::detectStalkWidth(Mat grad64, int maxWidth, uchar scale)
{
	// integral image
	Mat intImg;
	integral(grad64, intImg, CV_64F);

	Mat haar;
	haar = Mat::zeros(grad64.rows, grad64.cols, CV_64F);

	Mat preHaar;
	preHaar = Mat::zeros(grad64.rows, grad64.cols, CV_64F);


	Mat widthMap;
	widthMap = Mat::zeros(grad64.rows, grad64.cols, CV_8U);

	for (int w = 3; w < maxWidth; w++)
	{
		int winR = w;
		int winR2 = winR - 3;
		int win2R = winR * 2;
		double area = (win2R*2.0 + 1.0)*(winR*2.0 + 1.0);

		for (int y = win2R+50; y < grad64.rows - win2R-50; y++)
		{
			double* pIntUp = intImg.ptr<double>(y - win2R + 1);
			double* pIntDown = intImg.ptr<double>(y + win2R + 1);
			double* pHaar = haar.ptr<double>(y);
			double* pPreHaar = preHaar.ptr<double>(y);
			uchar* pWidth = widthMap.ptr<uchar>(y);

			for (int x = winR+50; x < grad64.cols - winR*50-50; x++)
			{
				//corner id
				/* .........
				| |	 | |
				.........  */
				//1234
				//5678
				double c1 = pIntUp[x - winR + 1];
				double c2 = pIntUp[x - winR2 + 1];
				double c3 = pIntUp[x + winR2 + 1];
				double c4 = pIntUp[x + winR + 1];
				double c5 = pIntDown[x - winR + 1];
				double c6 = pIntDown[x - winR2 + 1];
				double c7 = pIntDown[x + winR2 + 1];
				double c8 = pIntDown[x + winR + 1];

				double haar = (c1 - 2.0*c2 + 2.0*c3 - c4 - c5 + 2.0*c6 - 2.0*c7 + c8 - abs(c1 - c2 - c3 + c4 - c5 + c6 + c7 - c8)) / area;

				if (haar >= pPreHaar[x])
				{
					pPreHaar[x] = haar;
					pWidth[x] = (uchar)(winR * 2 - 1) * scale;
				}
			}
		}

		for (int y = 0; y < widthMap.rows; y++)
		{
			uchar* pWidth = widthMap.ptr<uchar>(y);

			for (int x = 0; x < widthMap.cols; x++)
			{
				if (pWidth[x] == 5*scale)
					pWidth[x] = 0;
			}
		}
	}

	for (int y = 0; y < widthMap.rows; y++)
	{
		uchar* pWidth = widthMap.ptr<uchar>(y);
		double* pPreHaar = preHaar.ptr<double>(y);

		for (int x = 1; x < widthMap.cols-1; x++)
		{
			if (pWidth[x] != 0)
			{
				if (pPreHaar[x] < pPreHaar[x + 1] || pPreHaar[x] < pPreHaar[x - 1])
				{
					pWidth[x] = 0;
				}
			}
				
		}
	}

	return widthMap;
}

int StalkDetection::run(int cameraType, cv::Mat colorImg, double scaleFactor, cv::Mat& markedImg)
{

	labelCount = 2;
	selectedCenterPoints.clear();
	finalCenterPoints.clear();
	potentialWidth.clear();
	finalWidth.clear();
	stage = 0;

	Mat img = colorImg.clone();
	Mat imgc;

	if (cameraType == CANON)
	{
		//img = imread(path1, 1);
		resize(img, img, Size(), 0.25, 0.25);
		imgc = img;
		rows = 3456.f*0.25f;
		cols = 5184.f*0.25f;
		
	}
	else if (cameraType == PG)
	{
		//img = imread(path, 0);
		//cvtColor(img, imgc, CV_BayerBG2BGR);
		imgc = img;
		rows = 1224.0f*scaleFactor;
		cols = 1624.0f*scaleFactor;
	}

	rowsi = (int)rows;
	colsi = (int)cols;

	//imshow("rgb", imgc);

	if (cameraType == CANON)
		e = imgc;
	else if (cameraType == PG)
		e = equalizeIntensity(imgc);

	/*if (NIR)
		imshow("e", imgc);
	else
		cv::imshow("e", e);*/

	cvtColor(imgc, img, CV_BGR2GRAY);
	
	Mat image;

	//if(NIR)
	//	image = img;
	//else
		equalizeHist(img, image);

	//image = img;
	//imshow("histeq", image);


	// calculate gradient
	Mat grad;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	int delta = 0;
	int ddepth = CV_64F;

	/// Gradient X
	//Scharr(image, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT);
	Sobel(image, grad_x, ddepth, 1, 0, 3, 1., delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	/// Gradient Y
	//Scharr(image, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);
	Sobel(image, grad_y, ddepth, 0, 1, 3, 1., delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	

	Mat grad64;

	magnitude(grad_x, grad_y, grad64);


	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	if (cameraType == CANON)
	for (int y = 0; y < e.rows; y++)
	{
		uchar* pE = e.ptr<uchar>(y);
		uchar* pGrad = grad.ptr<uchar>(y);

		for (int x = 0; x < e.cols; x++)
		{
			pE[3*x] -= pGrad[x];
			pE[3*x+1] -= pGrad[x];
			pE[3 * x + 2] -= pGrad[x];
		}
	}

    //imshow("gradient mag", grad);


	//rotate grad64 by 90
	transpose(grad64, grad64);
	flip(grad64, grad64, 0);

	/*double min = 0;
	double max = 0;
	minMaxLoc(grad64, &min, &max);

	grad64 = (grad64 - min) / (max - min) * 255;

	Mat dst;
	convertScaleAbs(grad64, dst);

	imshow("d", dst);
	waitKey(0);*/
	

	//imshow("trans", grad64);
	//waitKey(0);
	
	widthMap = detectStalkWidth(grad64, 16, 10);

	flip(widthMap, widthMap, 0);
	transpose(widthMap, widthMap);

	Mat tmp;
	//resize(widthMap, tmp, Size(), 0.5, 0.5);

	namedWindow("width", 1);

	//create valid width mask and indicate on RGB image
	widthBinary = Mat::zeros(widthMap.rows, widthMap.cols, CV_32F);

	for (int y = 0; y < widthMap.rows; y++)
	{
		uchar* pWidth = widthMap.ptr<uchar>(y);
		float* pWidthB = widthBinary.ptr<float>(y);

		for (int x = 0; x < widthMap.cols; x++)
		{
			if (pWidth[x] != 0)
			{
				pWidthB[x] = 1.0f;
				e.at<Vec3b>(y, x)= Vec3b(255,0,0);
				
			}
		}
	}

	
	cv::imshow("width", e);

	setMouseCallback("width", StalkDetection::mouseCallback, NULL);
	cv::waitKey(0);

	stage = 1;

	// Create and LSD detector with standard or no refinement.
	Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);

	double start = double(getTickCount());
	vector<Vec4f> lines_std;
	// Detect the lines
	ls->detect(image, lines_std);
	double duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
	std::cout << "It took " << duration_ms << " ms." << std::endl;

	vector<Vec4f> lines;

	//remove invalid line segments
	for (int i = 0; i < lines_std.size(); i++)
	{
		if (lineSegValid(lines_std[i]))
			lines.push_back(lines_std[i]);
	}

	cout << lines.size() << " " << lines_std.size() << endl;

	//calculate line angle and distance to principle point
	vector<Line> linesPara;
	//if (0)
	float maxDis = -INF;
	float maxAngle = -INF;
	float minDis = INF;
	float minAngle = INF;

	// Show found lines
	Mat drawnLines(grad);
	vector<Vec4f> empty;
	ls->drawSegments(drawnLines, lines);


	//draw end point
	if(0)
	for (int i = 0; i < lines.size(); i++)
	{
		//putText(drawnLines, String(std::to_string(i)), Point((int)(0.5f*(lines[i][0] + lines[i][2])), (int)(0.5f*(lines[i][1] + lines[i][3]))), cv::HersheyFonts::FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0));
		line(e, Point((int)roundf(lines[i][0]), (int)roundf(lines[i][1])), Point((int)roundf(lines[i][2]), (int)roundf(lines[i][3])), Scalar(0, 0, 255));
		//circle(e, Point((int)roundf(lines[i][0]), (int)roundf(lines[i][1])), 2, Scalar(0, 255, 255));
		//circle(e, Point((int)roundf(lines[i][2]), (int)roundf(lines[i][3])), 2, Scalar(0, 255, 255));
	}


	// detect stalk
	//if (0)
	for (int i = 0; i < lines.size(); i++)
	{
		Vec4f line = lines[i];

		Vec2f normal = unitNormalLine(lines[i]);

		float len = lineSegLen(line);

		// step through the line segment from P1 to P2
		float x1 = line[0];
		float y1 = line[1];
		float x2 = line[2];
		float y2 = line[3];

		if (x1 < 2 * winR || x1>grad.cols - 2 * winR || y1<2 * winR || y1>grad.rows - 2 * winR)
			continue;

		if (x2 < 2 * winR || x2>grad.cols - 2 * winR || y2<2 * winR || y2>grad.rows - 2 * winR)
			continue;

		// line segment direction from p1 to p2
		Vec2f lineDir((x2 - x1) / len, (y2 - y1) / len);

		float max_grad = 0.f;
		int best_offset = 0;
		float best_haar = -1e-10f;


		// search two directions
		for (int j = -1; j <= 1; j += 2)
		{
			float surfaceCost = 0.0f;
			float colorCost = 0.0f;
			float preHaarEdge = 0.0f;
			float preHaarStalk = 0.0f;
			
			// search different width
			for (int i = minWidth; i <= winRInt; i++)
			{
				float curLen = 0;

				float offset = j*i;

				Vec2f preStartPoint(x1 + (offset - j)*normal[0], y1 + (offset - j)*normal[1]);
				Vec2f startPoint(x1 + offset*normal[0], y1 + offset*normal[1]);

				// go through points on the line segment
				float tmp_cost = 0;
				while (curLen <= len)
				{
					// current point on the line segment
					Vec2f curPoint(startPoint[0] + curLen*lineDir[0], startPoint[1] + curLen*lineDir[1]);

					// corresponding point on the previous line segment
					Vec2f curPointPreLine(preStartPoint[0] + curLen*lineDir[0], preStartPoint[1] + curLen*lineDir[1]);

					surfaceCost += (float)grad.at<uchar>((int)roundf(curPoint[1]), (int)roundf(curPoint[0]));
					
					tmp_cost += (float)grad.at<uchar>((int)roundf(curPoint[1]), (int)roundf(curPoint[0]))
								- (float)grad.at<uchar>((int)roundf(curPointPreLine[1]), (int)roundf(curPointPreLine[0]));

					// b - g
					//colorCost += (float)e.at<Vec3b>((int)roundf(curPoint[1]), (int)roundf(curPoint[0])).val[0]
						//	   - (float)e.at<Vec3b>((int)roundf(curPoint[1]), (int)roundf(curPoint[0])).val[1];

					//circle(drawnLines, Point((int)curPoint[0], (int)curPoint[1]), 1, Scalar(0, 0, 255));
					curLen++;
				}

				if (tmp_cost > max_grad && surfaceCost / (float)i / len < 35.0f)// && colorCost < 0.0f)
				{
					//cout << surfaceCost / (float)i / len << " ";
					max_grad = tmp_cost;
					best_offset = offset;
				}
			}
		}

		if (max_grad/len > 35.0f)
		{
			//draw normal
			//Point center((int)((line[0] - line[2])*0.5f + line[2]), (int)((line[1] - line[3])*0.5f + line[3]));
			//Point end((int)(center.x + 15.0f * normal[0]), (int)(center.y + 15.0f * normal[1]));
			//cv::arrowedLine(drawnLines, center, end, Scalar(0, 255, 0));
			cv::line(e, Point((int)line[0], (int)line[1]), Point((int)(line[0] + best_offset*normal[0]), (int)(line[1] + best_offset*normal[1])), Scalar(0, 255, 0), 1);
			cv::line(e, Point((int)(line[0] + best_offset*normal[0]), (int)(line[1] + best_offset*normal[1])),
				Point((int)(line[2] + best_offset*normal[0]), (int)(line[3] + best_offset*normal[1])), Scalar(0, 255, 255));
			//cout << best_haar << " ";
			//putText(drawnLines, String(std::to_string(i)), Point((int)(0.5f*(lines[i][0] + lines[i][2])), (int)(0.5f*(lines[i][1] + lines[i][3]))), cv::HersheyFonts::FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0));
			potentialWidth.push_back(Vec4f(line[0], line[1], line[0] + best_offset*normal[0], line[1] + best_offset*normal[1]));
		}
	}

	// shrink image
	Mat smallerImage;
	//cv::resize(drawnLines, smallerImage, Size(), 0.7, 0.7, CV_INTER_AREA);

	// rotate image
	/*Mat M = getRotationMatrix2D(Point2f(smallerImage.cols / 2.0f, smallerImage.rows / 2.0f), 90, 1);
	Mat result;
	cv::warpAffine(smallerImage, result, M, Size(smallerImage.rows, smallerImage.cols));*/

	//cv::imshow("Standard refinement", drawnLines);
	//imshow("grad", grad);

	imshow("width", e);

	cv::waitKey(0);

	markedImg = e.clone();

	return 0;

}