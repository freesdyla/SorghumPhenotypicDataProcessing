//unix
#include <unistd.h>
#include <signal.h>
#include <sys/stat.h>

//#define NOMINMAX
//include PCL before OPENCV there is a naming ambiguity
#define INCLUDE_PCL 1
#if INCLUDE_PCL
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_picking_event.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/features/fpfh.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/gp3.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/surface/mls.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ia_ransac.h>
#endif

// read jpeg exif 
#include <exiv2/exiv2.hpp>

#include <fstream>
#include <ctime>
#include <string>
#include "Map2FileMapping.h"
#include "PreprocessingTools.h"
#include "StalkDetection.h"
#include "StereoMatching.h"
#include "HeightDetection.h"
//#include "PatchMatchStereoGPU.h"
#include "Stereo3DMST.h"
#include "PhenoFeatureExtraction.h"

#include <boost/filesystem.hpp>

using namespace boost::filesystem;

#define DIAMETER 0
#define HEIGHT 1
#define SAVE_RESULT 1
#define SIMPLE_SOIL_CUT 1

#if INCLUDE_PCL
//convenient typedefs
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

float pre_x = 0, pre_y = 0, pre_z = 0, Dist = 0;
void pp_callback(const pcl::visualization::PointPickingEvent& event)
{
	float x, y, z;
	event.getPoint(x, y ,z);
	Dist = sqrt( pow(x-pre_x, 2) + pow(y-pre_y, 2) + pow(z-pre_z, 2) );
	Eigen::Vector3f dir(pre_x-x, pre_y-y, pre_z-z);
	dir.normalize();	
	pre_x = x;
	pre_y = y;
	pre_z = z;
	std::cout<<"x:"<<x<<" y:"<<y<<" z:"<<z<<" distance:"<<Dist<<" nx:"<<dir(0)<<" ny:"<<dir(1)<<" nz:"<<dir(2)<<std::endl;
	
}
#endif


ofstream result;
std::string resultPath = "result.csv";
int task = DIAMETER;
int fieldID = CH1;
int rangeID = 2;
int rowID = 3;
int dateID = 0;
int stereoImgNum = 2;
int curRange = rangeID;
int curRow = rowID;
int cameraType = PG;
std::string userName="?";
int curWinRX = 0;
int curWinRY = 0;
int curCentroidX = 0;
int curCentroidY = 0;
int curPlantSide = -1;
int curStereoID = -1;
double curDiameter = 0;

PreprocessTools pt;
double scale_stereo = .5;
double scale1 = 0.8;
double scale_zoom_in = 4.;
double scale_display = 1.;// 0.25;
int winR = 50;

int stalkEdgePickStatus = 0;
cv::Point edge;
const float minZ = 500.f;
const float maxZ = 2000.f;
int plantSide;
std::vector<cv::Point> stalkEdgePoints4;

int maxDisp = 150;
bool measure_complete = false;
int numSubBoxUsed = 20;
double volumeThresh = 0.3;	//sub convex hull volume > 0.2*subAABB volume

int wait_key_time = 100;
int pcl_view_time = 100;

int stereo_method = 0; //0:3dmst	1:sgbm


FileStorage fs;

// euclidean cluster param
int clusterTolerance = 50;
int minClusterSize = 400;

// true if file exists
bool fileExists(const std::string& file) {
    struct stat buf;
    return (stat(file.c_str(), &buf) == 0);
}


// name of data folder and number of images per location
struct dataFolderImgNum
{
	std::string folderName;
	int imgNum;
};

struct pointCloudProcParam
{
	float theta_x[2];
	float theta_y[2];
	float theta_z[2];
	
	float zMin[2];
	float zMax[2];
	float soilHeight[2];

	void setParam(float theta_x0, float theta_x1, float theta_y0, float theta_y1, 
		 float theta_z0, float theta_z1, float zMin0, float zMin1,
		 float zMax0, float zMax1, float height0, float height1)
	{
		theta_x[0] = theta_x0;
		theta_x[1] = theta_x1;
		theta_y[0] = theta_y0;
		theta_y[1] = theta_y1;
		theta_z[0] = theta_z0;
		theta_z[1] = theta_z1;
		zMin[0] = zMin0;
		zMin[1] = zMin1;
		zMax[0] = zMax0;
		zMax[1] = zMax1;
		soilHeight[0] = height0;
		soilHeight[1] = height1;
	}	

};

struct RGBImagePackage
{
	cv::Mat img;
};

struct ImagePackage
{
	cv::Mat origImg;
	cv::Mat disp;
	cv::Mat pointCloud;
	cv::Mat subImg;
	cv::Point anchor;
	std::vector<cv::Mat> origImgVec;
	std::vector<cv::Mat> dispVec;
	std::vector<cv::Mat> pointCloudVec;
	std::vector<cv::Mat> stereoPairs;

	int plantSide;
	cv::Vec6f line;
	cv::Vec6f soilPlane;
	int imgIdx;
	ImagePackage()
	{
		cv::Mat img;
		origImgVec.push_back(img);
		origImgVec.push_back(img);
		origImgVec.push_back(img);
		dispVec.push_back(img);
		dispVec.push_back(img);
		dispVec.push_back(img);
		pointCloudVec.push_back(img);
		pointCloudVec.push_back(img);
		pointCloudVec.push_back(img);
	}
};

std::string getEasternTime()
{
	// current date/time based on current system
	time_t now = time(0);

	struct tm* ltm = localtime(&now);

	std::string time;

	// print various components of tm structure.
	time = std::to_string(1 + ltm->tm_hour) + ":" + std::to_string(1 + ltm->tm_min) + ":"
		+ std::to_string(1 + ltm->tm_sec) + "/" + std::to_string(ltm->tm_mday) + "/"
		+ std::to_string(1 + ltm->tm_mon) + "/" + std::to_string(1900 + ltm->tm_year);

	return time;
}


std::vector<cv::Point> rootLinePoints;
std::vector<cv::Point> preRootLinePoints;
void pickRootLineCallBack(int event, int x, int y, int flags, void* userdata)
{
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		if(rootLinePoints.size() < 2)
		{
			cv::Mat canvas = ((RGBImagePackage*)userdata)->img;
			cv::Point p(x,y);
			cv::circle(canvas, p, 1, cv::Scalar(0, 0, 255), 3);
			
			rootLinePoints.push_back(p);
			//std::cout<<p<<std::endl;

			if(rootLinePoints.size() == 2)
				cv::line(canvas, rootLinePoints[0], rootLinePoints[1], cv::Scalar(0, 0, 255), 1);

			cv::imshow("RGB Image", canvas);
		}
	}
	
	if(event == cv::EVENT_RBUTTONDOWN)
	{
		if(rootLinePoints.size() > 0)
		{
			
			
		} 
	}
}

void voidCallBack(int event, int x, int y, int flags, void* userdata)
{
}

void getCoordinatesCallBack(int event, int x, int y, int flags, void* userdata)
{
	// use right click to remove last line of result
	if (event == cv::EVENT_RBUTTONDOWN)
	{
		if (stalkEdgePoints4.size() > 0 && stalkEdgePoints4.size() < 4)
		{
			ImagePackage *ip = (ImagePackage*)userdata;
			//result.close();
			//result.open(resultPath, ios::app);
			
			if (stalkEdgePoints4.size() == 3)
				stalkEdgePickStatus = 0;

			stalkEdgePoints4.pop_back();
			//edge.y = y;
			//edge.x = (int)round((ip->anchor.x + (winR - y) / scale_zoom_in)*scale_stereo);
			//edge.y = (int)round((ip->anchor.y + (x - winR) / scale_zoom_in)*scale_stereo);
			cv::Mat subImgCopy = ip->subImg.clone();
			for (int i = 0; i < stalkEdgePoints4.size(); i++)
				cv::circle(subImgCopy, stalkEdgePoints4[i], 1, cv::Scalar(0, 0, 255), 2);

			cv::imshow("Pick 4 Points", subImgCopy);
		}

		if (measure_complete)
		{
			result << "Remove above line" << std::endl;
			std::cout << "Last measurement marked wrong" << std::endl;
			measure_complete = false;
		}
	}

	if (event == cv::EVENT_LBUTTONDOWN)
	{
		ImagePackage *ip = (ImagePackage*)userdata;

		std::vector<cv::Mat> stereoPairs = ip->stereoPairs;

		if (task == HEIGHT)
		{
			cv::circle(ip->subImg, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), 2);
			cv::imshow("Pick Point", ip->subImg);

			// depth image size
			int x1 = (int)round((ip->anchor.x + (winR - y/scale_zoom_in))*scale_stereo);
			int y1 = (int)round((ip->anchor.y + (x/scale_zoom_in - winR))*scale_stereo);

			cv::Point3f p0(ip->pointCloudVec[ip->imgIdx].at<cv::Vec3f>(cv::Point(x1,y1)));

			//std::cout <<"x:"<< x1 << " y:" << y1 <<"  P:"<<p0<< std::endl;
			//std::cout << "d:" << ip->dispVec[ip->imgIdx].at<float>(cv::Point(x1, y1)) << std::endl;

			if (p0.z <minZ || p0.z> maxZ)
			{
				//std::cout <<p0.x <<" "<<p0.y<<" "<<p0.z<<" ";
				std::cout<< "Depth not available. Pick anohter point." << std::endl;
				return;
			}

			//transform to bottom camera frame
			if (ip->imgIdx != 0)
			{
				cv::Mat cvPoint(3, 1, CV_64F);
				cvPoint.at<double>(0, 0) = p0.x;
				cvPoint.at<double>(1, 0) = p0.y;
				cvPoint.at<double>(2, 0) = p0.z;

				//transform to the bottom camera view
				for (int k = ip->imgIdx * 2 - 1; k >= 0; k--)
				{
					cv::Mat R;
					cv::transpose(pt.stereoArrayParamVec[5 * ip->plantSide + k]._R, R);
					cvPoint = R*(cvPoint - pt.stereoArrayParamVec[5 * ip->plantSide + k]._T);
				}

				p0.x = cvPoint.at<double>(0, 0);
				p0.y = cvPoint.at<double>(1, 0);
				p0.z = cvPoint.at<double>(2, 0);
			}

			// point to plane 
			cv::Point3f normal(ip->soilPlane[0], ip->soilPlane[1], ip->soilPlane[2]);
			cv::Point3f p3(ip->soilPlane[3], ip->soilPlane[4], ip->soilPlane[5]);
			float height = abs(normal.ddot(p0 - p3));
			
			
			//compute another point on base line
	/*		Point3f p1(ip->line[3] + ip->line[0] * 500.0f,
				ip->line[4] + ip->line[1] * 100.0f,
				ip->line[5] + ip->line[2] * 500.0f);
			Point3f p2(ip->line[3], ip->line[4], ip->line[5]);
			float height = norm((p0 - p1).cross(p0 - p2)) / norm(p2 - p1);*/
			

			std::cout << "Height: " << height<<" mm"<<std::endl;

			int tipx = (int)round((ip->anchor.x + (winR - y/scale_zoom_in)));
			int tipy = (int)round((ip->anchor.y + (x/scale_zoom_in - winR)));

			//field,range,row,task,height,user,time,x1,y1
			//result << fieldID << "," << curRange << "," << curRow << "," << task << "," << height << "," << userName << ","
			//	<< getEasternTime() << "," << tipx << "," << tipy << ","<<ip->imgIdx<<ip->plantSide<<std::endl;

		}
		else if (task == DIAMETER)
		{
			if (stalkEdgePickStatus == 0)
			{
				cv::Point edgePoint(x, y);
				stalkEdgePoints4.push_back(edgePoint);

				if (stalkEdgePoints4.size() == 3)
					stalkEdgePickStatus = 1;
				
				//edge.y = y;
				//edge.x = (int)round((ip->anchor.x + (winR - y) / scale_zoom_in)*scale_stereo);
				//edge.y = (int)round((ip->anchor.y + (x - winR) / scale_zoom_in)*scale_stereo);
				cv::Mat subImgCopy = ip->subImg.clone();
				for (int i = 0; i < stalkEdgePoints4.size(); i++)
					cv::circle(subImgCopy, stalkEdgePoints4[i], 1, cv::Scalar(0, 0, 255), 2);

				cv::imshow("Pick 4 Points", subImgCopy);
			}
			else // clicked 4 points
			{
				//store the 4th point
				stalkEdgePoints4.push_back(cv::Point(x, y));

				// draw the points
				cv::Mat subImgCopy = ip->subImg.clone();
				for (int i = 0; i < stalkEdgePoints4.size(); i++)
					cv::circle(subImgCopy, stalkEdgePoints4[i], 1, cv::Scalar(0, 0, 255), 2);
				cv::imshow("Pick 4 Points", subImgCopy);

				// transform edge points back to origional image coordinates
				for (int i = 0; i < 4; i++)
				{
					int x_copy = stalkEdgePoints4[i].x;
					stalkEdgePoints4[i].x = (int)round(ip->anchor.x + (winR - stalkEdgePoints4[i].y / scale_zoom_in));
					stalkEdgePoints4[i].y = (int)round(ip->anchor.y + (x_copy / scale_zoom_in - winR));
					//std::cout<<"Point:" << i<<" "<<stalkEdgePoints4[i].x<<" "<<stalkEdgePoints4[i].y<<std::endl;
				}

				// compute centroid
				cv::Point centroid(0, 0);
				for (int i = 0; i < 4; i++)
					centroid += stalkEdgePoints4[i];

				centroid.x /= 4;
				centroid.y /= 4;

				//for (int i = 0; i < 4; i++)
					//cv::imwrite("C:\\Users\\lietang\\Desktop\\PhenotypingDataProcessing\\Release\\" + std::to_string(i) + ".png", stereoPairs[i]);

				// sort stalk edge points in origional image coordinates
				// -    -
				// -    -
				// sort by x
				for (int i = 1; i < 4; i++)
				{
					int j = i;
					while (j>0 && stalkEdgePoints4[j - 1].x > stalkEdgePoints4[j].x)
					{
						cv::Point tmp = stalkEdgePoints4[j - 1];
						stalkEdgePoints4[j - 1] = stalkEdgePoints4[j];
						stalkEdgePoints4[j] = tmp;
						j--;
					}
				}

				int winRX = (stalkEdgePoints4[3].x - stalkEdgePoints4[0].x)/2;

				// sort by y in 1st pair and 2nd pair
				for (int i = 0; i < 4; i+=2)
				{
					if (stalkEdgePoints4[i].y > stalkEdgePoints4[i + 1].y)
					{
						cv::Point tmp = stalkEdgePoints4[i];
						stalkEdgePoints4[i] = stalkEdgePoints4[i + 1];
						stalkEdgePoints4[i + 1] = tmp;
					}
				}

				//for (int i = 0; i < 4; i++)
					//std::cout << stalkEdgePoints4[i] << std::endl;
				
				// decide window size
				int winRY = (stalkEdgePoints4[3].y - stalkEdgePoints4[0].y)/2;
				
				// template matching
				if (centroid.y - winRY >= 0 && centroid.y + winRY < stereoPairs[0].rows
					&& centroid.x - winRX >= 0 && centroid.x + winRX < stereoPairs[0].cols)
				{
					double cost;
					double maxNCC = -1;
					int bestDisp = -1;

					// get base template
					cv::Mat base(stereoPairs[ip->imgIdx * 2], cv::Rect(centroid.x-winRX, centroid.y-winRY, 2*winRX+1, 2*winRY+1));
					int w = base.cols * 5;
					int h = base.rows * 5;
					cv::Mat canvas;
					canvas.create(2*h+1, w, CV_8UC3);
					cv::Mat left = canvas(cv::Rect(0, 0, w, h));
					cv::resize(base, left, left.size(), 0, 0, CV_INTER_AREA);
					
					cv::Mat baseGray;
					cv::cvtColor(base, baseGray, CV_BGR2GRAY);
					cv::Mat baseGray_f;
					baseGray.convertTo(baseGray_f, CV_32F);
					baseGray_f -= cv::mean(baseGray_f);
					cv::Mat baseGray2_f; // element wise square
					cv::multiply(baseGray_f, baseGray_f, baseGray2_f);
					cv::Scalar baseDenominatorSum = cv::sum(baseGray2_f);

					// left image as base
					for (int d = 0; d<=maxDisp; d++)
					{
						if (centroid.x - maxDisp - winRX >= 0)
						{
							cv::Mat match(stereoPairs[ip->imgIdx * 2 + 1], cv::Rect(centroid.x - d - winRX, centroid.y - winRY, 2 * winRX + 1, 2 * winRY + 1));

							cv::Mat matchGray;
							cv::cvtColor(match, matchGray, CV_BGR2GRAY);
							cv::Mat matchGray_f;
							matchGray.convertTo(matchGray_f, CV_32F);
							matchGray_f -= cv::mean(matchGray_f);
							cv::Mat matchGray2_f;
							cv::multiply(matchGray_f, matchGray_f, matchGray2_f);

							cv::Mat numerator;
							cv::multiply(baseGray_f, matchGray_f, numerator);
							cv::Scalar sum_numerator = cv::sum(numerator);
							cv::Scalar matchDenominatorSum = cv::sum(matchGray2_f);
							cv::Scalar denominator;
							cv::sqrt(baseDenominatorSum*matchDenominatorSum, denominator);
							cv::Scalar ncc = sum_numerator / denominator;
							//std::cout << "disp:" << d << " cost" << ncc << std::endl;

							if (ncc[0] > maxNCC)
							{
								maxNCC = ncc[0];
								bestDisp = d;
							}
						}
					}

					if (bestDisp != -1)
					{
						// show matching result
						cv::Mat match(stereoPairs[ip->imgIdx * 2 + 1], cv::Rect(centroid.x - bestDisp - winRX, centroid.y - winRY, 2 * winRX + 1, 2 * winRY + 1));
						cv::Mat right = canvas(cv::Rect(0, h+1, w, h));
						cv::resize(match, right, right.size(), 0, 0, CV_INTER_AREA);
						cv::line(canvas, cv::Point(0, h), cv::Point(w, h), cv::Scalar(0, 0, 255));
						cv::Mat rotCanvas;
						cv::transpose(canvas, rotCanvas);
						cv::flip(rotCanvas, rotCanvas, 0);
						cv::imshow("Base-Match", rotCanvas);
						cv::moveWindow("Base-Match", 50, 50);
						std::cout << "NCC score:" << maxNCC << std::endl;

						if (maxNCC > 0.5)
						{
							double diameterPixel = 0;

							for (int i = 0; i < 2; i++)
							{
								for (int j = 0; j < 4; j += 2)
								{
									cv::Point2f v_base(stalkEdgePoints4[i] - stalkEdgePoints4[i + 2]);
									cv::Point2f v_side(stalkEdgePoints4[i] - stalkEdgePoints4[(j + 1 + i) % 4]);
									diameterPixel += sqrt(v_side.dot(v_side) - powf(v_base.dot(v_side) / cv::norm(v_base), 2));
								}
							}

							diameterPixel /= 4;

							double baseline = 1.0 / pt.stereoArrayParamVec[ip->plantSide * 5 + 2 * ip->imgIdx]._Q.at<double>(3, 2);
							double diameter = diameterPixel / bestDisp*baseline;

							curCentroidX = centroid.x;
							curCentroidY = centroid.y;
							curWinRX = winRX;
							curWinRY = winRY;
							curDiameter = diameter;
							curStereoID = ip->imgIdx;

							std::cout << "diameter:" << diameter << " mm " /*<< curPlantSide*/<<std::endl;

							//field,range,row,task,stalk diameter,user,time,x,y,w,h,stereo id,side
							/*result << fieldID << "," << curRange << "," << curRow << "," << task << "," << curDiameter << "," << userName << ","
								<< getEasternTime() << "," << curCentroidX << "," << curCentroidY << "," << curWinRX << "," << curWinRY << "," << curStereoID << "," << curPlantSide << std::endl;*/

							measure_complete = true;
						}
						else
							std::cout << "No good match" << std::endl;
					}
					else
						std::cout<<"Depth not available, pick another location!"<<std::endl;
				}
				else
					std::cout<<"Too close to border, pick another location!"<<std::endl;
			
				stalkEdgePickStatus = 0;
				stalkEdgePoints4.clear();
			}
		}
	}
}


// locate where to zoom in
void locateSubImageCallBack(int event, int x, int y, int flags, void* userdata)
{
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		ImagePackage *ip = (ImagePackage*)userdata;

		stalkEdgePoints4.clear();

		//std::cout << "x:" << x << " y:" << y << std::endl;
		int imgIdx = 0;

		if (x >= ip->origImg.rows*scale_stereo*scale_display)
			imgIdx = 1;
		if (x >= ip->origImg.rows*scale_stereo*scale_display*2)
			imgIdx = 2;

		cv::Mat origImg = ip->origImg;
		cv::Mat disp = ip->disp;

		x = (int)(x - imgIdx*ip->origImg.rows*scale_stereo*scale_display);

		//scale to origional size
		x = (int)(x / scale_display / scale_stereo);
		y = (int)(y / scale_display / scale_stereo);

		// rotate to origional orientation
		int tmp = y;
		y = x;
		x = origImg.cols-tmp;

		if (x - winR >= 0 && y - winR >= 0 && x + winR <= origImg.cols && y + winR <= origImg.rows)
		{
			//x y in origional image
			cv::Mat subImg(ip->origImgVec[imgIdx], cv::Rect(x - winR, y - winR, 2 * winR, 2 * winR));

			cv::Mat out;
			cv::transpose(subImg, out);
			cv::flip(out, out, 0);
			cv::resize(out, out, cv::Size(), scale_zoom_in, scale_zoom_in, CV_INTER_CUBIC);
			cv::imshow("Pick 4 Points", out);
			cv::moveWindow("Pick 4 Points", 600, 300);

			/*cv::Mat subDisp;
			if (task == DIAMETER)
			{
				cv::Mat subDisp0(disp, cv::Rect((int)round((x - winR)*scale_stereo), (int)round((y - winR)*scale_stereo), (int)round(2 * winR*scale_stereo), (int)round(2 * winR*scale_stereo)));
				subDisp = subDisp0;
			}
			else if (task == HEIGHT)
			{
				cv::Mat subDisp0(ip->dispVec[imgIdx], cv::Rect((int)round((x - winR)*scale_stereo), (int)round((y - winR)*scale_stereo), (int)round(2 * winR*scale_stereo), (int)round(2 * winR*scale_stereo)));
				subDisp = subDisp0;
			}*/
			
		/*	cv::Mat subDisp(ip->dispVec[imgIdx], cv::Rect((int)round((x - winR)*scale_stereo), (int)round((y - winR)*scale_stereo), (int)round(2 * winR*scale_stereo), (int)round(2 * winR*scale_stereo)));

			cv::Mat subDisp8;
			subDisp.convertTo(subDisp8, CV_8U);
			cv::Mat out1;
			cv::transpose(subDisp8, out1);
			cv::flip(out1, out1, 0);
			cv::resize(out1, out1, cv::Size(), scale_zoom_in/scale_stereo, scale_zoom_in/scale_stereo, CV_INTER_CUBIC);
			cv::imshow("Depth", out1*3);*/
			((ImagePackage*)userdata)->anchor = cv::Point(x, y);
			((ImagePackage*)userdata)->subImg = out;
			((ImagePackage*)userdata)->imgIdx = imgIdx;
			cv::setMouseCallback("Pick 4 Points", getCoordinatesCallBack, userdata);
			stalkEdgePickStatus = 0;
		}
	}
}

void exit_handler(int s)
{
	 cv::destroyAllWindows();
	 result.close();
	 fs.release();
         printf("ctrl+c signal");
         exit(1); 
}



int main(int argc , char** argv)
{

	if(argc != 6)
	{
		std::cout<<"range, row, date, field, plant base detect mode\n";
		std::cout<<"Field ID: AH-0, CH1-1, CH2-2, AS-3, CSP-4, BURKEY-5, KELLY-6, A2016-7"<<'\n';
		return 0;
	}

	fs.open("parameters.yml", FileStorage::READ);

	fs["wait_key_time"] >> wait_key_time;

	pt.view_time = wait_key_time;

	fs["pcl_view_time"] >> pcl_view_time;
	
	fs["stereo_method"] >> stereo_method;	//0:3dmst	1:sgbm

	Map2FileMapping M;

	int plant_base_detect_mode = 0;

	rangeID = atoi(argv[1]);
	rowID = atoi(argv[2]);
	dateID = atoi(argv[3]);
	fieldID = atoi(argv[4]);
	plant_base_detect_mode = atoi(argv[5]); 
		//std::cout<<"range ID "<< rangeID<<std::endl;

	/*double t = (double)getTickCount();

	std::vector<std::string> timeVec;
	std::string prePath = "Burkey-07-21/";

	pt.loadAllNameTime(prePath);

	std::vector<cv::Mat> stereoPairs;

	pt.loadCanonStereoPairs("Right_Re_0_Ro_14_Ra_84", prePath, 1, stereoPairs);



	cv::waitKey(0);
	
	return 0;*/

	signal(SIGINT, exit_handler);
	
	boost::shared_ptr<pcl::visualization::PCLVisualizer> pcViewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	pcViewer->registerPointPickingCallback(&pp_callback);
	pcViewer->addCoordinateSystem (400.0);
	pcViewer->setSize(1100,700);
	
	std::map<std::string, double> calib_correct_lookup_2016;

	/*-------------------------------------------------------------------------------------------*/

	StereoMatching sm;

	if(fieldID == A2016) {
	
		if(dateID == 0) {
		
			pt.loadAllStereoArrayParam(CANON, "CameraParam2016_1");
			
			calib_correct_lookup_2016["28.5"] = 30;
			calib_correct_lookup_2016["29.5"] = 30;
			calib_correct_lookup_2016["30.5"] = 40;
			calib_correct_lookup_2016["31.5"] = 25;
			calib_correct_lookup_2016["32.5"] = 25;
			calib_correct_lookup_2016["33.5"] = 25;
			calib_correct_lookup_2016["34.5"] = 10;
			calib_correct_lookup_2016["35.5"] = 25;
			calib_correct_lookup_2016["36.5"] = 20;
			calib_correct_lookup_2016["37.5"] = 32;
			calib_correct_lookup_2016["38.5"] = 36;
			calib_correct_lookup_2016["39.5"] = 15;
			calib_correct_lookup_2016["28.7"] = 30;
			calib_correct_lookup_2016["29.7"] = 25;
			calib_correct_lookup_2016["30.7"] = 30;
			calib_correct_lookup_2016["31.7"] = 30;
			calib_correct_lookup_2016["32.7"] = 40;
			calib_correct_lookup_2016["33.7"] = 40;
			calib_correct_lookup_2016["34.7"] = 20;
			calib_correct_lookup_2016["35.7"] = 15;
			calib_correct_lookup_2016["36.7"] = 15;
			calib_correct_lookup_2016["37.7"] = 45;
			calib_correct_lookup_2016["38.7"] = 35;
			calib_correct_lookup_2016["39.7"] = 35;
		}
		else if(dateID == 1) {
		
			pt.loadAllStereoArrayParam(CANON, "CameraParam2016_2");
		}

		cameraType = CANON;
		scale_stereo = 0.2;//15;
		

	}
	else if(fieldID == KELLY || fieldID == BURKEY) 	{
	
		pt.loadAllStereoArrayParam(CANON, "CameraParam2015");
		cameraType = CANON;
		scale_stereo = 0.2;//15;
	}
	else {
	
		pt.loadAllStereoArrayParam(PG, "CameraParam2014");
		cameraType= PG;
	}
	
	// fix calibration error, shift in y direction
	// lm, rm ok for both fields
	if(fieldID == AH)
	{
		// looking at right side btm stereo
		if(dateID < 2)
		{	
			pt.stereoArrayParamVec[0]._M2.at<double>(1, 2) += 4.;
		}
		

		// left btm stereo
		if(dateID < 4)
		{
			pt.stereoArrayParamVec[5]._M2.at<double>(1, 2) += 6.;
		}

		// left top stereo
		pt.stereoArrayParamVec[9]._M2.at<double>(1, 2) += 5.;

		// right top ok

	}
	else if(fieldID == CH1 || fieldID == CH2)
	{
		// looking at right side btm stereo
		if(dateID < 2)
		{
			pt.stereoArrayParamVec[0]._M2.at<double>(1, 2) += 5.;
		}	

		if(dateID < 4)
		{
			pt.stereoArrayParamVec[5]._M2.at<double>(1, 2) += 5.;
		}

		// left btm ok

		// right top
		if(dateID == 5)
			pt.stereoArrayParamVec[4]._M2.at<double>(1, 2) -= 5.;

		if(dateID == 6)
			pt.stereoArrayParamVec[4]._M2.at<double>(1, 2) -= 2.;


		//left top
		if(dateID == 5)
			pt.stereoArrayParamVec[9]._M2.at<double>(1, 2) += 2.;
		if(dateID == 6)
			pt.stereoArrayParamVec[9]._M2.at<double>(1, 2) += 4.;

	}
	else if(fieldID == A2016)
	{
		double offset;
	
		if(dateID == 0)	{
		
			fs["offset_A16_0"] >> offset;
			pt.stereoArrayParamVec[0]._M2.at<double>(1, 2) += offset;	//+ move right image up
		}
		else {
		
			fs["offset_A16_1_b"] >> offset;
			pt.stereoArrayParamVec[0]._M2.at<double>(1, 2) += offset;	//+ move right image up

			fs["offset_A16_1_m"] >> offset;
			pt.stereoArrayParamVec[2]._M2.at<double>(1, 2) += offset;	//+ move right image up
		}
	}

	// load	all data folders
	std::vector<dataFolderImgNum> AHHedgeFolderImgVec;
	dataFolderImgNum dfin;

	dfin.folderName = "AgronomyHedge07182014";	//date ID 0
	dfin.imgNum = 2;
	AHHedgeFolderImgVec.push_back(dfin);

	dfin.folderName = "AgronomyHedge07242014";	//1
	dfin.imgNum = 4;
	AHHedgeFolderImgVec.push_back(dfin);

	dfin.folderName = "AgronomyHedge07302014";	//2
	dfin.imgNum = 4;
	AHHedgeFolderImgVec.push_back(dfin);

	dfin.folderName = "AgronomyHedge08132014";	//3
	dfin.imgNum = 6;
	AHHedgeFolderImgVec.push_back(dfin);

	dfin.folderName = "AgronomyHedge08252014";	//4
	dfin.imgNum = 6;
	AHHedgeFolderImgVec.push_back(dfin);


	std::vector<dataFolderImgNum> CHHedgeFolderImgVec;
	dfin.folderName = "CurtissHedge07202014";	//0
	dfin.imgNum = 2;
	CHHedgeFolderImgVec.push_back(dfin);

	dfin.folderName = "CurtissHedge07232014";	//1
	dfin.imgNum = 2;
	CHHedgeFolderImgVec.push_back(dfin);

	dfin.folderName = "CurtissHedge07312014";	//2
	dfin.imgNum = 2;
	CHHedgeFolderImgVec.push_back(dfin);

	dfin.folderName = "CurtissHedge08042014";	//3
	dfin.imgNum = 2;
	CHHedgeFolderImgVec.push_back(dfin);

	dfin.folderName = "CurtissHedge08142014";	//4
	dfin.imgNum = 4;
	CHHedgeFolderImgVec.push_back(dfin);

	dfin.folderName = "CurtissHedge08192014";	//5
	dfin.imgNum = 6;
	CHHedgeFolderImgVec.push_back(dfin);

	dfin.folderName = "CurtissHedge09032014";	//6
	dfin.imgNum = 6;
	CHHedgeFolderImgVec.push_back(dfin);

	std::vector<dataFolderImgNum> BHHedgeFolderImgVec;
	dfin.folderName = "Burkey-07-21";
	dfin.imgNum = 2;
	BHHedgeFolderImgVec.push_back(dfin);

	dfin.folderName = "Burkey-07-31";
	dfin.imgNum = 4;
	BHHedgeFolderImgVec.push_back(dfin);


	std::vector<dataFolderImgNum> A2016FolderImgVec;
	dfin.folderName = "KneeHigh";
	dfin.imgNum = 2;
	A2016FolderImgVec.push_back(dfin);

	dfin.folderName = "Mature";
	dfin.imgNum = 4;
	A2016FolderImgVec.push_back(dfin);


	std::string dataPath;
	pointCloudProcParam pc_p_param;

	std::vector<dataFolderImgNum> HedgeFolderImgVec;

	if (fieldID == AH)
	{
		if(dateID<0 || dateID>AHHedgeFolderImgVec.size())
		{
			std::cout<<"Date ID wrong"<<std::endl;
			return -1;
		}

		dataPath = AHHedgeFolderImgVec[dateID].folderName;
		stereoImgNum = AHHedgeFolderImgVec[dateID].imgNum;

		// theta_x, theta_y, theta_z, zmin, zmax, height
		pc_p_param.setParam(-M_PI/36, -M_PI/36, 
				   -M_PI/9, -M_PI/9, 
				   -M_PI/90, M_PI/90,
				   800, 750,
				   1600, 1600,
				   -500, -540);	// height

		HedgeFolderImgVec = AHHedgeFolderImgVec;
	}
	else if(fieldID == CH1 || fieldID == CH2)
	{
		if(dateID<0 || dateID>=CHHedgeFolderImgVec.size())
		{
			std::cout<<"Date ID wrong"<<std::endl;
			return -1;
		}

		dataPath =  CHHedgeFolderImgVec[dateID].folderName;
		stereoImgNum = CHHedgeFolderImgVec[dateID].imgNum;

		pc_p_param.setParam(-M_PI/36, -M_PI/36, 
				   -M_PI/9, -M_PI/9, 
				   -M_PI/90, M_PI/90,
				   800, 800,
				   1600, 1600,
				   -500, -540);

		HedgeFolderImgVec = CHHedgeFolderImgVec;
	}
	else if(fieldID == BURKEY)
	{
		if(dateID >= BHHedgeFolderImgVec.size() || dateID < 0 )
		{
			std::cout<<"Date ID wrong"<<std::endl;
			return 0;
		}

		pt.loadAllNameTime(BHHedgeFolderImgVec[dateID].folderName+"/");

		int nextDateID = dateID==BHHedgeFolderImgVec.size()-1 ? dateID-1 : dateID+1;
		pt.loadAllNextNameTime(BHHedgeFolderImgVec[nextDateID].folderName+"/");
		
		pc_p_param.setParam(-M_PI/72, -M_PI/72, 
				   -M_PI/18, -M_PI/18, 
				   M_PI/2, M_PI/2,
				   500, 500,
				   1800, 1800,
				   -500, -540);


		HedgeFolderImgVec = BHHedgeFolderImgVec;

		stereoImgNum = A2016FolderImgVec[dateID].imgNum;

		
	}
	else if(fieldID == KELLY)
	{

	}
	else if(fieldID == A2016)
	{
		if(dateID >= A2016FolderImgVec.size() || dateID < 0 )
		{
			std::cout<<"Date ID wrong"<<std::endl;
			return 0;
		}

		pt.loadAllNameTime(A2016FolderImgVec[dateID].folderName+"/");

		int nextDateID = dateID==A2016FolderImgVec.size()-1 ? dateID-1 : dateID+1;
		pt.loadAllNextNameTime(A2016FolderImgVec[nextDateID].folderName+"/");
		
		pc_p_param.setParam(0, 0, 
				   0, 0, 
				   M_PI/2, M_PI/2,
				   500, 500,
				   1500, 1500,
				   -500, -540);

		HedgeFolderImgVec = A2016FolderImgVec;

		stereoImgNum = A2016FolderImgVec[dateID].imgNum;
	}

	std::vector<std::string> imgNameVec;

	std::vector<std::vector<int> > plantLocationVec;

	//append
	result.open(resultPath, ios::app);

	std::cout<<"result file open: "<<result.is_open()<<std::endl;

	result << std::endl << "field(AH-0;CH1-1;CH2-2),range,row,plant height(mm),hedge width(mm),volume(m^3),VVI,leaf area(m^2),VAI,Centroid2HeightRatio,projectOccupancyAlongRow,NumValidSubBox,numSubBox,time,slice volume(m^3)";

	for(int i=1; i<=numSubBoxUsed; i++) result<<",sub volume " <<i<<" (m^3)";

	for(int i=1; i<=numSubBoxUsed; i++) result<<",sub area " <<i<<" (m^2)";

	result<<",low width(mm),middle width(mm),high width(mm)";

	result << std::endl;


	//M.getRandomImages(AH, REP_1, SHORT_PLANT, 7, plantLocationVec);
	//M.getRandomImages(AH, REP_2, SHORT_PLANT, 7, plantLocationVec);
	//M.getRandomImages(AH, REP_1, TALL_PLANT, 7, plantLocationVec);
	//M.getRandomImages(AH, REP_2, TALL_PLANT, 7, plantLocationVec);
	//M.getRandomImages(AH, REP_1, PS_PLANT, 7, plantLocationVec);
	//M.getRandomImages(AH, REP_2, PS_PLANT, 7, plantLocationVec);


	// previous root line 
	pcl::PointXYZ preRootPoint;
	//if(fieldID == AH || fieldID == CH1 || fieldID == CH2)
	{
		preRootPoint.x = -559.56;
		preRootPoint.y = 523.443;
		preRootPoint.z = 1400.96;
	}

	pcl::PointXYZ preRootLineDir(0, 1, 0);

	// initialize root line end points in the RGB Image window
	preRootLinePoints.clear();
	preRootLinePoints.push_back(cv::Point(22, 589));
	preRootLinePoints.push_back(cv::Point(487, 562));

	Eigen::Matrix4f preTemplateTransform;
	if(cameraType == PG)
	preTemplateTransform << 0.997418f, -0.01673f, 0.0699239f, -559.904f,
				0.0113719f, 0.997023f, 0.0763345f, -11.8371f,
				-0.0709946f, -0.0753417f, 0.994631f, 1326.08f,
			        0.f, 0.f, 0.f, 1.f;
	else if(cameraType == CANON)
	{

		preTemplateTransform << 1.f, 0.f, 0.f, -270.f,
					0.f, 1.f, 0.f, 900.f,
					0.f, 0.f, 1.f, 1400.0f,
					0.f, 0.f, 0.f, 1.f;
	}

	// right angle template
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_baseTemplate(new pcl::PointCloud<pcl::PointXYZRGB>);
	int stepSize = 20;
	int templateHeight = 200;
	int templateWidth = 1000;


	for(int z=0; z<templateHeight; z+=stepSize)
	{
		for(int y=0; y<templateWidth; y+=stepSize)
		{
			pcl::PointXYZRGB point;
			point.x = 0.f;
			point.y = (float)y;
			point.z = -1.0f*(float)z;
			point.r = 0;
			point.g = 255;
			point.b = 0;
			pc_baseTemplate->push_back(point);
		}
	}

	for(int x=0; x<templateHeight; x+=stepSize)
	{
		for(int y=0; y<templateWidth; y+=stepSize)
		{
			pcl::PointXYZRGB point;
			point.x = (float)x;
			point.y = (float)y;
			point.z = 0.f;
			point.r = 0;
			point.g = 255;
			point.b = 0;
			pc_baseTemplate->push_back(point);
		}
	}

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_originBaseTemplate(new pcl::PointCloud<pcl::PointXYZRGB>);

	*pc_originBaseTemplate += *pc_baseTemplate;

	if(cameraType == CANON)
	{
		// -90 deg rotation 
		//Eigen::Affine3f transform(Eigen::AngleAxisf(-M_PI/2, Eigen::Vector3f::UnitZ()));	
		//pcl::transformPointCloud(*pc_baseTemplate, *pc_baseTemplate, transform);
	}
	

	float tmp_x, tmp_y, tmp_z;
	fs["template_A16_init_x"] >> tmp_x;
	fs["template_A16_init_y"] >> tmp_y;	
	fs["template_A16_init_z"] >> tmp_z;		

	// translation
	Eigen::Matrix4f baseTemplateInitTransform = Eigen::Matrix4f::Identity();
	baseTemplateInitTransform(0,3) = cameraType==PG ? -570.f : tmp_x;//-270.f;
	baseTemplateInitTransform(1,3) = cameraType==PG ? 0.f : tmp_y;//900.f;
	baseTemplateInitTransform(2,3) = cameraType==PG ? 1260.f : tmp_z;//1400.f;	
	pcl::transformPointCloud(*pc_baseTemplate, *pc_baseTemplate, baseTemplateInitTransform);

	StalkDetection sd;
	HeightDetection hd;
	PhenoFeatureExtraction pfe;

	int rangeStart = rangeID;
	int rangeEnd = M.fieldConfigVec[fieldID].rangeEnd;
	int rowStart = rowID;
	int rowEnd = M.fieldConfigVec[fieldID].rowEnd;
	bool EXIT = false;

	fs.release();

	int counter = 0;
	for (int row = rowStart; row <= rowEnd; row++)
	{
		if(EXIT)
			break;
		curRow = row;

		//std::cout << "row " << row << endl;
		for (int range = M.fieldConfigVec[fieldID].rangeStart; range <= rangeEnd; range++)
		{
			if (counter == 0)
				range = rangeID;

			curRange = range;

#if 0
			//skip PS type
			if (!M.checkRangeRowInBoundForType(fieldID, SHORT_PLANT, range, row) &&
				!M.checkRangeRowInBoundForType(fieldID, TALL_PLANT, range, row))
			{
				std::cout << "PS type" << std::endl;
				counter++;
				continue;
			}
#endif
				
			counter++;

			plantSide = -1;
			//int range = plantLocationVec[i][0];
			//int row = plantLocationVec[i][1];		
			int passID = -1;
			if (!M.getFileName(fieldID, range, row, imgNameVec, plantSide, passID))
			{
				std::cout << "File name not found range " << range << " row " << row << std::endl;
				continue;
			}
			
			for(auto name : imgNameVec) cout<<name<<"\n";
			
			std::vector<cv::Mat> imgVec;
			
			if(cameraType == PG)
			{
				if (!pt.loadPGStereoPairs(PG, plantSide, stereoImgNum, HedgeFolderImgVec[dateID].folderName+"/"+imgNameVec[0], imgVec))
				{
					std::cout <<std::endl<< "Missing pg stereo image for range "<<range<<" row "<<row<<std::endl;
					continue;
				}
			}
			else if(cameraType == CANON)
			{
				if (!pt.loadCanonStereoPairs(imgNameVec[0], HedgeFolderImgVec[dateID].folderName+"/", passID, imgVec))
				{
					std::cout <<std::endl<< "Missing canon stereo image for range "<<range<<" row "<<row<<std::endl;	
					continue;
				}
			}

		
			int num_valid_images = 0;

			for( auto & img : imgVec ) {

				if(!img.empty()) ++num_valid_images;
			}

			if( !(num_valid_images == 2 || num_valid_images == 4 || num_valid_images == 6) ) {

				std::cout <<std::endl<< "Missing stereo image for range "<<range<<" row "<<row<<std::endl;
				continue;
			}	

			std::vector<cv::Mat> cvNextDateImgVec;

			int nextDateID = dateID == HedgeFolderImgVec.size()-1 ? dateID-1 : dateID+1;

			if ( cameraType==PG && nextDateID >= 0 && 
			     !pt.loadPGStereoPairs(PG, plantSide, HedgeFolderImgVec[nextDateID].imgNum, HedgeFolderImgVec[nextDateID].folderName+"/"+imgNameVec[0], cvNextDateImgVec))
			{
				std::cout <<std::endl<< "Missing Next DATE stereo image for range "<<range<<" row "<<row<<std::endl;
			}

			if( cameraType==CANON && nextDateID >=0 &&
			    !pt.loadCanonStereoPairsNextDate(imgNameVec[0], HedgeFolderImgVec[nextDateID].folderName+"/", passID, cvNextDateImgVec))
			{
				std::cout <<std::endl<< "Missing Next DATE stereo image for range "<<range<<" row "<<row<<std::endl;
			}

			//std::cout << imgNameVec[0];
			curPlantSide = plantSide;

			//continue;
			/*-----------------------------------------------------------------------*/

			std::vector<cv::Mat> cvPointCloudVec;
			std::vector<cv::Mat> cvColorImgVec;
			std::vector<cv::Mat> cvNextDateColorImgVec;
			std::vector<cv::Mat> cvOrigionalImgVec;
			std::vector<cv::Mat> dispVec;
			ImagePackage ip;

			//double t = (double)getTickCount();
			std::cout<<std::endl<<"Range:"<<curRange<<" Row:"<<curRow<<std::endl;		

#if 0			
			if(fieldID == A2016) {
			
				if(dateID == 0) {
				
					std::string key = std::to_string(curRange)+"."+std::to_string(curRow);
					//+ move right image up
					pt.stereoArrayParamVec[0]._M2.at<double>(1, 2) += calib_correct_lookup_2016.find(key)->second;	
				}
			}
#endif

			std::vector<cv::Mat> stereoPairs;
	
			for (int j = 0; j < imgVec.size()/2; j++)
			{			
				int stereoParamIdx = 5 * plantSide + 2 * j;		

				// rectify image
				std::vector<cv::Mat> RectifyStereoPairVec;
				pt.rectifyStereoPair(PG, stereoParamIdx, imgVec[2 * j], imgVec[2 * j + 1], RectifyStereoPairVec, true, false, scale_stereo);
				cvOrigionalImgVec.push_back(RectifyStereoPairVec[0]);


				// prepare stereo pair, scale image and Q matrix
				std::vector<cv::Mat> stereoPair;
				cv::Mat Q, disp;
				sm.scaleStereoPairQMatrix(RectifyStereoPairVec, pt.stereoArrayParamVec[stereoParamIdx]._Q, scale_stereo, stereoPair, Q);
				stereoPairs.push_back(stereoPair[0]);
				stereoPairs.push_back(stereoPair[1]);

				if(cameraType == PG)
					cvColorImgVec.push_back(/*stereoPair[0]*/pt.equalizeIntensity(stereoPair[0]));
				else if(cameraType == CANON)
					cvColorImgVec.push_back(stereoPair[0]);
				//cvColorImgVec.push_back(stereoPair[0]);


				// next date image
				if(cvNextDateImgVec.size() != 0 && cvNextDateImgVec.size() / 2 != 0 && j < HedgeFolderImgVec[nextDateID].imgNum/2) 
				{
					std::vector<cv::Mat> tmpPair;
					pt.rectifyStereoPair(PG, stereoParamIdx, cvNextDateImgVec[2 * j], cvNextDateImgVec[2 * j + 1], tmpPair, false, false, scale_stereo);

					std::vector<cv::Mat> nextDateStereoPair;
					sm.scaleStereoPairQMatrix(tmpPair, pt.stereoArrayParamVec[stereoParamIdx]._Q, scale_stereo, nextDateStereoPair, Q);
					if(cameraType == PG)
						cvNextDateColorImgVec.push_back(nextDateStereoPair[0]/*pt.equalizeIntensity(nextDateStereoPair[0])*/);
					else if(cameraType == CANON)
						cvNextDateColorImgVec.push_back(nextDateStereoPair[0]);
				}

				std::string saved_disp_file_name = HedgeFolderImgVec[dateID].folderName+"/cloud/"
									+ "fi_"+std::to_string(fieldID)+"_da_"+std::to_string(dateID)
									+ "_ro_"+std::to_string(row) + "_ra_"+std::to_string(range)
									+ "_di_"+std::to_string(j) + "_wi_" + std::to_string(stereoPair[0].cols)
									+ "_hi_" + std::to_string(stereoPair[0].rows) + ".bin";

				bool saved_disp_file_exists = false;
				
#if 1
				// stereo matching and reproject
				if( !stereoPair[0].empty() && !stereoPair[1].empty() ) {

					if(stereo_method != 0 || !fileExists(saved_disp_file_name)) {
					
						double t = (double)cv::getTickCount();
						// PatchMatchStereo GPU
						cv::Mat cvRightDisp_f;
						// left, right, winRad, minD, maxD, iteration, scale, showDisp, left disp, right disp
						//PatchMatchStereoHuberGPU(stereoPair[0], stereoPair[1], 5, 0, 96, 15, 3.0, true, disp, cvRightDisp_f);
						//PatchMatchStereoGPU(stereoPair[0], stereoPair[1], 5, 0, 100, 10, 3.0, true, disp, cvRightDisp_f);

	
						if(stereo_method == 0) {
							// save stereo image pair to mccnn folder
							std::vector<int> compression_params;
							compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
							compression_params.push_back(9);
							//imwrite("/home/lietang/mc-cnn-master/l.png", stereoPair[0], compression_params);
							//imwrite("/home/lietang/mc-cnn-master/r.png", stereoPair[1], compression_params);
							imwrite("l.png", stereoPair[0], compression_params);
							imwrite("r.png", stereoPair[1], compression_params);


							std::string data_cost = "MCCNN_acrt"; 
							std::string smoothness_prior = "NL2TGV";
							std::string left_name = "";
							std::string right_name = "";
							//PatchMatchStereoNL2TGV(stereoPair[0], stereoPair[1], 50, 0, 120, 80, 1.0, false, disp, cvRightDisp_f, 
							//		       data_cost, smoothness_prior, left_name, right_name);
							stereo3dmst("l.png", "r.png", stereoPair[0], stereoPair[1], disp, cvRightDisp_f, data_cost, 120);
			
							// change back to project working directory
							//int status = chdir("/media/lietang/SSD/PhenotypingDataProcessing");
						}
						else {	
							// 2nd argument is the number of dispairy, will be x16
							sm.SGBMStereo(stereoPair, 7, true, disp);
						}

						std::cout <<"stereo time:"<< ((double)cv::getTickCount() - t) / cv::getTickFrequency() << std::endl;

						// remove steel bar and tire
						if(cameraType==PG && plantSide == 0 && j == 0) {
							// wheel big chunk
							//disp(cv::Range(1040/2, disp.rows), cv::Range(0, 610/2)) = 0;
							// wheel upper part
							//disp(cv::Range(1116/2, disp.rows), cv::Range(610/2, 732/2)) = 0;
							//disp(cv::Range((1116-20)/2,1116/2), cv::Range(610/2, 650/2)) = 0;
							// bar
							//disp(cv::Range(0, disp.rows), cv::Range(0, 250/2)) = 0;

							//curtiss 0903
							disp(cv::Range(500, disp.rows), cv::Range(0, 400)) = 0;
							disp(cv::Range(470, 500), cv::Range(0, 350)) = 0;
						}			
				
						// remove border
						int borderWidth = 50;
						disp(cv::Range(0, borderWidth), cv::Range(0, disp.cols)) = 0.f;
						disp(cv::Range(disp.rows-borderWidth, disp.rows), cv::Range(0, disp.cols)) = 0.f; 					
				
						if (cameraType==PG && j==0) {
						
							if (plantSide == 0)
								disp(cv::Range(0, disp.rows), cv::Range(0, 170)) = 0;
							else if (plantSide == 1)
								disp(cv::Range(0, disp.rows), cv::Range(0, 120)) = 0;
						}

	#if 1
						// remove sky
						if(cameraType == PG)
						{
							//make a mask of blue sky, convert rgb to lab
							cv::Mat hsv; 				
							cv::cvtColor(stereoPair[0], hsv, CV_BGR2HSV);
							std::vector<cv::Mat> channels;
							cv::split(hsv, channels);
							//cv::imshow("HSV", channels[0]);
							//cv::waitKey(0);
			
							for(int y=0; y<stereoPair[0].rows; y++)
							{
								for(int x=0; x<stereoPair[0].cols; x++)
								{
									uchar hue = channels[0].at<uchar>(y,x);
									if(hue > 85 && hue < 105)
									{
										//(cvColorImgVec[j].at<Vec3b>(y,x)).val[0] = 255;
										//(cvColorImgVec[j].at<Vec3b>(y,x)).val[1] = 255;
										//(cvColorImgVec[j].at<Vec3b>(y,x)).val[2] = 255;
										disp.at<float>(y, x) = 0.f;
									}
								}
							}

							// remove disparity of saturated pixel
							for(int y=0; y<stereoPair[0].rows; y++)
							{
								for(int x=0; x<stereoPair[0].cols; x++)
								{
									cv::Vec3b rgb = stereoPair[0].at<Vec3b>(y,x);
									if(rgb[0]==255 && rgb[1]==255 && rgb[2]==255)
									{
										disp.at<float>(y, x) = 0.f;
									}
								}
							}
						}
						else if(cameraType == CANON)
						{
							for(int y=0; y<stereoPair[0].rows; y++)
							{
								for(int x=0; x<stereoPair[0].cols; x++)
								{
							
									int b = (int)((cvColorImgVec[j].at<Vec3b>(y,x)).val[0]);
									int g = (int)((cvColorImgVec[j].at<Vec3b>(y,x)).val[1]);
									int r = (int)((cvColorImgVec[j].at<Vec3b>(y,x)).val[2]);

									if(abs(g-b) > 50 || b == 255)
									{
										(cvColorImgVec[j].at<Vec3b>(y,x)).val[0] = 255;
										(cvColorImgVec[j].at<Vec3b>(y,x)).val[1] = 0;
										(cvColorImgVec[j].at<Vec3b>(y,x)).val[2] = 0;
										disp.at<float>(y, x) = 0.f;
									}
								}
							}
						}
				
	#endif
						saved_disp_file_exists = false;
					}
					else {
					
						saved_disp_file_exists = true;

						const int rows = stereoPair[0].rows;
						const int cols = stereoPair[0].cols;
						disp.create(rows, cols, CV_32F);
						std::ifstream infile(saved_disp_file_name, std::ifstream::binary);
						infile.read((char*)disp.ptr<float>(0), sizeof(float)*rows*cols);
					}

#if 1
					if(stereo_method == 0 && !saved_disp_file_exists) {
					
						ofstream my_file(saved_disp_file_name, std::ofstream::binary);
						my_file.write((char*)disp.ptr<float>(0), sizeof(float)*disp.cols*disp.rows);
						my_file.close();
					}
#endif

// display disparity map
#if 1
					cv::Mat disp8;
					disp.convertTo(disp8, CV_8U);
					imshow("disp"+std::to_string(j), disp8*3);
					//imshow("left img", stereoPair[0]);
					//imshow("right img", stereoPair[1]);
					cv::waitKey(wait_key_time);
					//cv:destroyWindow("left img");
					//cv:destroyWindow("right img");
#endif
				}


				dispVec.push_back(disp);
				cv::Mat cvPointCloud;
				//f32 point cloud
				if( !disp.empty() )
					cv::reprojectImageTo3D(disp, cvPointCloud, Q, true);
				
				cvPointCloudVec.push_back(cvPointCloud);
				
#endif
			}

#if 0
			//reset
			if(fieldID == A2016) {
			
				if(dateID == 0) {
				
					std::string key = std::to_string(curRange)+"."+std::to_string(curRow);
					//+ move right image up
					pt.stereoArrayParamVec[0]._M2.at<double>(1, 2) -= calib_correct_lookup_2016.find(key)->second;	
				}
			}
#endif

			//continue;
			
			//std::cout <<"stereo time:"<< ((double)getTickCount() - t) / getTickFrequency() << std::endl;

			// root line points
			pcl::PointCloud<pcl::PointXYZ>::Ptr pc_rootLinePoints(new pcl::PointCloud<pcl::PointXYZ>);

			// show rgb image
			if (cvColorImgVec.size() != 0)
			{
				cv::Mat canvas;

				for (int k = 0; k < cvColorImgVec.size(); k++)
				{
					if( cvColorImgVec[k].empty() ) continue;

					cv::Mat shrinked;
					cv::resize(cvColorImgVec[k], shrinked, cv::Size(), scale_display, scale_display, CV_INTER_AREA);
					
					if (cameraType == PG)
					{
						cv::transpose(shrinked, shrinked);
						cv::flip(shrinked, shrinked, 0);
					}

					if ( canvas.empty() )
					{	
						canvas.create(shrinked.rows, shrinked.cols * 3, CV_8UC3);
						canvas = 0;
					}

					shrinked.copyTo(canvas(cv::Rect(k*shrinked.cols, 0, shrinked.cols, shrinked.rows)));

					if(k < cvColorImgVec.size()-1)
						cv::line(canvas, cv::Point(shrinked.cols*(k+1)-1,0), cv::Point(shrinked.cols*(k+1)-1,shrinked.rows-1), cv::Scalar(0,0,255));
				}

				cv::Mat nextDateCanvas;

				for (int k = 0; k < cvNextDateColorImgVec.size(); k++)
				{
					if( cvNextDateColorImgVec[k].empty() ) continue;

					cv::Mat shrinked;
					cv::resize(cvNextDateColorImgVec[k], shrinked, cv::Size(), scale_display, scale_display, CV_INTER_AREA);
					
					if (cameraType == PG)
					{
						cv::transpose(shrinked, shrinked);
						cv::flip(shrinked, shrinked, 0);
					}

					if ( nextDateCanvas.empty() )
					{
						nextDateCanvas.create(shrinked.rows, shrinked.cols * 3, CV_8UC3);
						nextDateCanvas = 0;
					}

					shrinked.copyTo(nextDateCanvas(cv::Rect(k*shrinked.cols, 0, shrinked.cols, shrinked.rows)));

					if(k < cvNextDateColorImgVec.size()-1)
						cv::line(nextDateCanvas, cv::Point(shrinked.cols*(k+1)-1,0), cv::Point(shrinked.cols*(k+1)-1,shrinked.rows-1), cv::Scalar(0,0,255));
				}


//user pick root line (this only works for pointgrey data now)
#if 1	
				if(plant_base_detect_mode == 0)
				{
					cv::line(canvas, preRootLinePoints[0], preRootLinePoints[1], cv::Scalar(0, 255, 255), 1);

					cv::imshow("RGB Image", canvas);
					cv::moveWindow("RGB Image", 100, 100);

					RGBImagePackage imgPkg;
					imgPkg.img = canvas;
					cv::setMouseCallback("RGB Image", pickRootLineCallBack, (void*)&imgPkg);
					rootLinePoints.clear();	

					bool NEW_ROOT_LINE = false;

					while(true)
					{
						int key = cv::waitKey(0);

						//std::cout<<key<<std::endl;

						// key = q
						if(key == 1048689)
							break;

						// key = o, overwrite root line end points
						if(key == 1048687)
						{
							if(rootLinePoints.size() == 2)
							{
								NEW_ROOT_LINE = true;				
								break;
							}
							else
								std::cout<<"overwrite root line fail-"<<rootLinePoints.size()<<std::endl;
						}

						// key = e, exit
						if(key == 1048677)
						{
							std::cout<<"EXIT..."<<std::endl;
							EXIT = true;
							break;
						}
					}

					if(EXIT) break;

					// transform points back to origional image coordinate
					cv::Point tmpRootLinePoints[2];
					if(cameraType == PG)
					{
						if(NEW_ROOT_LINE)
						{
							for(int i=0; i<2; i++)
							{
								int tmp = rootLinePoints[i].y;
								tmpRootLinePoints[i].y = rootLinePoints[i].x;
								tmpRootLinePoints[i].x = canvas.rows-1-tmp;
							}
						}
						else
						{ 			
							for(int i=0; i<2; i++)
							{
								int tmp = preRootLinePoints[i].y;
								tmpRootLinePoints[i].y = preRootLinePoints[i].x;
								tmpRootLinePoints[i].x = canvas.rows-1-tmp;
							}
						}
					}
					else if(cameraType == CANON) 							
					{
						if(NEW_ROOT_LINE)
						{
							for(int i=0; i<2; i++)
							{
								tmpRootLinePoints[i].y = rootLinePoints[i].y;
								tmpRootLinePoints[i].x = rootLinePoints[i].x;
							}
						}
						else
						{ 			
							for(int i=0; i<2; i++)
							{
								tmpRootLinePoints[i].y = preRootLinePoints[i].y;
								tmpRootLinePoints[i].x = preRootLinePoints[i].x;
							}
						}
					}


					// compute 2D line equation
					cv::Point2f p0(tmpRootLinePoints[0].x/scale_display, tmpRootLinePoints[0].y/scale_display);
					cv::Point2f p1(tmpRootLinePoints[1].x/scale_display, tmpRootLinePoints[1].y/scale_display);
					cv::Point2f pt = p0-p1;		

					std::cout<<p0<<"\n"<<p1<<"\n";

					if(cameraType == PG)
					{
						if( pt.y != 0.f )
						{										
							// sample root line points and remove points below root line
							for(int y=0; y<cvPointCloudVec[0].rows; y++)
							{
								// given y, compute x on the line
								int x = (int)((y-p0.y)/pt.y*pt.x + p0.x);
								pcl::PointXYZ point;
								point.x = cvPointCloudVec[0].at<cv::Vec3f>(y, x).val[0];
								point.y = cvPointCloudVec[0].at<cv::Vec3f>(y, x).val[1];
								point.z = cvPointCloudVec[0].at<cv::Vec3f>(y, x).val[2];
								pc_rootLinePoints->push_back(point);
								dispVec[0](cv::Rect(0, y, x, 1)) = 0;
								cvPointCloudVec[0](cv::Rect(0, y, x, 1)) = 0;
							}				
						}
						else	// slope 90 deg
						{
							// given y, compute x on the line
							int x = (int)(p0.x);

							for(int y=0; y<cvPointCloudVec[0].rows; y++)
							{			
								pcl::PointXYZ point;
								point.x = cvPointCloudVec[0].at<cv::Vec3f>(y, x).val[0];
								point.y = cvPointCloudVec[0].at<cv::Vec3f>(y, x).val[1];
								point.z = cvPointCloudVec[0].at<cv::Vec3f>(y, x).val[2];
								pc_rootLinePoints->push_back(point);							
							}

							dispVec[0](cv::Rect(0, 0, x, dispVec[0].rows)) = 0;
							cvPointCloudVec[0](cv::Rect(0, 0, x, cvPointCloudVec[0].rows)) = 0;
						}
					}
					else if(cameraType == CANON)
					{
						// sample root line points and remove points below root line
						for(int x=0; x<cvPointCloudVec[0].cols; x++)
						{
							// given x, compute y on the line
							int y = (int)((x-p0.x)/pt.x*pt.y + p0.y);
							pcl::PointXYZ point;
							point.x = cvPointCloudVec[0].at<cv::Vec3f>(y, x).val[0];
							point.y = cvPointCloudVec[0].at<cv::Vec3f>(y, x).val[1];
							point.z = cvPointCloudVec[0].at<cv::Vec3f>(y, x).val[2];
							pc_rootLinePoints->push_back(point);
							dispVec[0](cv::Rect(x, y, 1, dispVec[0].rows-y)) = 0;
							cvPointCloudVec[0](cv::Rect(x, y, 1, dispVec[0].rows-y)) = 0;	
						}

					}

					cv::destroyWindow("RGB Image");
				}
#endif

				if(!nextDateCanvas.empty())
					cv::imshow("Next Date RGB Images", nextDateCanvas);

				cv::imshow("RGB Images", canvas);
				cv::moveWindow("RGB Images", 100,100);
				
				cv::waitKey(wait_key_time);

				//continue;

// show disparity after removing soil
#if 0
				cv::Mat disp8;
				dispVec[0].convertTo(disp8, CV_8U);	

				cv::Mat disp8_;
				if(cameraType == PG)
				{
					cv::transpose(disp8, disp8_);
					cv::flip(disp8_, disp8_, 0);		
				}
				else 
					disp8_ = disp8;

				imshow("disp", disp8_);
				cv::waitKey(0);
#endif

				/*if (task == DIAMETER)
				{
					cv::imshow("Stalk Diameter", canvas);
					cv::moveWindow("Stalk Diameter", 0, 0);
					
					ip.origImg = pt.equalizeIntensity(cvOrigionalImgVec[0]);

					for (int i = 0; i < cvOrigionalImgVec.size(); i++)
					{
						ip.origImgVec[i] = pt.equalizeIntensity(cvOrigionalImgVec[i]);
					//	ip.dispVec[i] = dispVec[i];
					//	ip.pointCloudVec[i] = cvPointCloudVec[i];
					}

					//ip.pointCloud = cvPointCloudVec[0];
					//ip.disp = dispVec[0];
					ip.plantSide = plantSide;
					cv::setMouseCallback("Stalk Diameter", locateSubImageCallBack, (void*)&ip);
					std::cout << "Process range:" << range << " row:" << row << std::endl;
					
					while (true)
					{
						int key = cv::waitKey(0);
						//std::cout << key << std::endl;

						// key = j, let user enter range row
						if (key == 74 || key == 106)
						{
							std::cout << "Jump to range:";

							cin.clear();
							fflush(stdin);
							std::cin >> rangeID;
							//cin.ignore();

							if (rangeID < M.fieldConfigVec[fieldID].rangeStart || rangeID > M.fieldConfigVec[fieldID].rangeEnd)
							{
								std::cout << "range not valid for this field" << std::endl;
								break;
							}

							std::cout << "Enter row:";

							cin.clear();
							fflush(stdin);
							std::cin >> rowID;
							//cin.ignore();

							if (rowID < M.fieldConfigVec[fieldID].rowStart || rowID > M.fieldConfigVec[fieldID].rowEnd)
							{
								std::cout << "range not valid for this field" << std::endl;
								break;
							}

							range = rangeID - 1;
							row = rowID;
							break;
						}

						// key == right, go to next range
						if (key == 2555904)
							break;
						
						// key = s
						if (key == 115 || key == 83)
						{
							std::cout << "Current display scale:"<<scale_display<<" Change display scale to (0.25-0.5):";
							cin.clear();
							fflush(stdin);
							double tmp_scale;
							std::cin >> tmp_scale;
							if (tmp_scale >= 0.25 && tmp_scale <= 0.5)
							{
								scale_display = tmp_scale;
								std::cout << "Next range scale will be " << scale_display << std::endl;
							}
							else
							{
								std::cout << "Scale change failed." << std::endl;
							}
						}
					}
				}*/
			}
			else
				std::cout<<"Image empty range "<<curRange<<" row "<<curRow<<std::endl;	

			//continue;

#if INCLUDE_PCL
			pcViewer->removeAllPointClouds(0);
			pcViewer->removeAllShapes(0);

			double t1 = (double)getTickCount();


			std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pcl_pcVec;
			
			// move point cloud from opencv to pcl format + prefilter by depth
			for (int j = 0; j < cvPointCloudVec.size(); j++)
			{
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp_pc(new pcl::PointCloud<pcl::PointXYZRGB>());

				if( !cvPointCloudVec[j].empty() )
				{
					cv::Mat pc = cvPointCloudVec[j];
					cv::Mat bgr = cvColorImgVec[j];			

					for (int y = 0; y < pc.rows; y++)
					{
						float* p_pc = pc.ptr<float>(y);
						uchar* p_bgr = bgr.ptr<uchar>(y);

						for (int x = 0; x < pc.cols; x++)
						{
							float z = p_pc[3*x+2];

							if (z>minZ && z < maxZ)
							{							
								pcl::PointXYZRGB p;
								p.x = p_pc[3 * x];
								p.y = p_pc[3 * x + 1];
								p.z = z;

								/*p.b = p_bgr[3 * x];
								p.g = p_bgr[3 * x + 1];
								p.r = p_bgr[3 * x + 2];*/

								p.b = 0;
								p.g = j*125;
								p.r = 255;

								tmp_pc->push_back(p);											
							}		
						}
					}
				}
				
				pcl_pcVec.push_back(tmp_pc);

				//cout<<j<<" size "<<tmp_pc->size()<<"\n";
			}

			//std::cout << "copy to pcl:" << ((double)getTickCount() - t1) / getTickFrequency() << std::endl;

#if 0
			for(int i=0; i<pcl_pcVec.size(); i++) 
			{
				pcViewer->addPointCloud(pcl_pcVec[i], "pc"+std::to_string(i), 0);
				pcViewer->spin();
				pcViewer->removeAllPointClouds(0);
			}
#endif

			// outlier removal
			pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sorf;
			sorf.setMeanK(50);
			sorf.setStddevMulThresh(1.0);

			pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_plantOutliers(new pcl::PointCloud<pcl::PointXYZRGB>());
			
			t1 = (double)getTickCount();
			for(int i=0; i<pcl_pcVec.size(); i++)
			{
				// down sample 
				pcl::VoxelGrid<pcl::PointXYZRGB> vg;
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_down(new pcl::PointCloud<pcl::PointXYZRGB>);
				vg.setInputCloud(pcl_pcVec[i]);
				vg.setLeafSize(10.f, 10.f, 10.f);
				vg.filter(*pc_down);

				pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZRGB>());
				sorf.setInputCloud(pc_down);
				sorf.filter(*tmp);

				pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmpOutliers(new pcl::PointCloud<pcl::PointXYZRGB>());

				// Euclidean cluster, remove small clusters
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_plant(new pcl::PointCloud<pcl::PointXYZRGB>);

				pfe.smallClusterRemoval(tmp, clusterTolerance, minClusterSize, pc_plant);		
				
				Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();

				if(cameraType == PG)
				for (int k = i * 2 - 1; k >= 0; k--) {					
				
					cv::Mat translation = pt.stereoArrayParamVec[5 * plantSide + k]._T;
					
					cv::Mat rotation = pt.stereoArrayParamVec[5 * plantSide + k]._R;
					
					Eigen::Matrix4f transMat = Eigen::Matrix4f::Identity();

					for (int y = 0; y < 3; y++)						
						for (int x = 0; x < 3; x++)									
							transMat(y, x) = rotation.at<double>(y, x);
												
					for(int y=0; y<3; y++)
						transMat(y, 3) = translation.at<double>(y, 0);

					transform = transform*transMat;					
				}
				else if(cameraType == CANON && i==1) {
				
					cv::Mat translation = pt.stereoArrayParamVec[1]._T;
					
					cv::Mat rotation = pt.stereoArrayParamVec[1]._R;
					for (int y = 0; y < 3; y++)						
						for (int x = 0; x < 3; x++)									
							transform(y, x) = rotation.at<double>(y, x);
												
					for(int y=0; y<3; y++)
						transform(y, 3) = translation.at<double>(y, 0);
				}


				Eigen::Matrix4f inverse = transform.inverse();
				pcl_pcVec[i]->clear();
				pcl::transformPointCloud(*pc_plant, *pcl_pcVec[i], inverse);		
				std::cout<<"cloud size "<<i<<":"<<pcl_pcVec[i]->points.size()<<std::endl;
				
				// not enough points, remove it			
				if(pcl_pcVec[i]->points.size()<4000)
					pcl_pcVec[i]->clear();
				else //if(0)	
				{			
					tmp->clear();
					// roughly align plant row direction with y axis
					Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
					transform_2.rotate(Eigen::AngleAxisf(/*-M_PI/36*/pc_p_param.theta_y[plantSide], Eigen::Vector3f::UnitX()));
					
					// rotate around y to aligh plant growth direction with x
					Eigen::Affine3f transform_3(Eigen::AngleAxisf(/*M_PI/36*/pc_p_param.theta_x[plantSide], Eigen::Vector3f::UnitY()));

					// rotate around Z axis
					Eigen::Affine3f transform_4(Eigen::AngleAxisf(/*0*/pc_p_param.theta_z[plantSide], Eigen::Vector3f::UnitZ()));

					if(cameraType == PG)
						transform_2 = transform_2*transform_3*transform_4;
					else
						transform_2 = transform_4*transform_3*transform_2;

					//if(cameraType == CANON)
					//	transform_2.translation() << 450.f, 300.f, 0.f;

					//Eigen::Vector4f btmCentroid;
					//pcl::compute3DCentroid(*pcl_pcVec[0], btmCentroid);

					std::cout<<transform_2.matrix()<<'\n';

					pcl::transformPointCloud (*pcl_pcVec[i], *tmp, transform_2);
					pcl_pcVec[i]->clear();
					
					// further crop in z direction
					pcl::PassThrough<pcl::PointXYZRGB> pass;
					pass.setInputCloud(tmp);
					pass.setFilterFieldName("z");
					pass.setFilterLimits(pc_p_param.zMin[plantSide], pc_p_param.zMax[plantSide]);
					pass.setFilterLimitsNegative(false);
					pass.filter(*pcl_pcVec[i]);
					tmpOutliers->clear();
					pass.setFilterLimitsNegative(true);
					pass.filter(*tmpOutliers);
					*pc_plantOutliers += *tmpOutliers;

					if(pc_rootLinePoints->size()!=0 && i==0)
					{
						// transform root line points
						pcl::PassThrough<pcl::PointXYZ> pass_;
						pass_.setFilterFieldName("z");
						pass_.setFilterLimits(pc_p_param.zMin[plantSide], pc_p_param.zMax[plantSide]);
						pass_.setFilterLimitsNegative(false);
					
						pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_(new pcl::PointCloud<pcl::PointXYZ>);
						pcl::transformPointCloud (*pc_rootLinePoints, *tmp_, transform_2);
						pass_.setInputCloud(tmp_);
						pass_.filter(*pc_rootLinePoints);
						//std::cout<<"root line point size:"<<pc_rootLinePoints->points.size()<<std::endl;
					}		
				}
			}

			
			//if(pcl_pcVec.size()>1)
			//{
				//Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity (), pairTransform;
			//	for(int i=1; i<pcl_pcVec.size(); i++)
			//	{
			/*
					pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
  					icp.setInputSource(pcl_pcVec[1]);
  					icp.setInputTarget(pcl_pcVec[0]);
					// Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored)
					icp.setMaxCorrespondenceDistance (100);
					// Set the maximum number of iterations (criterion 1)
					icp.setMaximumIterations (50);
					// Set the transformation epsilon (criterion 2)
					//icp.setTransformationEpsilon (1e-8);
					// Set the euclidean distance difference epsilon (criterion 3)
					//icp.setEuclideanFitnessEpsilon (1000);
  					pcl::PointCloud<pcl::PointXYZRGB>::Ptr Final(new pcl::PointCloud<pcl::PointXYZRGB>());
  					icp.align(*Final);
					std::cout << "has converged:" << icp.hasConverged() << " score: " <<
					  		icp.getFitnessScore() << std::endl;

					std::cout << icp.getFinalTransformation() << std::endl;
					*/

				//	pcl::transformPointCloud (*temp, *result, GlobalTransform);
			//	}
			//}

			//std::cout << "clean & registration:" << ((double)getTickCount() - t1) / getTickFrequency() << std::endl;

			// check point cloud size, too small, go to next range
			int pointCloudSize = 0;
			for(int i=0; i<pcl_pcVec.size(); i++)
				pointCloudSize += pcl_pcVec[i]->points.size();

			if(pointCloudSize < 4000) {
			
				std::cout<<"Not enough data points"<<std::endl;
				continue;
			}


			// SOIL REMOVAL
			// base line variables
			pcl::PointXYZ rootPoint1, rootPoint2, rootPoint0, rootLineDir;
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_tmp(new pcl::PointCloud<pcl::PointXYZRGB>);
			bool FOUND_NEW_LINE = false;

#if 1
			if(plant_base_detect_mode == 0)
			{
				// RANSAC fit root line from line by user
				if( pc_rootLinePoints->points.size() > 20)
				{
					pcl::ModelCoefficients::Ptr coefficients_root(new pcl::ModelCoefficients);
					pcl::PointIndices::Ptr inliers_root(new pcl::PointIndices);
					pcl::SACSegmentation<pcl::PointXYZ> seg_root;
					seg_root.setOptimizeCoefficients(true);
					seg_root.setModelType(pcl::SACMODEL_LINE);
					seg_root.setMethodType(pcl::SAC_LMEDS);
					seg_root.setDistanceThreshold(30);
					seg_root.setAxis(Eigen::Vector3f::UnitY());
					seg_root.setEpsAngle(0);
					seg_root.setInputCloud(pc_rootLinePoints);
					seg_root.segment(*inliers_root, *coefficients_root);
					std::cout<<"root line point:"<<coefficients_root->values[0]<<" "<<coefficients_root->values[1]<<" "<<coefficients_root->values[2]<<" dir:"
						<<coefficients_root->values[3]<<" "<<coefficients_root->values[4]<<" "<<coefficients_root->values[5]<<std::endl;

					rootPoint0.x = coefficients_root->values[0];
					rootPoint0.y = coefficients_root->values[1];
					rootPoint0.z = coefficients_root->values[2];
					rootLineDir.x = coefficients_root->values[3];
					rootLineDir.y = coefficients_root->values[4];
					rootLineDir.z = coefficients_root->values[5];

					// check if newly computed root line is accurate
					float line2lineDistance = fabs(rootPoint0.x-preRootPoint.x);
				
					//if( line2lineDistance < 200 && fabs(rootLineDir.y) > 0.99f)	// if the distance is ok, update 
					{
						std::cout<<"use New line!"<<std::endl;
						preRootPoint.x = rootPoint0.x;
						preRootPoint.y = rootPoint0.y;
						preRootPoint.z = rootPoint0.z;
						preRootLineDir.x = rootLineDir.x;
						preRootLineDir.y = rootLineDir.y;
						preRootLineDir.z = rootLineDir.z;
				
						if(rootLinePoints.size()==2)
						{
							preRootLinePoints[0] = rootLinePoints[0];
							preRootLinePoints[1] = rootLinePoints[1];	
						}
						FOUND_NEW_LINE = true;
					}	
				}

				if(!FOUND_NEW_LINE)	// if distance is too large, use previous
				{
					std::cout<<"use PREVIOUS baseline!"<<std::endl;
					rootPoint0.x = preRootPoint.x;
					rootPoint0.y = preRootPoint.y;
					rootPoint0.z = preRootPoint.z;
					rootLineDir.x = preRootLineDir.x;
					rootLineDir.y = preRootLineDir.y;
					rootLineDir.z = preRootLineDir.z;		
				}	
			
				//pcViewer->addPointCloud(pc_rootLinePoints, "root line points", 0);
			}
#endif	
			else if(plant_base_detect_mode == 1)
			{
			// find baseline by using ICP template matching
#if 1
				Eigen::Vector4f btmCentroid;
				if(cameraType == CANON)
					pcl::compute3DCentroid(*pcl_pcVec[0], btmCentroid);
				//std::cout<<"btm pc centroid:"<<btmCentroid<<std::endl;

				pcl::PassThrough<pcl::PointXYZRGB> pass;
			
				pass.setInputCloud(pcl_pcVec[0]);

				if(cameraType == PG)			
				{
					pass.setFilterFieldName("x");
					pass.setFilterLimits(-3000., -180.);
				}
				else if(cameraType == CANON)
				{
					//pass.setFilterFieldName("y");
					//pass.setFilterLimits(btmCentroid(1), 3000.);
				
					pass.setFilterFieldName("x");
					pass.setFilterLimits(-3000., btmCentroid(0)+100);
				}

				pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_btmPart(new pcl::PointCloud<pcl::PointXYZRGB>);
				pass.setFilterLimitsNegative(false);
				pc_tmp->clear();
				pass.filter(*pc_tmp);

				pass.setFilterFieldName("z");
				if(cameraType == PG)
					pass.setFilterLimits(1000., 1500.);
				else if(cameraType == CANON)
					pass.setFilterLimits(800., 1800.);

				pass.setInputCloud(pc_tmp);
				pass.filter(*pc_btmPart);
				for(int i=0; i<pc_btmPart->points.size(); i++)
					pc_btmPart->points[i].b = 255;

				Eigen::Matrix4f baseTemplateTransform;
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_transformedTemplate(new pcl::PointCloud<pcl::PointXYZRGB>);
				std::cout<<"btm size:"<<pc_btmPart->points.size()<<std::endl;
				if(pc_btmPart->points.size() > 1000)
				{
					pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
					icp.setInputSource(pc_baseTemplate);
					icp.setInputTarget(pc_btmPart);
					// Set the max correspondence distance to 40cm (e.g., correspondences with higher distances will be ignored)
					icp.setMaxCorrespondenceDistance (400.f);
					// Set the maximum number of iterations (criterion 1)
					icp.setMaximumIterations (50);
					// Set the transformation epsilon (criterion 2)
					//icp.setTransformationEpsilon (1e-8);
					// Set the euclidean distance difference epsilon (criterion 3)
					//icp.setEuclideanFitnessEpsilon (1000);
				
					icp.align(*pc_transformedTemplate);
					baseTemplateTransform = icp.getFinalTransformation()*baseTemplateInitTransform;
					std::cout<< baseTemplateTransform<<std::endl;

					pcViewer->addPointCloud(pc_baseTemplate, "init template", 0);

					pcViewer->addPointCloud(pc_btmPart, "btmPart", 0);
					//pcViewer->spinOnce(pcl_view_time);
					pcViewer->addPointCloud(pc_transformedTemplate, "template", 0);
					pcViewer->spinOnce(pcl_view_time);
					pcViewer->removePointCloud("init template", 0);
					pcViewer->removePointCloud("btmPart", 0);
					pcViewer->removePointCloud("template", 0);

					const float xDifferenceThresh = cameraType == PG ? 150.f : 300.f;
					const float zDifferenceThresh = cameraType == PG ? 150.f : 300.f;
					const float xAlignThresh = cameraType == PG ? 0.98f : 0.9f;
					const float zAlignThresh = cameraType == PG ? 0.98f : 0.9f;

					//check if transformation valid, if not use previous
					if(cameraType == PG)
					{
						if(   baseTemplateTransform(1,1) < 0.99f 	//Not align well with y axis
						   || fabs(baseTemplateTransform(0,3)-preTemplateTransform(0,3)) > xDifferenceThresh	// x translation too large 
						   || fabs(baseTemplateTransform(2,3)-preTemplateTransform(2,3)) > zDifferenceThresh	// z translation too large
						   || baseTemplateTransform(0,0) < xAlignThresh 	//Not align well with x axis
						   || baseTemplateTransform(2,2) < zAlignThresh 	//Not align well with z axis
						  )
						{
							baseTemplateTransform = preTemplateTransform;
							pc_transformedTemplate->clear();
							pcl::transformPointCloud(*pc_baseTemplate, *pc_transformedTemplate, baseTemplateTransform*baseTemplateInitTransform.inverse());
							std::cout<<"Use PREVIOUS baseline"<<std::endl;
						}
						else
						{
							preTemplateTransform = baseTemplateTransform;
							std::cout<<"Use NEW baseline"<<std::endl;
						}
					}
					else if(cameraType == CANON)
					{
						if(   baseTemplateTransform(0,0) < 0.9f 	//Not align well with x axis
					/*	   || fabs(baseTemplateTransform(1,3)-preTemplateTransform(1,3)) > xDifferenceThresh	// y translation too large 
						   || fabs(baseTemplateTransform(2,3)-preTemplateTransform(2,3)) > zDifferenceThresh	// z translation too large
						   || baseTemplateTransform(1,1) < xAlignThresh 	//Not align well with y axis
						   || baseTemplateTransform(2,2) < zAlignThresh 	//Not align well with z axis
					*/	  )
						{
							baseTemplateTransform = preTemplateTransform;
							pc_transformedTemplate->clear();
							pcl::transformPointCloud(*pc_baseTemplate, *pc_transformedTemplate, baseTemplateTransform*baseTemplateInitTransform.inverse());
							std::cout<<"Use PREVIOUS baseline"<<std::endl;
						}
						else
						{
							preTemplateTransform = baseTemplateTransform;
							std::cout<<"Use NEW baseline"<<std::endl;
						}
					}
				}
				else
				{
					std::cout<<"btm pc size too small"<<std::endl;
					baseTemplateTransform = preTemplateTransform;
					pc_transformedTemplate->clear();
					pcl::transformPointCloud(*pc_baseTemplate, *pc_transformedTemplate, baseTemplateTransform*baseTemplateInitTransform.inverse());
					std::cout<<"Use PREVIOUS baseline"<<std::endl;
				}

				//pick two vectors on the yz plane of the template and form a plane	
				Eigen::Vector3f v0(0.f, 1.f, 0.f);
				Eigen::Vector3f v1(0.f, 0.f, -1.f);
				Eigen::Vector3f v2(1.f, 0.f, 0.f);

				v0 = baseTemplateTransform.block<3,3>(0,0)*v0;
				v1 = baseTemplateTransform.block<3,3>(0,0)*v1;
				v2 = baseTemplateTransform.block<3,3>(0,0)*v2;

				Eigen::Vector3f v3(v1(0),v1(1),v1(2));

				// intersection of right angle template
				Eigen::Matrix3f rotationMatrix;
				rotationMatrix = Eigen::AngleAxisf(-M_PI/180.*15., v0);

				v1 = rotationMatrix*v1;

				Eigen::Vector3f cross;
				cross = v0.cross(v1);

				Eigen::Matrix3f rotationMatrix2;
				rotationMatrix2 = Eigen::AngleAxisf(M_PI/180.*15., v0);

				v3 = rotationMatrix2 * v3;

				Eigen::Vector3f crossBack;
				crossBack = v0.cross(v3);
			
				Eigen::Vector3f pointOnPlane = v0*500.f + baseTemplateTransform.block<3,1>(0,3);
		
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_soil(new pcl::PointCloud<pcl::PointXYZRGB>);

				// save points above plane
				pc_tmp->clear();
				for(int i=0; i<pcl_pcVec[0]->points.size(); i++)
				{
					Eigen::Vector3f point(pcl_pcVec[0]->points[i].x, pcl_pcVec[0]->points[i].y, pcl_pcVec[0]->points[i].z);
					point = point -20.f*v2 - pointOnPlane;

					if(point.dot(cross) < 0.0f && point.dot(crossBack) < 0.0f) {
					
						// plant points
						pc_tmp->push_back(pcl_pcVec[0]->points[i]);
						//pc_btmPart->points[i].g = 255;
					}
					else {
					
						pcl_pcVec[0]->points[i].b = 200;
						pcl_pcVec[0]->points[i].g = 200;
						pc_soil->push_back(pcl_pcVec[0]->points[i]);
					}
				}

				pcl_pcVec[0]->clear();
				*pcl_pcVec[0] += *pc_tmp;	

				// update rootline point 0
				rootPoint0.x = pointOnPlane(0);
				rootPoint0.y = pointOnPlane(1);
				rootPoint0.z = pointOnPlane(2);
				rootLineDir.x = v0(0);
				rootLineDir.y = v0(1);
				rootLineDir.z = v0(2);

				//pcViewer->addPointCloud(pc_transformedTemplate, "align", 0);

				pcViewer->addPointCloud(pc_soil, "soil", 0);

				pcViewer->spinOnce(pcl_view_time);
				
				// Euclidean cluster, remove small clusters
				pc_tmp->clear();	

				pfe.smallClusterRemoval(pcl_pcVec[0], clusterTolerance, minClusterSize, pc_tmp);	
	
				pcl_pcVec[0]->clear();

				*pcl_pcVec[0] += *pc_tmp;
			}
#endif	

			rootPoint1.x = rootPoint0.x+rootLineDir.x*1000.f;
			rootPoint1.y = rootPoint0.y+rootLineDir.y*1000.f;
			rootPoint1.z = rootPoint0.z+rootLineDir.z*1000.f;
			rootPoint2.x = rootPoint0.x-rootLineDir.x*1000.f;
			rootPoint2.y = rootPoint0.y-rootLineDir.y*1000.f;
			rootPoint2.z = rootPoint0.z-rootLineDir.z*1000.f;

			pointCloudSize = 0;
			for(int i=0; i<pcl_pcVec.size(); i++)
				pointCloudSize += pcl_pcVec[i]->points.size();

			if(pointCloudSize < 4000)
			{
				std::cout<<"Not enough data points"<<std::endl;
				continue;
			}

			// vegetation bounding box
			pcl::PointXYZRGB min_point_AABB;
			pcl::PointXYZRGB max_point_AABB;

			
			// ransac fit soil plane, 
#if 0	// old soil
			// fit soil plane
			// cut out bottom part of btm point cloud
			pass.setInputCloud(pcl_pcVec[0]);
			pass.setFilterFieldName("x");
			
			pass.setFilterLimits(-3000., -280.);
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_btmPart(new pcl::PointCloud<pcl::PointXYZRGB>());
			pass.setFilterLimitsNegative(false);
			pass.filter(*pc_btmPart);
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_upperPart(new pcl::PointCloud<pcl::PointXYZRGB>());
			pass.setFilterLimitsNegative(true);
			pass.filter(*pc_upperPart);

			// RANSAC fit soil plane
			pcl::ModelCoefficients::Ptr coefficients_soil(new pcl::ModelCoefficients);
			pcl::PointIndices::Ptr inliers_soil(new pcl::PointIndices);
			pcl::SACSegmentation<pcl::PointXYZRGB> seg;
			seg.setOptimizeCoefficients(true);
			seg.setModelType(pcl::SACMODEL_PLANE);
			seg.setMethodType(pcl::SAC_RANSAC);
			seg.setDistanceThreshold(30);
			Eigen::Vector3f axis_x(1, 0, 0);
			seg.setAxis(axis_x);
			seg.setEpsAngle(M_PI/9);
			seg.setInputCloud(pc_btmPart);
			seg.segment(*inliers_soil, *coefficients_soil);
			std::cout<<"Soil Plane:"<<coefficients_soil->values[0]<<" "<<coefficients_soil->values[1]<<" "<<coefficients_soil->values[2]<<" "
				<<coefficients_soil->values[3]<<std::endl;

			// extract soil outlier			
			pcl::ExtractIndices<pcl::PointXYZRGB> extract;
			extract.setInputCloud(pc_btmPart);
			extract.setIndices(inliers_soil);
			extract.setNegative(true);
			pc_tmp->clear();
			extract.filter(*pc_tmp);
			extract.setNegative(false);
			extract.filter(*pc_inlierOfSoilPlane); // points within 30 mm from soil plane

			// compute soil plane centroid
			/*Eigen::Vector4f soilCentroid;
			pcl::compute3DCentroid(*pc_inlierOfSoilPlane, soilCentroid);
			std::cout<<"soil centroid:"<<soilCentroid<<std::endl;*/


			// extract  inliers and outliers of fitted soil plane			
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_outlierOfSoilPlane(new pcl::PointCloud<pcl::PointXYZRGB>());
			
			// check points 30 mm away from soil plane in pc_btmPart
			for(int i=0; i<pc_tmp->points.size(); i++)
			{
				pcl::PointXYZRGB point = pc_tmp->points[i];
				
				// above the plane
				if( point.x*coefficients_soil->values[0]+point.y*coefficients_soil->values[1]+
				    point.z*coefficients_soil->values[2]+coefficients_soil->values[3] > 0.f)
					pc_outlierOfSoilPlane->push_back(point);
				else
					pc_inlierOfSoilPlane->push_back(point);
			}

			// remove soil from btm point cloud
			pcl_pcVec[0]->clear();
			*pcl_pcVec[0] += *pc_upperPart;
			*pcl_pcVec[0] += *pc_outlierOfSoilPlane;
#endif

#if 0	// enable if soil is used
			// Euclidean cluster, remove small clusters close to soil plane
			pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
			std::vector<pcl::PointIndices> cluster_indices;
			pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
			ec.setClusterTolerance(clusterTolerance); //distance mm
			ec.setMinClusterSize(1);
			ec.setMaxClusterSize(pcl_pcVec[0]->points.size());
			ec.setSearchMethod(tree);
			ec.setInputCloud(pcl_pcVec[0]);
			ec.extract(cluster_indices);

			pc_tmp->clear();
			for(int j=0; j<cluster_indices.size(); j++)
			{
				if(cluster_indices[j].indices.size() > minClusterSize)
				for(int i=0; i<cluster_indices[j].indices.size(); i++)
				{
					pc_tmp->push_back(pcl_pcVec[0]->points[cluster_indices[j].indices[i]]);
				}
			}	

			pcl_pcVec[0]->clear();		
			*pcl_pcVec[0] += *pc_tmp;
						
			// change color of soil plane inlier
			for(int i=0; i<pc_inlierOfSoilPlane->points.size(); i++)
			{
				pc_inlierOfSoilPlane->points[i].b = 255;
				pc_inlierOfSoilPlane->points[i].g = 255;
			}
#endif

			
#if 0	// old soil cut
			// align soil plane norm with x axis
			Eigen::Matrix4f transform_alignX = Eigen::Matrix4f::Identity();
			Eigen::Vector3f soilPlaneNormal(coefficients_soil->values[0], coefficients_soil->values[1], coefficients_soil->values[2]);
			soilPlaneNormal.normalize();
			transform_alignX.block<1, 3>(0, 0) = soilPlaneNormal;
			Eigen::Vector3f axis_y(0, 1, 0);		
			transform_alignX.block<1, 3>(2, 0) = soilPlaneNormal.cross(axis_y);		
			
			for(int i=0; i<pcl_pcVec.size(); i++)
			{
				pc_tmp->clear();
				pcl::transformPointCloud(*pcl_pcVec[i], *pc_tmp, transform_alignX);
				pcl_pcVec[i]->clear();
				*pcl_pcVec[i] += *pc_tmp;
			}

			pc_tmp->clear();
			pcl::transformPointCloud(*pc_inlierOfSoilPlane, *pc_tmp, transform_alignX);
			pc_inlierOfSoilPlane->clear();
			*pc_inlierOfSoilPlane += *pc_tmp;
			
			// Find two points on the plane and draw line
			pcl::PointCloud<pcl::PointXYZ>::Ptr linePoints(new pcl::PointCloud<pcl::PointXYZ>());
			pcl::PointCloud<pcl::PointXYZ>::Ptr plantPlaneLineProjection(new pcl::PointCloud<pcl::PointXYZ>());
			pcl::PointCloud<pcl::PointXYZ>::Ptr soilPlaneLineProjection(new pcl::PointCloud<pcl::PointXYZ>());
			linePoints->push_back(pcl::PointXYZ(0, -1000.f, 1000.f));
			linePoints->push_back(pcl::PointXYZ(0, 1000.f, 2000.f));
			linePoints->push_back(pcl::PointXYZ(0, -1000.f, 2000.f));
			linePoints->push_back(pcl::PointXYZ(0, 1000.f, 1000.f));
			pcl::ProjectInliers<pcl::PointXYZ> proj;
			proj.setModelType(pcl::SACMODEL_PLANE);
			proj.setInputCloud(linePoints);
			proj.setModelCoefficients(coefficients_soil);
			proj.filter(*soilPlaneLineProjection);

			// transform soil line points 
			pcl::PointCloud<pcl::PointXYZ>::Ptr pc_tmp1(new pcl::PointCloud<pcl::PointXYZ>());
			pcl::transformPointCloud(*soilPlaneLineProjection, *pc_tmp1, transform_alignX);
			soilPlaneLineProjection->clear();
			*soilPlaneLineProjection += *pc_tmp1;

			// compute eigen vectors
			// crop out the bottom part to estimate plant row direction
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_plantRowDir(new pcl::PointCloud<pcl::PointXYZRGB>());
			*pc_plantRowDir += *pcl_pcVec[0];
			*pc_plantRowDir += *pc_inlierOfSoilPlane;
			pass.setInputCloud(pc_plantRowDir);
			pass.setFilterFieldName("x");
			pass.setFilterLimits(-3000., 0.);
			pass.setFilterLimitsNegative(false);
			pc_tmp->clear();
			pass.filter(*pc_tmp); 

			pcl::MomentOfInertiaEstimation <pcl::PointXYZRGB> feature_extractor;
			feature_extractor.setInputCloud (pc_tmp);
			feature_extractor.compute ();

			std::vector<float> moment_of_inertia;
			std::vector<float> eccentricity;
			
			pcl::PointXYZRGB min_point_OBB;
			pcl::PointXYZRGB max_point_OBB;
			pcl::PointXYZRGB position_OBB;
			Eigen::Matrix3f rotational_matrix_OBB;
			float major_value, middle_value, minor_value;
			Eigen::Vector3f major_vector, middle_vector, minor_vector;
			Eigen::Vector3f mass_center;

			//feature_extractor.getEigenValues (major_value, middle_value, minor_value);
			feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);
			feature_extractor.getMassCenter(mass_center);

			// find the vector aligned with the plant row
			Eigen::Vector3f plantRow_vector;
			// find max y value
			plantRow_vector = fabs(major_vector(1)) > fabs(middle_vector(1)) ? major_vector : middle_vector;
			//plantRow_vector = major_vector + middle_vector;
			//std::cout<<"plant row vector:"<<plantRow_vector<<std::endl;

			// project plant row vector onto soil plane
			plantRow_vector(0) = 0;
			plantRow_vector.normalize();
			if(plantRow_vector(1) < 0)
				plantRow_vector *= -1;
			//std::cout<<"normalized plant row vector:"<<plantRow_vector<<std::endl;
			std::cout<<"angle:"<< atan(plantRow_vector(2)/plantRow_vector(1))/M_PI*180.f<<std::endl;
			

			// align y axis with plant row direction			
			Eigen::Matrix4f transform_alignY = Eigen::Matrix4f::Identity();
			transform_alignY.block<1, 3>(0, 0) = axis_x;			
			transform_alignY.block<1, 3>(1, 0) = plantRow_vector;
			transform_alignY.block<1, 3>(2, 0) = axis_x.cross(plantRow_vector);	
			// transform plant points
			for(int i=0; i<pcl_pcVec.size(); i++)
			{
				pc_tmp->clear();
				pcl::transformPointCloud(*pcl_pcVec[i], *pc_tmp, transform_alignY);
				pcl_pcVec[i]->clear();
				*pcl_pcVec[i] += *pc_tmp;
			}
			// transform soil points
			pc_tmp->clear();
			pcl::transformPointCloud(*pc_inlierOfSoilPlane, *pc_tmp, transform_alignY);
			pc_inlierOfSoilPlane->clear();
			*pc_inlierOfSoilPlane += *pc_tmp;

			// transform soil line points 
			pc_tmp1->clear();
			pcl::transformPointCloud(*soilPlaneLineProjection, *pc_tmp1, transform_alignY);
			soilPlaneLineProjection->clear();
			*soilPlaneLineProjection += *pc_tmp1;

			// transform mass center and eigen vectors
			Eigen::Vector4f homo_vector;
			homo_vector.block<3,1>(0,0) = mass_center;
			homo_vector(3) = 1;
			homo_vector = transform_alignY*homo_vector;
			mass_center = homo_vector.block<3,1>(0,0);

			homo_vector.block<3,1>(0,0) = major_vector;
			homo_vector(3) = 1;
			homo_vector = transform_alignY*homo_vector;
			major_vector = homo_vector.block<3,1>(0,0);

			homo_vector.block<3,1>(0,0) = minor_vector;
			homo_vector(3) = 1;
			homo_vector = transform_alignY*homo_vector;
			minor_vector = homo_vector.block<3,1>(0,0);

			homo_vector.block<3,1>(0,0) = middle_vector;
			homo_vector(3) = 1;
			homo_vector = transform_alignY*homo_vector;
			middle_vector = homo_vector.block<3,1>(0,0);

#endif
			// axis aligned bounding box
			pc_tmp->clear();
			for(int i=0; i<pcl_pcVec.size(); i++)
				*pc_tmp += *pcl_pcVec[i];

			// down sample 
			pcl::VoxelGrid<pcl::PointXYZRGB> vg;
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_voxelDown(new pcl::PointCloud<pcl::PointXYZRGB>);
			vg.setInputCloud(pc_tmp);
			vg.setLeafSize(10.f, 10.f, 10.f);
			vg.filter(*pc_voxelDown);
			pc_tmp->clear();
			*pc_tmp = *pc_voxelDown;


			// initial AABB
			pcl::getMinMax3D (*pc_tmp, min_point_AABB, max_point_AABB); 

			//std::cout<<"before:"<<max_point_AABB.z;

			// Refine Z MAX
			max_point_AABB.z = pfe.refineZMax(pc_tmp, max_point_AABB.z, 4);

			//std::cout<<"   after:"<<max_point_AABB.z<<std::endl;

			// Refine X MIN (BASE)
			min_point_AABB.x = rootPoint0.x;

#if 0
			// number of subbox experiment
			float weightedMedianHeight1 = 0;
			float weightedMedianWidth1 = 0;

			pcViewer->addPointCloud(pc_voxelDown, "pc0", 0);

			ofstream out("heightwidthexp.csv", ios::app);

			out<<std::endl<<"range:"<< curRange<<" row:"<<curRow<<" dateID:"<<dateID<<" fieldID:"<<fieldID<<" range length:"<<max_point_AABB.y-min_point_AABB.y<<std::endl;			
		
			for(int n=1; n<=128; n++)
			{
				std::string sliceAxis = "y";

				double runtime = (double)cv::getTickCount();
				
				pfe.computeSlicedSubPointCloudVec(pc_tmp, n, sliceAxis);

				pfe.weightedMedianHeightWidth(weightedMedianHeight1, weightedMedianWidth1);	

				runtime = ((double)getTickCount() - runtime) / getTickFrequency();			
			
				std::string shapeName = "AABB"+std::to_string(n-1);

				pcViewer->removeShape(shapeName, 0);

				shapeName = "AABB"+std::to_string(n);

				//pcViewer->addCube (min_point_AABB.x, weightedMedianHeight1, min_point_AABB.y, max_point_AABB.y, weightedMedianWidth1, max_point_AABB.z, 1.0, 1.0, 0.0, shapeName);
				pcViewer->setRepresentationToWireframeForAllActors(); 	

				pcViewer->spinOnce(100);

				weightedMedianHeight1 -= min_point_AABB.x;

				weightedMedianWidth1 = max_point_AABB.z - weightedMedianWidth1;
				
				float length = (max_point_AABB.y-min_point_AABB.y)/(float)n;

				out<<"n,"<<n<<", slice len,"<<length<<", height,"<<weightedMedianHeight1<<", width,"<<weightedMedianWidth1<<", time,"<<runtime<<std::endl;
			}
			
			out.close();

			pcViewer->spin();

			continue;
#endif

#if 0
			//pcViewer->addPointCloud(pc_baseTemplate, "template", 0);
			//pcViewer->addPointCloud(pc_transformedTemplate, "icp template", 0);

			pcViewer->addCube (min_point_AABB.x, weightedMedianHeight1/*max_point_AABB.x*/, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 1.0, 1.0, 0.0, "AABB");
			pcViewer->addPointCloud(pcl_pcVec[0], "pc0", 0);
			if(pcl_pcVec.size()>1)

			pcViewer->addPointCloud(pcl_pcVec[1], "pc1", 0);
			if(pcl_pcVec.size()>2)
			pcViewer->addPointCloud(pcl_pcVec[2], "pc2", 0);

			//pcViewer->addPointCloud(pc_outsideAABB, "outside", 0); 

			//pcViewer->addPointCloud(pc_btmPart, "btm part", 0);
			pcViewer->spin();
			continue;
#endif	
			// slice point cloud 
			std::string sliceAxis = "y";

			t1 = (double)getTickCount();
			pfe.computeSlicedSubPointCloudVec(pc_tmp, numSubBoxUsed, sliceAxis);
			
			// DENSITY WEIGHTED MEDIAN PLANT HEIGHT
			float weightedMedianHeight = 0;
			float weightedMedianWidth = 0;

			pfe.weightedMedianHeightWidth(weightedMedianHeight, weightedMedianWidth);

			std::cout << "time of height and width: " << ((double)getTickCount() - t1) / getTickFrequency() << std::endl;

			// refine bounding box
			min_point_AABB.z = weightedMedianWidth;

			max_point_AABB.x = weightedMedianHeight;
			
			// subtract soil height
			weightedMedianHeight -= min_point_AABB.x;
			
			weightedMedianWidth = max_point_AABB.z - weightedMedianWidth;	

			std::cout<<"weighted median height of "<< numSubBoxUsed << " sub boxs:" << weightedMedianHeight<<std::endl;

			// apply new bounding box 
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_outsideAABB(new pcl::PointCloud<pcl::PointXYZRGB>);

			pfe.updateBoundingBox(pcl_pcVec, min_point_AABB, max_point_AABB, pc_outsideAABB);

			std::cout<<"outside size:"<<pc_outsideAABB->points.size()<<std::endl;

			//std::cout<<"above height size:"<<pc_outsideAABB->points.size()<<std::endl;


			// project to side of the bounding box
			pc_tmp->clear();
			for(int i=0; i<pcl_pcVec.size(); i++)
				*pc_tmp += *pcl_pcVec[i];


			//recompute sliced point cloud
			pfe.computeSlicedSubPointCloudVec(pc_tmp, numSubBoxUsed, sliceAxis);

			// CONVEX HULL VOLUME
			// add two btm corner points to plant point cloud in case of no data
			/*pcl::PointXYZRGB btmCornerPoint;
			btmCornerPoint.x = min_point_AABB.x;
			btmCornerPoint.y = max_point_AABB.y;
			btmCornerPoint.z = max_point_AABB.z;
			btmCornerPoint.r = 255; btmCornerPoint.g = 0; btmCornerPoint.b = 0;
			pcl_pcVec[0]->push_back(btmCornerPoint);

			btmCornerPoint.x = min_point_AABB.x;
			btmCornerPoint.y = min_point_AABB.y;
			btmCornerPoint.z = max_point_AABB.z;
			btmCornerPoint.r = 255; btmCornerPoint.g = 0; btmCornerPoint.b = 0;
			pcl_pcVec[0]->push_back(btmCornerPoint);*/

			
			pc_tmp->clear();
			for(int i=0; i<pcl_pcVec.size(); i++)
				*pc_tmp += *pcl_pcVec[i];

			// sliced convex hull volume
			std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> cloudHullVec;

			std::vector<std::vector<pcl::Vertices>> verticesChullVec;

			int numValidSubBox = 0;

			double volume = 0.;

			//double area = 0.;
			
			t1 = (double)cv::getTickCount();

			std::vector<double> sub_plant_volume_vec;
			std::vector<double> sub_plant_chull_area_vec;
			double slice_volume = 0.;
			volume = pfe.getPlantVolume(volumeThresh, cloudHullVec, verticesChullVec, numValidSubBox, sub_plant_volume_vec, slice_volume, sub_plant_chull_area_vec);

			std::cout << "time of volume: " << ((double)cv::getTickCount() - t1) / cv::getTickFrequency() << std::endl;

			double adjustFactor = (double)numValidSubBox/(double)numSubBoxUsed;

			double groundArea = (max_point_AABB.y-min_point_AABB.y)*(max_point_AABB.z-min_point_AABB.z)*1e-6*adjustFactor;
        		std::cout<<"Ground area (m^2):"<<groundArea<<std::endl;

        		double cubeVolume = groundArea*(max_point_AABB.x-min_point_AABB.x)*1e-3;
        		std::cout<<"Cube volume (m^3):"<<cubeVolume<<std::endl;

			
			//pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZRGB>);
			//std::vector<pcl::Vertices> vertices_chull;
			//pfe.getConvexHull(pc_tmp, cloud_hull, vertices_chull, volume, area);
			
			
			double vegetationVolume = volume/*chull.getTotalVolume()*/*1e-9;
			std::cout<<"Plant Volume (m^3):"<<vegetationVolume<<" ";
			double VVI = vegetationVolume/cubeVolume;
			std::cout<<"VVI:"<<VVI<<std::endl;
			
			//double chullArea = area/*chull.getTotalArea()*/*1e-6;
			//std::cout<<"chull area (m^2):"<< chullArea<<std::endl;

			// projection onto bounding box

			std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pc_projectVec;

			for(int i=0; i<3; i++)
			{
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_project(new pcl::PointCloud<pcl::PointXYZRGB>);

				pfe.projectToAABBSideFace(min_point_AABB, max_point_AABB, i, pc_project);

				pc_projectVec.push_back(pc_project);

				if(i==1) pcViewer->addPointCloud(pc_projectVec[i], "plant project"+std::to_string(i), 0);
			}

			// projection occupancy
			// y axis
			double projectOccupancy[3] = {0.};		
			
			projectOccupancy[1] = (double)pc_projectVec[1]->points.size()/(weightedMedianHeight/10.*weightedMedianWidth/10.);

			double validRowLen = (max_point_AABB.y - min_point_AABB.y)*adjustFactor;

			projectOccupancy[0] = (double)pc_projectVec[0]->points.size()/(validRowLen/10.*weightedMedianWidth/10.);

			projectOccupancy[2] = (double)pc_projectVec[2]->points.size()/(validRowLen/10.*weightedMedianHeight/10.);

			for(int i=0; i<3; i++)
			{
				projectOccupancy[i] = min(projectOccupancy[i], 1.);			
			}

			std::cout<<"projection occupany: " << projectOccupancy[0] << "; "<<projectOccupancy[1] 
			<< "; "<< projectOccupancy[2] << std::endl;	


			// centroid to range ratio
			pc_tmp->clear();
			for(int i=0; i<pcl_pcVec.size(); i++)
				*pc_tmp += *pcl_pcVec[i];

			// down sample 
			vg.setInputCloud(pc_tmp);
			vg.setLeafSize(10.f, 10.f, 10.f);
			pc_voxelDown->points.clear();
			vg.filter(*pc_voxelDown);

			Eigen::Vector4f mass2_center;

			pcl::compute3DCentroid(*pc_voxelDown, mass2_center);
			
			double Centroid2HeightRatio = (mass2_center(0) - min_point_AABB.x)/weightedMedianHeight;

			double Centroid2WidthRatio = (max_point_AABB.z - mass2_center(2))/weightedMedianWidth;

			std::cout<<"Centroid2HeightRatio: " << Centroid2HeightRatio << "  Centroid2WidthRatio: " << Centroid2WidthRatio << std::endl;

			double leafArea = 0.f;
			double LAI = 0.f;		
#if 1
			// TRIANGLE MESH
			// Moving least square smoothing
			t1 = (double)cv::getTickCount();
			pc_tmp->clear();
			for(int i=0; i<pcl_pcVec.size(); i++) *pc_tmp += *pcl_pcVec[i];

		//	pcViewer->addPointCloud(pcl_pcVec[0], "vec0"); pcViewer->spin();
			
		//	sorf.setInputCloud(pcl_pcVec[0]);
			
		//	sorf.filter(*pc_tmp);

		//	*pc_tmp += *pcl_pcVec[1];
				

			pcl::PointCloud<pcl::PointXYZRGB>::Ptr mls_points(new pcl::PointCloud<pcl::PointXYZRGB>);
			
			pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr mls_points_normal(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
			
			//pcl::copyPointCloud(*pc_tmp, *mls_points);		
			pfe.movingLeastSquareSmooth(pc_tmp, mls_points_normal);
			
			pcl::copyPointCloud(*mls_points_normal, *mls_points);

			std::cout<<"size: "<<mls_points->points.size()<<std::endl;

			std::cout << "MLS smoothing:" << ((double)cv::getTickCount() - t1) / cv::getTickFrequency() << std::endl;
			
			//pcViewer->addPointCloud(mls_points, "mls"); pcViewer->spin();

			t1 = (double)cv::getTickCount();
			pcl::PolygonMesh triangles;	
			pfe.greedyTriangulation(mls_points_normal, triangles);
			std::cout << "Normal+Triangulation:" << ((double)cv::getTickCount() - t1) / cv::getTickFrequency() << std::endl;

			// calculate total area
			t1 = (double)cv::getTickCount();
			leafArea = pfe.getMeshAreaSum(triangles)*1e-6;
			std::cout << "time of total leaf area: " << ((double)cv::getTickCount() - t1) / cv::getTickFrequency() << std::endl;
			
			LAI = leafArea/groundArea;
			std::cout<<"Total leaf area (m^2):"<<leafArea<<" ";
			std::cout<<"LAI:"<<LAI<<std::endl;
			
#endif
			/*pcViewer->addPointCloud(pc_originBaseTemplate, "origin base template", 0);
			pcViewer->addSphere (pc_originBaseTemplate->points[25], 20, 1., 0., 0.0, "sphere", 0);

			pcViewer->spin();*/

	
			// calculate hedge width at 3 levels
			std::vector<float> three_hedge_widths(3, 1e5f);

			pfe.compute3LevelHedgeWidths(pc_tmp, numSubBoxUsed, min_point_AABB, max_point_AABB, three_hedge_widths);

			for(auto & h : three_hedge_widths) { h = max_point_AABB.z - h; cout<<h<<" ";}cout<<"\n";


#if 1
			std::string displayText = "Height(mm): "+std::to_string(weightedMedianHeight) + "\n\nWidth(mm): " + std::to_string(weightedMedianWidth) 
						+ "\n\nVolume(m^3): " + std::to_string(vegetationVolume) + "\n\nVolume Index: " + std::to_string(VVI) 
						+ "\n\nArea(m^2): " + std::to_string(leafArea)  + "\n\nArea Index: " + std::to_string(LAI);

			pcViewer->addText(displayText, 50, 200, "text", 0);
#endif

			// save 
			result<<fieldID<<","<<curRange<<","<<curRow<<","<<weightedMedianHeight<<","<<weightedMedianWidth<<","<<vegetationVolume<<","<<VVI<<","/*<<chullArea<<","*/;
			result<<leafArea<<","<<LAI<<","<<Centroid2HeightRatio<<","<<projectOccupancy[1]<<","<<numValidSubBox<<","<<numSubBoxUsed<<","<<getEasternTime()<<","<<slice_volume*1e-9;

			for(auto& v : sub_plant_volume_vec) result <<","<<v*1e-9;

			for(auto& a : sub_plant_chull_area_vec) result <<","<<a*1e-6;

			for(auto& h : three_hedge_widths) result<<","<<h;
			
			result <<std::endl;
				
			
			//pcViewer->addLine(soilPlaneLineProjection->at(0), soilPlaneLineProjection->at(1), "line_soil");
        		//pcViewer->addLine(soilPlaneLineProjection->at(2), soilPlaneLineProjection->at(3), "line_soil1");
			//pcViewer->addLine(plantPlaneLineProjection->at(0), plantPlaneLineProjection->at(1), "line_plant");
        		//pcViewer->addLine(plantPlaneLineProjection->at(2), plantPlaneLineProjection->at(3), "line_plant1");

			// vegetation bounding box
			//pcViewer->addCube (min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 1.0, 1.0, 1.0, "AABB");

			pcViewer->addLine(rootPoint1, rootPoint2, 0.0, 0.0, 1.0, "root line", 0);

			//pcViewer->addPointCloud(pc_rootLinePoints, "root line points", 0);
			
			pcl::PointXYZ p1(min_point_AABB.x, min_point_AABB.y, min_point_AABB.z);
			pcl::PointXYZ p7(max_point_AABB.x, max_point_AABB.y, max_point_AABB.z);				
			pcl::PointXYZ p2 = p1; p2.y = p7.y;
			pcl::PointXYZ p3 = p2; p3.z = p7.z;
			pcl::PointXYZ p4 = p1; p4.z = p7.z;
			pcl::PointXYZ p5 = p1; p5.x = p7.x;
			pcl::PointXYZ p6 = p5; p6.y = p7.y;
			pcl::PointXYZ p8 = p7; p8.y = p1.y;
			
			pcViewer->addLine(p1, p2, 1.0, 1.0, 1.0, "line1", 0);
			pcViewer->addLine(p2, p3, 1.0, 1.0, 1.0, "line2", 0);
			pcViewer->addLine(p3, p4, 1.0, 1.0, 1.0, "line3", 0);
			pcViewer->addLine(p4, p1, 1.0, 1.0, 1.0, "line4", 0);
			pcViewer->addLine(p5, p6, 1.0, 1.0, 1.0, "line5", 0);
			pcViewer->addLine(p6, p7, 1.0, 1.0, 1.0, "line6", 0);
			pcViewer->addLine(p7, p8, 1.0, 1.0, 1.0, "line7", 0);
			pcViewer->addLine(p8, p5, 1.0, 1.0, 1.0, "line8", 0);
			pcViewer->addLine(p1, p5, 1.0, 1.0, 1.0, "line9", 0);
			pcViewer->addLine(p2, p6, 1.0, 1.0, 1.0, "line10", 0);
			pcViewer->addLine(p3, p7, 1.0, 1.0, 1.0, "line11", 0);
			pcViewer->addLine(p4, p8, 1.0, 1.0, 1.0, "line12", 0);

			//low width line end points
			float height_step = (max_point_AABB.x - min_point_AABB.x)/3.f;

			for(int width_line_id=0; width_line_id<3; width_line_id++)
			{
				const float x = min_point_AABB.x + (width_line_id+0.5f)*height_step;

				pcl::PointXYZ p_0(x ,max_point_AABB.y, max_point_AABB.z);

				pcl::PointXYZ p_1(x ,max_point_AABB.y, max_point_AABB.z - three_hedge_widths[width_line_id]);

				pcViewer->addLine(p_0, p_1, 0.0, 1.0, 0.0, "width"+std::to_string(width_line_id), 0);
			}
				

			//visualize point cloud		
			//pcViewer->addPointCloud(pcl_pcVec[0], "pc0", 0);
			//if(pcl_pcVec.size()>1)
			//pcViewer->addPointCloud(pcl_pcVec[1], "pc1", 0);
			//if(pcl_pcVec.size()>2)
			//pcViewer->addPointCloud(pcl_pcVec[2], "pc2", 0);
			//pcViewer->addPointCloud(pc_inlierOfSoilPlane, "soil", 0);
			//pcViewer->addPointCloud(pc_outsideAABB, "outside AABB", 0);
			//pcViewer->addPointCloud(pc_upperHalf, "upperHalf", 0);
			pcViewer->addPointCloud(mls_points, "mls", 0);

			
			
			// draw convex hull
			//pcViewer->addPolygonMesh<pcl::PointXYZRGB>(cloud_hull, vertices_chull, "convex hull", 0);
			pcViewer->addPointCloud(pc_plantOutliers, "plant outliers", 0);
			pcViewer->addPointCloud(pc_outsideAABB, "outsider", 0);
			//pcViewer->setRepresentationToWireframeForAllActors(); 	

			
			pcViewer->setCameraPosition(399, 855, -1012, 1,0,0, 0);
			pcViewer->resetCamera();

			pcViewer->spinOnce(pcl_view_time);

			for(int i=0; i<cloudHullVec.size(); i++)
				pcViewer->addPolygonMesh<pcl::PointXYZRGB>(cloudHullVec[i], verticesChullVec[i], "convex hull"+std::to_string(i), 0);
			
			pcViewer->spinOnce(pcl_view_time);

			pcViewer->removePointCloud("mls", 0);
			pcViewer->removePointCloud("plant outliers", 0);
			pcViewer->removePointCloud("outsider", 0);
			pcViewer->removePointCloud("soil", 0);
			pcViewer->removePointCloud("plant project1", 0);
			pcViewer->removePointCloud("root line", 0);

			for(int i=0; i<cloudHullVec.size(); i++)
				pcViewer->removePolygonMesh("convex hull"+std::to_string(i), 0);

			pcViewer->addPolygonMesh(triangles,"meshes",0);

			pcViewer->spinOnce(pcl_view_time);
#endif
		}
	}

	cv::destroyAllWindows();
	fs.release();
	result.close();

	return 0;
}
