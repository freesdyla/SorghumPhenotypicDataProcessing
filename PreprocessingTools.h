#ifndef __PREPROCESSING_TOOLS_H
#define __PREPROCESSING_TOOLS_H

#include "Names.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <map>
//#include <Windows.h>


class PreprocessTools
{
public:
	cv::Mat M1, D1, M2, D2;
	cv::Mat R, T, R1, P1, R2, P2, Q;

	int curCameraType = -1;

	

	std::vector<std::string> PGCamParamFolderVec;

	struct StereoCameraParam
	{
		cv::Mat _M1, _D1, _M2, _D2;
		cv::Mat _R, _T, _R1, _P1, _R2, _P2, _Q;

		StereoCameraParam(cv::Mat M1, cv::Mat D1, cv::Mat M2, cv::Mat D2, cv::Mat R, cv::Mat T,
			cv::Mat R1, cv::Mat P1, cv::Mat R2, cv::Mat P2, cv::Mat Q);

		void setAllParams(cv::Mat M1, cv::Mat D1, cv::Mat M2, cv::Mat D2, cv::Mat R, cv::Mat T,
						  cv::Mat R1, cv::Mat P1, cv::Mat R2, cv::Mat P2, cv::Mat Q);
		
		void setExtrinsics(cv::Mat R, cv::Mat T);
	};

	// right bottom up then left bottom up
	std::vector<StereoCameraParam> stereoArrayParamVec;

	std::vector<cv::Mat> stereoPairVec;

	bool paramLoaded = false;

	PreprocessTools();

	cv::Mat equalizeIntensity(const cv::Mat& inputImage);

	bool loadStereoHeadParamFile(std::string folderPath);

	bool loadAllStereoArrayParam(int cameraType, std::string folderPath);

	cv::Mat loadBGRImgBasedOnCameraType(int cameraType, std::string fileName);

	bool loadPGStereoPairs(int cameraType, int plantSide, std::string prefixFileName, std::vector<cv::Mat>& stereoPairVec);


	bool rectifyStereoPair(int cameraType, int stereoHeadIdx, cv::Mat leftCImg, cv::Mat rightCImg, std::vector<cv::Mat>& outStereoPairVec, bool showImage, bool histEq, double scale);
};


#endif
