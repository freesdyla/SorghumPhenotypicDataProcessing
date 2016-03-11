#include "PreprocessingTools.h"

PreprocessTools::PreprocessTools()
{
	//R=right side camera, left side plant
	PGCamParamFolderVec.push_back("/LB");
	PGCamParamFolderVec.push_back("/LBM");
	PGCamParamFolderVec.push_back("/LM");
	PGCamParamFolderVec.push_back("/LMT");
	PGCamParamFolderVec.push_back("/LT");
	PGCamParamFolderVec.push_back("/RB");
	PGCamParamFolderVec.push_back("/RBM");
	PGCamParamFolderVec.push_back("/RM");
	PGCamParamFolderVec.push_back("/RMT");
	PGCamParamFolderVec.push_back("/RT");
	
}


PreprocessTools::StereoCameraParam::StereoCameraParam(cv::Mat M1, cv::Mat D1, cv::Mat M2, cv::Mat D2, cv::Mat R, cv::Mat T,
	cv::Mat R1, cv::Mat P1, cv::Mat R2, cv::Mat P2, cv::Mat Q)
{
	setAllParams(M1, D1, M2, D2, R, T, R1, P1, R2, P2, Q);
}


void PreprocessTools::StereoCameraParam::setAllParams(cv::Mat M1, cv::Mat D1, cv::Mat M2, cv::Mat D2, cv::Mat R, cv::Mat T,
	cv::Mat R1, cv::Mat P1, cv::Mat R2, cv::Mat P2, cv::Mat Q)
{
	M1.copyTo(_M1);
	D1.copyTo(_D1);
	M2.copyTo(_M2);
	D2.copyTo(_D2);
	R.copyTo(_R);
	T.copyTo(_T);
	R1.copyTo(_R1);
	P1.copyTo(_P1);
	R2.copyTo(_R2);
	P2.copyTo(_P2);
	Q.copyTo(_Q);
}

void PreprocessTools::StereoCameraParam::setExtrinsics(cv::Mat R, cv::Mat T)
{
	R.copyTo(_R);
	T.copyTo(_T);
}

cv::Mat PreprocessTools::equalizeIntensity(const cv::Mat& inputImage)
{
	if (inputImage.channels() >= 3)
	{
		std::vector<cv::Mat> channels;
		split(inputImage, channels);

		equalizeHist(channels[0], channels[0]);
		equalizeHist(channels[1], channels[1]);
		equalizeHist(channels[2], channels[2]);

		cv::Mat result;
		merge(channels, result);

		return result;
	}

	return cv::Mat();
}

bool PreprocessTools::loadStereoHeadParamFile(std::string folderPath)
{
	cv::FileStorage fs;

	if (fs.open(folderPath + "/intrinsics.yml", CV_STORAGE_READ))
	{
		fs["M1"] >> M1;
		fs["D1"] >> D1;
		fs["M2"] >> M2;
		fs["D2"] >> D2;

		if (fs.open(folderPath + "/extrinsics.yml", CV_STORAGE_READ))
		{
			fs["R"] >> R;
			fs["T"] >> T;
			fs["R1"] >> R1;
			fs["R2"] >> R2;
			fs["P1"] >> P1;
			fs["P2"] >> P2;
			fs["Q"] >> Q;
		}
		else
		{
			std::cout << "extrinsics.yml missing!" << std::endl;
			return false;
		}
	}
	else
	{
		std::cout << "intrinsics.yml missing!" << std::endl;
		return false;
	}

	paramLoaded = true;

	return true;
}

// assume working direction contains folder CameraParam2013, 2014, 2015
bool PreprocessTools::loadAllStereoArrayParam(int cameraType, std::string folderPath)
{
	if (cameraType == PG)
	{
		for (int i = 0; i < PGCamParamFolderVec.size(); i++)
		{

			if (loadStereoHeadParamFile(folderPath + PGCamParamFolderVec[i]))
			{
				stereoArrayParamVec.push_back(StereoCameraParam(M1, D1, M2, D2, R, T, R1, P1, R2, P2, Q));
			}
			else
			{
				std::cout << PGCamParamFolderVec[i] + " has no .yml file!" << std::endl;
				return false;
			}
		}
	}
	else if (cameraType == CANON)
	{

	}

	return true;
}

cv::Mat PreprocessTools::loadBGRImgBasedOnCameraType(int cameraType, std::string fileName)
{
	cv::Mat colorImg;

	if (cameraType == PG)
	{
		// load .pgm file
		cv::Mat grayImg = cv::imread(fileName, cv::IMREAD_GRAYSCALE);

		if (grayImg.data == NULL)
			return colorImg;

		cvtColor(grayImg, colorImg, CV_BayerBG2BGR);
	}
	else if (cameraType == CANON)
	{
		// load .jpg file
		colorImg = cv::imread(fileName, cv::IMREAD_COLOR);
	}

	return colorImg;
}

bool PreprocessTools::loadPGStereoPairs(int cameraType, int plantSide, std::string prefixFileName, std::vector<cv::Mat>& stereoPairVec)
{
	stereoPairVec.clear();

	if (cameraType == PG)
	{
		if (plantSide == RIGHT)
		{
			for (int i = 0; i < 6; i++)
			{
				cv::Mat img = loadBGRImgBasedOnCameraType(PG, prefixFileName + std::to_string(i) + ".pgm");

				if (img.data != NULL)
				{
					stereoPairVec.push_back(img);
				}
				else
					return false;
			}
		}
		else if (plantSide == LEFT)
		{
			for (int i = 6; i < 12; i++)
			{
				cv::Mat img = loadBGRImgBasedOnCameraType(PG, prefixFileName + std::to_string(i) + ".pgm");

				if (img.data != NULL)
				{
					stereoPairVec.push_back(img);
				}
				else
					return false;
			}
		}
	}

	if (stereoPairVec.size() != 0)
		return true;
	else
		return false;
}

// does not clear outStereoPairVec
bool PreprocessTools::rectifyStereoPair(int cameraType, int stereoHeadIdx, cv::Mat leftCImg, cv::Mat rightCImg, std::vector<cv::Mat>& outStereoPairVec, bool showImage = false, bool histEq = false, double scale = 0.5)
{
	if (leftCImg.data == NULL || rightCImg.data == NULL )
	{
		std::cout << "data null" << std::endl;
		return false;
	}

	if (leftCImg.size() != rightCImg.size())
	{
		std::cout << "stereo pair size not equal!" << std::endl;
		return false;
	}

	cv::Mat M1 = stereoArrayParamVec[stereoHeadIdx]._M1;
	cv::Mat M2 = stereoArrayParamVec[stereoHeadIdx]._M2;
	cv::Mat D1 = stereoArrayParamVec[stereoHeadIdx]._D1;
	cv::Mat D2 = stereoArrayParamVec[stereoHeadIdx]._D2;
	cv::Mat R1 = stereoArrayParamVec[stereoHeadIdx]._R1;
	cv::Mat R2 = stereoArrayParamVec[stereoHeadIdx]._R2;
	cv::Mat P1 = stereoArrayParamVec[stereoHeadIdx]._P1;
	cv::Mat P2 = stereoArrayParamVec[stereoHeadIdx]._P2;

	cv::Mat map11, map12, map21, map22;
	cv::initUndistortRectifyMap(M1, D1, R1, P1, leftCImg.size(), CV_16SC2, map11, map12);
	cv::initUndistortRectifyMap(M2, D2, R2, P2, leftCImg.size(), CV_16SC2, map21, map22);

	cv::Mat img1r, img2r;
	cv::remap(leftCImg, img1r, map11, map12, cv::INTER_LINEAR);
	cv::remap(rightCImg, img2r, map21, map22, cv::INTER_LINEAR);

	cv::Mat img1rHE, img2rHE;

	if (histEq)
	{
		img1rHE = equalizeIntensity(img1r);
		img2rHE = equalizeIntensity(img2r);
	}
	else
	{
		img1rHE = img1r;
		img2rHE = img2r;
	}
	
	/*outStereoPairVec.push_back(img1rHE);
	outStereoPairVec.push_back(img2rHE);*/
	outStereoPairVec.push_back(img1r);
	outStereoPairVec.push_back(img2r);

	if (showImage)
	{
		cv::Mat canvas;
		int w, h;
		double sf;
		sf = scale;
		w = cvRound(leftCImg.cols*sf);
		h = cvRound(leftCImg.rows*sf);
		canvas.create(h, w * 2, CV_8UC3);

		cv::Mat canvasl = canvas(cv::Rect(0, 0, w, h));
		cv::Mat canvasr = canvas(cv::Rect(w, 0, w, h));
		resize(img1rHE, canvasl, canvasl.size(), 0, 0, CV_INTER_AREA);
		resize(img2rHE, canvasr, canvasr.size(), 0, 0, CV_INTER_AREA);

		for (int j = 0; j < canvas.rows; j += 16)
			cv::line(canvas, cv::Point(0, j), cv::Point(canvas.cols, j), cv::Scalar(0, 255, 0), 1, 8);

		cv::line(canvas, cv::Point(w, 0), cv::Point(w, h), cv::Scalar(0, 255, 0), 1, 8);
		cv::imshow("rectified", canvas);
		cv::waitKey(0);
	}

	return true;
}
