#ifndef __PATCHMATCHSTEREO_GPU_H__
#define __PATCHMATCHSTEREO_GPU_H__
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

extern "C" void PatchMatchStereoGPU(const cv::Mat& leftImg, const cv::Mat& rightImg, int winRadius, int Dmin, int Dmax, int iteration, float scale, bool showLeftDisp, cv::Mat& leftDisp, cv::Mat& rightDisp);

extern "C" void PatchMatchStereoHuberGPU(const cv::Mat& leftImg, const cv::Mat& rightImg, int winRadius, int Dmin, int Dmax, int iteration, float scale, bool showLeftDisp, cv::Mat& leftDisp, cv::Mat& rightDisp);


#endif
