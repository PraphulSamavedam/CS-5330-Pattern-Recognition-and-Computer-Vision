/* This file holds the function signatures for the extensions. Documentation is provided in the source file.
Written by: Samavedam Manikhanta Praphul. */
#include <opencv2/opencv.hpp> // For cv::Mat definition

int PositiveLaplacianFilter(cv::Mat& src, cv::Mat& dst);

int NegativeLaplacianFilter(cv::Mat& src, cv::Mat& dst);

int boxBlur3x3(cv::Mat& src, cv::Mat& dst);

int medianFilter3x3(cv::Mat& src, cv::Mat& dst);

int boxBlur3x3Brute(cv::Mat& src, cv::Mat& dst);

int boxBlur5x5(cv::Mat& src, cv::Mat& dst);

int boxBlur5x5Brute(cv::Mat& src, cv::Mat& dst);

int gaussianLaplacian(cv::Mat& src, cv::Mat& dst);