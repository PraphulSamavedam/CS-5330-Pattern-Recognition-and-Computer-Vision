#include <opencv2/opencv.hpp>

int laplacian_filter(cv::Mat& src, cv::Mat& dst);

//int laplacian_filter2(cv::Mat& src, cv::Mat& dst);

int medianFilter3x3(cv::Mat& src, cv::Mat& dst);

int boxBlur3x3(cv::Mat& src, cv::Mat& dst);

int boxBlur3x3Brute(cv::Mat& src, cv::Mat& dst);

int boxBlur5x5(cv::Mat& src, cv::Mat& dst);

int boxBlur5x5Brute(cv::Mat& src, cv::Mat& dst);
