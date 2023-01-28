#include <opencv2/opencv.hpp>


int saveImage(cv::Mat& src, const char* effect);

int displayImage(bool displayStatus, cv::Mat& image, const char* windowName, bool scale = false);