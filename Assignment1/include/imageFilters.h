#include <opencv2/opencv.hpp>

int blur5x5(cv::Mat& src, cv::Mat& dst);
int blur5x5_brute(cv::Mat& src, cv::Mat& dst);
int sobelX3x3(cv::Mat& src, cv::Mat& dst);
int sobelY3x3(cv::Mat& src, cv::Mat& dst);
int absolute_image(cv::Mat& src, cv::Mat& dst);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat& dst);

//int sobelX3x3_brute(cv::Mat& src, cv::Mat& dst);

int blurQuantize(cv::Mat& src, cv::Mat& dst, int levels);
int quantize(cv::Mat& src, cv::Mat& dst, int levels);
int cartoon(cv::Mat& src, cv::Mat& dst, int levels, int magThreshold);

//Some additional functionalities
int negativeImage(cv::Mat& src, cv::Mat& dst);
int 