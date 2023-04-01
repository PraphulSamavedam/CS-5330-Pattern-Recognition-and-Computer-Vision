#ifndef MARKER_DETECT_H
#define MARKER_DETECT_H
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <opencv2/objdetect/objdetect.hpp>


class marker
{
	private:
	int size;
	cv::Mat calib_frame;
	double ThreshParam1, ThreshParam2;
	float probDetect,  markersize;
	cv::aruco::Board boardconfig;
	cv::aruco::ArucoDetector boarddetect;
	cv::aruco::DetectorParameters camparams;
	std::vector< cv::Point3f > object_points;
	std::vector< cv::Point2f > image_points;

public:
	cv::Mat  rvecs, tvecs;
	bool detect_flag;
	marker(char**, cv::Mat&, cv::Mat&);
	void marker_detect(cv::Mat);
}; 

#endif