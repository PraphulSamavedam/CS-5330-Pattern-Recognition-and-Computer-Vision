/**
Written by: Samavedam Manikhanta Praphul
			Poorna Chandra Vemula
Version: 1.0
Description: This file has helper/utility functions required for the extensions.
*/
#include <opencv2/opencv.hpp>
#include "../include/tasks.h"

/** This function solves the translations and rotations of the camera position if the camera
* This function will be internally called in getCameraPoseAndDrawVirtualObject.
* @param cornersPts the image points of the chessboard in the image.
* @param cameraMatrix intrinsic parameters of the camera in cv::Mat of CV_64FC1 format.
* @param distortionCoefficients coefficients of distortion in the lens.
* @param rVector output computed vector corresponding to the orientation of the camera.
* @param tVector output computed vector corresponding to the position of the camera.
*/
bool __getCameraPosition(std::vector<cv::Point2f>& cornerPts, cv::Mat& cameraMatrix, 
	std::vector<float>& distortionCoefficients, cv::Mat& rVector, cv::Mat& tVector);

/** This function solves for the camera pose and then projects the virtual object selected on the image provided.
* @param image cv::Mat object on which the virtual object has to be projected.
* @param cornerPts image points corresponding to the internal corners of the chessboard.
* @param cameraMatrix intrinsic parameters of the camera in cv::Mat of CV_64FC1 format.
* @param distortionCoefficients coefficients of distortion in the lens.
* @param virtual_object char corresponding to intended object.
*/
bool getCameraPoseAndDrawVirtualObject(cv::Mat& image, std::vector<cv::Point2f>& cornerPts,
	cv::Mat& cameraMatrix, std::vector<float>& distortionCoefficients, char virtual_object);