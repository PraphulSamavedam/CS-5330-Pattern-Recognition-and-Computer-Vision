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
bool __getCameraPosition(std::vector<cv::Point2f>& cornerImgPts, cv::Mat& cameraMatrix, std::vector<float>& distortionCoefficients,
	cv::Mat& rVector, cv::Mat& tVector) {
	std::vector<cv::Vec3f> cornerObjPts;
	buildPointsSet(cornerImgPts, cornerObjPts);
	//printf("Built the points set of size %zd.", cornerObjPts.size());
	cv::solvePnP(cornerObjPts, cornerImgPts, cameraMatrix, distortionCoefficients, rVector, tVector);
}

/** This function solves for the camera pose and then projects the virtual object selected on the image provided.
* @param image cv::Mat object on which the virtual object has to be projected.
* @param cornerImgPts image points corresponding to the internal corners of the chessboard.
* @param cameraMatrix intrinsic parameters of the camera in cv::Mat of CV_64FC1 format.
* @param distortionCoefficients coefficients of distortion in the lens.
* @param virtual_object char corresponding to intended object.
*/
bool getCameraPoseAndDrawVirtualObject(cv::Mat& image, std::vector<cv::Point2f>& cornerPts, cv::Mat& cameraMatrix,
	std::vector<float>& distortionCoefficients, char virtual_object) {
	cv::Mat rVector;
	cv::Mat tVector;
	__getCameraPosition(cornerPts, cameraMatrix, distortionCoefficients, rVector, tVector);
	// Project virtual object
	std::vector<cv::Vec3f> ObjPts;
	buildVirtualObjectPoints(ObjPts, virtual_object);
	std::vector<cv::Vec2f> imgPts;
	cv::projectPoints(ObjPts, rVector, tVector, cameraMatrix, distortionCoefficients, imgPts);
	drawVirtualObject(image, imgPts, virtual_object);
}
