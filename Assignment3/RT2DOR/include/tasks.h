/**
* Written by: Samavedam Manikhanta Praphul
* This file provides the signatures of several functions required in the project.
*/

#include <opencv2/opencv.hpp>

/* This image is thresholded in the range of hueMin, hueMax; satMin, satMax;
and valMin , valMax; from the source image.
*/
int thresholdImage(cv::Mat& srcImg, int hueMin, int hueMax, int satMin, int satMax, int valMin, int valMax, cv::Mat& thresholdedImg);

/*This function masks the source image by marking the image above threshold as black below as white.
*/
int thresholdImage(cv::Mat& srcImg, cv::Mat& thresholdedImg, int greyScaleThreshold);

/** This function returns only the fileName from the filePath provided.
@param filePath path of the file whose name needs to be obtained.
@param fileName placeholder for result.
@return 0 for successfully obtaining the fileName.
@note Assumes that the filePath is valid (doesn't validate filePath)
	  Method: Parses the filePath to find the last folder separator like '/' or '\\' and
	  populates from that index to end.
*/
int getOnlyFileName(char*& filePath, char*& fileName);

/**  This function does the Grass Fire transformation to obtain the distance of the pixel from the background.
* Assumes the foreground to be white (255), background to be black (0)
* @param srcImg address of the source binary image
* @param dstImg address of the destination image which needs to have distance values
* @param connectValue[default=4] set value as 4 or 8 to mark 4-connected, 8-connected technique
* @param foreGround[default=255] value of the foreground pixel value.
* @param backGround[default=0] value of the background pixel value.
* @returns 0 if the values are computed completed.
* @note AssertionError if connectValue not in (4,8)
*		AssertionError if foreGround or backGround values are not in range [0,255].
*/
int grassFireAlgorithm(cv::Mat& srcImg, cv::Mat& dstimg, int connectValue = 4, int foreGround = 255, int backGround = 0);

/** This function makes a foreground pixel into a background pixel based on the connect method chosen.
* Assumes the foreground to be white (255), background to be black (0)
* @param srcImg address of the source binary image
* @param erodedimg address of the destination binary image
* @param numberOftimes times the erosion operation needs to be performed.
* @param connectValue[default=4] set value as 4 or 8 to mark 4-connected, 8-connected technique
* @param foreGround[default=255] value of the foreground pixel value.
* @param backGround[default=0] value of the background pixel value.
* @returns 0 if the erosion is success.
* @note This function internally uses grassFireAlgorithm to obtain the number of erosions required to erode the specific pixel.
*		AssertionError if connectValue not in (4,8)
*		AssertionError if foreGround or backGround values are not in range [0,255].
*/
int erosion(cv::Mat& srcImg, cv::Mat& erodedImg, int numberOfTimes, int connectValue = 4, int foreGround = 255, int backGround = 0);

/** This function makes a background pixel into a foreground pixel based on the connect method chosen.
* Assumes the foreground to be white (255), background to be black (0)
* @param srcImg address of the source binary image
* @param erodedimg address of the destination binary image
* @param numberOftimes times the dilation operation needs to be performed.
* @param connectValue[default=4] set value as 4 or 8 to mark 4-connected, 8-connected technique
* @param foreGround[default=255] value of the foreground pixel value.
* @param backGround[default=0] value of the background pixel value.
* @returns 0 if the dilusions is success.
* @note This function internally uses grassFireAlgorithm
*		AssertionError if connectValue not in (4,8)
*		AssertionError if foreGround or backGround values are not in range [0,255].
*/
int dilation(cv::Mat& srcImg, cv::Mat& dilatedImg, int numberOfTimes, int connectValue = 4, int foreGround = 255, int backGround = 0);

/** This function find the conencted foreground regions in a binary image using stack.
* Assumes the foreground to be white (255), background color as 255 - foreGround.
* @param srcImg address of the source binary image
* @param dstImg address of the destination binary image
* @param connectValue[default=4] set value as 4 or 8 to mark 4-connected, 8-connected technique
* @param foreGround[default=255] value of the foreground pixel value.
* @returns 0 if the segmentation is successful.
* @note AssertionError if connectValue not in (4,8)
*		AssertionError if foreGround or backGround values are not in exactly 0 or 255.
*/
int segmentationStack(cv::Mat& srcImg, cv::Mat& dstImg, int connectValue = 4,int foreGround = 255);