/*
* Written by : Samavedam Manikhanta Praphul
* This file has the function which process an image for its feature vector. 
* This file represents the common functionality in many programs developed in this code. 
*/

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <opencv2/opencv.hpp>
#include "../include/tasks.h"
#include <string>
#include "../include/readfiles.h" // To get the list of the files in the directory
#include "../include/csv_util.h" // To work with csv file of feature vectors



/** This function obtains the features in the source image by applying converting into binary based
on the grayscale threshold. The binary image is then cleaned based on params of cleansing like number of times,
which connection to use. Connected component analysis is then run to get top N segments in the image based on area.

@param image source address of color image.
@param featureVector address of the feature vector to be populated.
@param grayscaleThreshold[default=124] to threshold the grayscale image.
@param erosionConnectValue[default=4] only 4/8 for 4 connected technique or 8 connected technique.
@param dilationConnectValue[default=8] only 4/8 for 4 connected technique or 8 connected technique.
@param numberOfSegments[default=1] set this number to desired number of segments required in the image.
@param debug[defaul=false] set this to enable verbose.
@param displayIntermediateImages set this to have display of intermediate results.


@note This function process the given image for the feature vector internally.
	  It internally uses the functions in tasks.h file.
*/
/** 
*/
int getFeaturesForImage(cv::Mat& image, std::vector<float>& featureVector, bool debug = false, 
	bool displayIntermediateImages = false, bool saveImages = false, int numberOfSegments = 1,
	int greyscaleThreshold = 124, int numberOfErosions = 4,
	int erosionConnectValue = 4, int dilationConnectValue = 8);

/** This function returns the fileName and label from the filePath provided.
@param filePath path of the file whose name needs to be obtained.
@param fileName placeholder for result.
@param label placeholder for the label read.
@return 0 for successfully obtaining the fileName.
@note Assumes that the filePath is valid (doesn't validate filePath)
	  Method: Parses the filePath to find the last folder separator like '/' or '\\' and
	  populates from that index to end.
*/
int getFileNameAndLabel(char*& filePath, char*& fileName, char*& label);

