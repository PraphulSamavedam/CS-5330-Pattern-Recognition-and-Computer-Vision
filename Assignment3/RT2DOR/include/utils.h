/*
* Written by : Samavedam Manikhanta Praphul
* This file has the function which process an image for its feature vector. 
*/

#define _CRT_SECURE_NO_WARNINGS
//#include "../include/RT2DOR.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "../include/tasks.h"
#include <string>
#include "../include/readfiles.h" // To get the list of the files in the directory
#include "../include/csv_util.h" // To work with csv file of feature vectors

/** This function process the given image for the feature vector internally
	It internally uses the functions in tasks.h file.
*/
int getFeaturesForImage(cv::Mat& image, std::vector<float>& featureVector,
	int greyscaleThreshold = 124, int numberOfErosions = 5,
	int erosionConnectValue = 4, int dilationConnectValue = 8, int numberOfSegments = 1,
	bool debug = false, bool displayIntermediateImages = false);

/** This function returns only the fileName from the filePath provided.
@param filePath path of the file whose name needs to be obtained.
@param fileName placeholder for result.
@param label placeholder for the label read.
@return 0 for successfully obtaining the fileName.
@note Assumes that the filePath is valid (doesn't validate filePath)
	  Method: Parses the filePath to find the last folder separator like '/' or '\\' and
	  populates from that index to end.
*/
int getFileNameAndLabel(char*& filePath, char*& fileName, char*& label);

