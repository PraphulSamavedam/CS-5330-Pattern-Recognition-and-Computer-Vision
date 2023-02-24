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
int getFeaturesForImage(char* filePath, std::vector<float>& featureVector,
	int greyscaleThreshold = 124, int numberOfErosions = 1,
	int erosionConnectValue = 4, int dilationConnectValue = 8, int numberOfSegments = 3,
	bool debug = false, bool displayIntermediateImages = false);

