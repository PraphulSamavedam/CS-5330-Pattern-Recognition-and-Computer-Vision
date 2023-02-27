/*
* Written by : Samavedam Manikhanta Praphul
* This file provides the required images for the static file without label.
*/

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <opencv2/opencv.hpp>
#include "../include/utils.h"
#include "../include/match_utils.h"


int main(int argc, char* argv[])
{
	if (argc<2)
	{
		printf("Missing required arguments.\n");
		printf("Usage: %s <filePath>\n", argv[0]);
		exit(-404);
	}

	// Get the image path from input
	char filePath[1024];
	std::strcpy(filePath, argv[1]);

	// Read the image from the file path
	cv::Mat image = cv::imread(filePath);
	if (image.data == NULL)
	{
		printf("Fatal Error file not found"); 
		exit(-100);
	}
	std::vector<float> featureVector;
	cv::imwrite("Original Image.jpg", image);
	getFeaturesForImage(image, featureVector, 124, 3, 4, 8, 1, true, true, true);
	return 0;

}
