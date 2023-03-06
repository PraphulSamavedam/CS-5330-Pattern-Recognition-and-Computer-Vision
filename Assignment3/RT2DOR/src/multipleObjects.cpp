/*
* Written by : Samavedam Manikhanta Praphul
* This file predicts labels for multiple objects in the stage image provided. 
*/

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <opencv2/opencv.hpp>
#include "../include/utils.h"
#include "../include/match_utils.h"


int main(int argc, char* argv[])
{
	int grayscaleThreshold = 124;
	bool debug = true;
	bool displayIntermediateImages = true;
	int windowSize = cv::WINDOW_NORMAL;
	int numberOfErosions = 4;
	int erosionConnectValue = 4;
	int dilationConnectValue = 8;
	int numberOfSegments = 1000;

	if (argc < 4)
	{
		printf("Missing required arguments.\n");
		printf("Usage: %s <filePath> <Features&LabelsFile> <distanceMetric> <n[default=1]>\n", argv[0]);
		exit(-404);
	}

	// Get the image path from input
	char filePath[1024];
	std::strcpy(filePath, argv[1]);

	// Get the features file path from input
	char featureVectorFile[1024];
	std::strcpy(featureVectorFile, argv[2]);

	// Get the distance metric to be used.
	char distanceMetric[16];
	std::strcpy(distanceMetric, argv[3]);

	int KNearest = 1;
	if (argc >= 4)
	{
		KNearest = atoi(argv[4]);
	}


	// Read the image from the file path
	cv::Mat image = cv::imread(filePath);
	if (image.data == NULL)
	{
		printf("Fatal Error file not found");
		exit(-100);
	}

	// Thresholding based on the grayscale value above threshold using function 
	cv::Mat binaryImg;
	thresholdImage(image, binaryImg, grayscaleThreshold);
	if (debug) { printf("Thresholded greyscale image to obtain binary image.\n"); }
	if (displayIntermediateImages) {
		cv::namedWindow("Binary Image", windowSize);
		cv::imshow("Binary Image", binaryImg);
	}

	// Morphological operations to clean the image

	// Dilation of binary image
	cv::Mat dilatedImg;
	dilation(binaryImg, dilatedImg, numberOfErosions, dilationConnectValue);
	if (debug) {
		printf("Dilated binary image %d times following %d-connected technique\n"
			, numberOfErosions, dilationConnectValue);
	}

	// Erosion of binary image
	cv::Mat erodedImg;
	erosion(dilatedImg, erodedImg, numberOfErosions, erosionConnectValue);
	if (debug) {
		printf("Erroded binary image %d times following %d-connected technique\n"
			, numberOfErosions, erosionConnectValue);
	}

	cv::Mat cleanImg;
	erodedImg.copyTo(cleanImg);

	// Cleaning of the binary image is complete.
	if (displayIntermediateImages) {
		cv::namedWindow("Cleaned Image", windowSize);
		cv::imshow("Cleaned Image", cleanImg);
	}

	// Segment the detected foreground pixels into regions. 
	cv::Mat regionMap = cv::Mat::zeros(cleanImg.size(), CV_32SC1);
	regionGrowing(cleanImg, regionMap, 8);

	// Restrict the segmentation to top N regions only.
	cv::Mat segImg = cv::Mat::zeros(cleanImg.size(), CV_8UC1);
	int segments = topNSegments(true, regionMap, segImg, numberOfSegments, false);
	cv::namedWindow("Segmented Image", windowSize);
	cv::imshow("Segmented Image", segImg);

	// Color the detected Segments
	cv::Mat segmentColoredImg = cv::Mat::zeros(cleanImg.size(), CV_32SC3);
	colorSegmentation(regionMap, segmentColoredImg);
	if (debug) { printf("Colored the segmented image having %d segments.\n", segments); }
	if (displayIntermediateImages) {
		cv::namedWindow("Colored Segmented Image", windowSize);
		cv::imshow("Colored Segmented Image", segmentColoredImg);
	}

	// Predict the image based on the region map
	cv::Mat boundedImage;
	image.copyTo(boundedImage);
	drawBoundingBoxes(regionMap, boundedImage, segments, false);
	if (debug) { printf("Bounded boxes drawn.\n"); }
	if (displayIntermediateImages) {
		cv::namedWindow("Image with Bounding boxes", windowSize);
		cv::imshow("Image with Bounding boxes", boundedImage);
	}

	std::vector<float> featureVector;
	getFeatures(regionMap, featureVector, segments);
	int lenOfRegSpecificVec = featureVector.size() / segments;
	for (int i = 0; i < segments; i++)
	{
		std::vector<float> regionSpecificFtVector(
			featureVector.begin() + i,
			featureVector.begin() + i + lenOfRegSpecificVec + 1);
		char prediction[256];
		ComputingNearestLabelUsingKNN(regionSpecificFtVector, featureVectorFile, distanceMetric, prediction, KNearest);
		placeLabel(boundedImage, prediction, 8, 10);
	}
	if (false) {
		cv::namedWindow("Multiple Predictions", windowSize);
		cv::imwrite("Multiple Predictions.jpg", boundedImage);
	}
	cv::waitKey(0);
	return 0;

}
