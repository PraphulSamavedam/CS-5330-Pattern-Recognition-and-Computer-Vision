/*
* Written by : Samavedam Manikhanta Praphul
* This file defines the entry point for the application -- RT2DOR.cpp
*/

#define _CRT_SECURE_NO_WARNINGS
//#include "../include/RT2DOR.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "../include/tasks.h"
#include <string>
#include "../include/readfiles.h" // To get the list of the files in the directory
#include "../include/csv_util.h" // To work with csv file of feature vectors


using namespace std;

int main(int argc, char* argv[])
{
	// Main configuration variables.
	int windowSize = cv::WINDOW_GUI_EXPANDED;
	int grayscaleThreshold = 124; // Value is based on the experimentation with sample images
	int numberOfErosions = 5;
	int numberOfSegments = 1;
	bool displaySteps = false;

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

	bool debug = true;
	if (debug) { printf("Read the original image.\n"); }

	if(displaySteps){
	// Displaying the image read.
	cv::namedWindow("Original Image", windowSize);
	cv::imshow("Original Image", image);
	}

	// Remove any salt and pepper noise from the image.
	cv::Mat noSnPImg;
	cv::medianBlur(image, noSnPImg, 5);
	if (debug){ printf("Removed salt and pepper noise.\n");}

	// Blur the image to smoothen the edges
	cv::Mat blurImg;
	cv::GaussianBlur(noSnPImg, blurImg, cv::Size(7, 7), 0.1);
	if (debug) { printf("Blurrred image.\n"); }

	// Convert  to HSV for using cv function
	cv::Mat hsvImg;
	image.copyTo(hsvImg);
	cv::cvtColor(blurImg, hsvImg, cv::COLOR_BGR2HSV);
	if (debug) { printf("Converted blurred image into HSV.\n"); }
	//cv::imshow("HSV Image", hsvImg);
	
	// HSV values are explored using the sample images provided.
	int hueMin = 0;
	int hueMax = 179; // Independent of this value as white background happens at high V and low sat
	int satMin = 0;
	int satMax = 42; // Low Saturation, satMax based on experimentation
	int valMin = 100; // High Value, valMin based on experimentation
	int valMax = 255; 

	cv::Scalar lowerBounds = cv::Scalar(hueMin, satMin, valMin);
	cv::Scalar upperBounds = cv::Scalar(hueMax, satMax, valMax);
	cv::Mat thresholdImg;
	cv::inRange(hsvImg, lowerBounds, upperBounds, thresholdImg);
	//cv::imshow("Thresholded CV's HSV Image", thresholdImg);


	// Thresholding based on HSV values using function from scratch
	cv::Mat maskedImg;
	image.copyTo(maskedImg);
	thresholdImage(image, hueMin, hueMax, satMin, satMax, valMin, valMax, maskedImg);
	//cv::imshow("Thresholded HSV Image", maskedImg);

	cv::Mat maskedImg2;
	image.copyTo(maskedImg2);
	thresholdImage(blurImg, hueMin, hueMax, satMin, satMax, valMin, valMax, maskedImg2);
	if (displaySteps)
	{
		cv::namedWindow("Thresholded HSV Blur Image", windowSize);
		cv::imshow("Thresholded HSV Blur Image", maskedImg2);
	}
	

	// Thresholding based on the grayscale value above threshold using function from 
	cv::Mat grayThImg;
	thresholdImage(image, grayThImg, grayscaleThreshold);
	if (displaySteps) {
		cv::namedWindow("Thresholded Grayscale Image", windowSize);
		cv::imshow("Thresholded Grayscale Image", grayThImg);
	}
	if (debug) { printf("Thresholded greyscale image.\n"); }

	cv::Mat grayThImg2;
	thresholdImage(blurImg, grayThImg2, grayscaleThreshold);
	//cv::imshow("Thresholded Grayscale Blur Image", grayThImg2);

	// Erosion of binary image
	cv::Mat erroredImage;
	erosion(grayThImg, erroredImage, numberOfErosions, 4);
	if (displaySteps)
	{
		cv::namedWindow("Erorded Image", windowSize);
		cv::imshow("Eroded Image", erroredImage);
	}

	// Dilation of binary image
	cv::Mat cleanImg;
	dilation(erroredImage, cleanImg, numberOfErosions, 8);
	if (debug) { printf("Cleaned the binary image.\n"); }
	if (displaySteps) {
		cv::namedWindow("Clearned Image", windowSize);
		cv::imshow("Clearned Image", cleanImg);
	}

	/* Connected component analysis to find regions
	cv::Mat labels = cv::Mat(cleanImg.size(), CV_8UC1);
	cv::Mat stats = cv::Mat(cleanImg.size(), CV_8UC1);
	cv::Mat centroids = cv::Mat(cleanImg.size(), CV_8UC1);
	cv::connectedComponentsWithStats(cleanImg, labels, stats, centroids);
	*/

	// Segment the detected foreground pixels into regions. 
	cv::Mat regionMap = cv::Mat::zeros(cleanImg.size(), CV_32SC1);
	regionGrowing(cleanImg, regionMap,8);

	// Restrict the segmentation to top N regions only.
	cv::Mat segImg = cv::Mat::zeros(cleanImg.size(), CV_8UC1);
	
	int segments = topNSegments(regionMap, segImg, 1);
	if (debug) { printf("Segmented the binary image to have top %d regions.\n", segments); }
	if (displaySteps) {
	cv::namedWindow("Top N segmented Image", windowSize);
	cv::imshow("Top N segmented Image", segImg);
	}

	// Color the detected Segments
	cv::Mat segmentColoredImg = cv::Mat::zeros(cleanImg.size(), CV_32SC3);
	colorSegmentation(regionMap, segmentColoredImg);
	if(displaySteps){
	cv::namedWindow("Colored Segmented Image", windowSize);
	cv::imshow("Colored Segmented Image", segmentColoredImg);
	}

	cv::Mat ImgWithBoxes;
	image.copyTo(ImgWithBoxes);
	drawBoundingBoxes(regionMap, ImgWithBoxes, segments);
	printf("Size: %d %d", ImgWithBoxes.rows, ImgWithBoxes.cols);
	//cv::namedWindow("With Boxes", windowSize);
	//cv::imshow("With Boxes", ImgWithBoxes);

	char folderDir[500] = "C:/Users/Samavedam/Documents/Studies/MS-AI/CS 5330 Pattern Recognition and CVision/Prog Assignments/Assignment3/RT2DOR/data/Images";
	std::vector<char*> filesList;
	getFilesFromDirectory(folderDir, filesList, false);

	char csvFilePath[50] = "data/db/features.csv";

	for (int index = 0; index < filesList.size(); index++)
	{
		// Read the specific image from the file path
		cv::Mat imageRead = cv::imread(filesList[index]);
		if (imageRead.data == NULL)
		{
			printf("Fatal Error file not found");
			exit(-100);
		}

		// Remove any salt and pepper noise from the image.
		cv::Mat noSaltPepperNoiseImg;
		cv::medianBlur(imageRead, noSaltPepperNoiseImg, 5);

		// Blurred Image
		cv::Mat BlurredImg;
		cv::GaussianBlur(noSaltPepperNoiseImg, BlurredImg, cv::Size(7, 7), 0.1);

		// Binary Image
		cv::Mat BinaryImg;
		thresholdImage(BlurredImg, BinaryImg, grayscaleThreshold);

		// Clean the Binary Image
		cv::Mat erodedImg;
		erosion(BinaryImg, erodedImg, numberOfErosions, 4);
		cv::Mat cleanedImg;
		dilation(erodedImg, cleanedImg, numberOfErosions, 8);

		// Segment the detected foreground pixels into regions. 
		cv::Mat regMap = cv::Mat::zeros(cleanedImg.size(), CV_32SC1);
		regionGrowing(cleanedImg, regMap, 8);

		// Restrict the segmentation to top N regions only.
		cv::Mat segmentedImg = cv::Mat::zeros(cleanedImg.size(), CV_8UC1);
		int segments = topNSegments(regMap, segmentedImg, numberOfSegments);

		// Color the detected N Segments
		cv::Mat colorSegmentedImg = cv::Mat::zeros(cleanImg.size(), CV_32SC3);
		int segs = colorSegmentation(regionMap, colorSegmentedImg);
		
		
		// Bounding the boxes based on the segmentation map
		cv::Mat ImgWithBox;
		imageRead.copyTo(ImgWithBox); 
		// Default with original image to have the boxes to shown.
		drawBoundingBoxes(regMap, ImgWithBox, segs);

		cv::namedWindow("Image", windowSize);
		// Request for the label from the user
		char label[256];
		while (true)
		{
			cv::imshow("Image", ImgWithBox);
			cv::waitKey(0);
			printf("Enter the label for the displayed image:\n");
			std::cin >> label;
			if (label == "") {
				printf("Invalid Emtpy Label\n");
				continue;
			}
			cv::destroyAllWindows();
			break;
		}
		vector<float> featureVector;

		// Check if the csv file is already present or not
		getFeatures(regMap, featureVector, segs);

		// Write the feature vector to csv file
		if (index == 0)
		{	// First entry should override the file contents to start afresh
			append_image_data_csv(csvFilePath, filesList[index], label, featureVector, true);
		}
		else { // Append to the file as it was created afresh for the first image.
			append_image_data_csv(csvFilePath, filesList[index], label, featureVector, false);
		}
	}

	while (true) {
		char key = cv::waitKey(0);
		if (key == 'q')
		{
			cv::destroyAllWindows();
			break;
		}
		else if (key == 's')
		{
			cv::setBreakOnError(true);
			cv::imwrite("output/No Salt & Pepper Noise image.png", noSnPImg);
			cv::imwrite("output/Blurred image.png", blurImg);
			cv::imwrite("output/HSV image.png", hsvImg);
			cv::imwrite("output/CV Threshold HSV image.png", thresholdImg);
			cv::imwrite("output/Threshold HSV image.png", maskedImg);
			cv::imwrite("output/Threshold HSV Blur image.png", maskedImg2);
			cv::imwrite("output/Threshold Gray image.png", grayThImg);
			cv::imwrite("output/Threshold Gray Blur image.png", grayThImg2);
		}
	}
}
