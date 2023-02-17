/*
* Written by : Samavedam Manikhanta Praphul
* This file defines the entry point for the application -- RT2DObjRecognition.cpp
*/

#define _CRT_SECURE_NO_WARNINGS
//#include "../include/RT2DObjRecognition.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "../include/tasks.h"
#include <string>

using namespace std;

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

	// Displaying the image read.
	//cv::imshow("Original Image", image);

	// Remove any salt and pepper noise from the image.
	cv::Mat noSnPImg;
	cv::medianBlur(image, noSnPImg, 5);

	// Blur the image to smoothen the edges
	cv::Mat blurImg;
	cv::GaussianBlur(noSnPImg, blurImg, cv::Size(7, 7), 0.1);

	// Convert  to HSV for using cv function
	cv::Mat hsvImg;
	image.copyTo(hsvImg);
	cv::cvtColor(blurImg, hsvImg, cv::COLOR_BGR2HSV);
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
	//cv::imshow("Thresholded HSV Blur Image", maskedImg2);

	// Thresholding based on the grayscale value above threshold using function from 
	cv::Mat grayThImg;
	int grayscaleThreshold = 124; // Value is based on the experimentation with sample images
	thresholdImage(image, grayThImg, grayscaleThreshold);
    cv::imshow("Thresholded Grayscale Image", grayThImg);

	cv::Mat grayThImg2;
	thresholdImage(blurImg, grayThImg2, grayscaleThreshold);
	//cv::imshow("Thresholded Grayscale Blur Image", grayThImg2);
	

	cv::Mat erodeImg;
	cv::erode(grayThImg, erodeImg, cv::Mat());
	cv::imshow("CV::Erode Image", erodeImg);

	// Erosion of binary image
	cv::Mat erroredImage;
	erosion(grayThImg, erroredImage, 1, 4);
	cv::imshow("Eroded Image", erroredImage);

	cv::Mat dilImg;
	cv::dilate(grayThImg, dilImg, cv::Mat());
	cv::imshow("CV::Dilated Image", dilImg);

	// Dilation of binary image
	cv::Mat dilatedImage;
	dilation(grayThImg, dilatedImage, 1, 8);
	cv::imshow("Dilated Image", dilatedImage);

	// Revert the erosion with same connect
	cv::Mat nochangeImg;
	dilation(erroredImage, nochangeImg, 1, 8);
	cv::imshow("No Changes Image", nochangeImg);

	// Proper clean up of the image.
	cv::Mat cleanedImg;
	dilation(erroredImage, cleanedImg, 1, 4);
	cv::imshow("CV::Erode Image", cleanedImg);

	

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
