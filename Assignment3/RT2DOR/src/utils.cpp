/**
*   Written by: Samavedam Manikhanta Praphul
* This file has the function which process an image for its feature vector.
*/

#include "../include/utils.h"

int getFeaturesForImage(char* filePath, std::vector<float> &featureVector,
	int grayscaleThreshold, int numberOfErosions, int erosionConnectValue,
	int dilationConnectValue, int numberOfSegments, 
	bool debug, bool displayIntermediateImages)
{
	int windowSize = cv::WINDOW_KEEPRATIO;
	// Read the image from the file path
	cv::Mat image = cv::imread(filePath);
	if (image.data == NULL)
	{
		printf("Fatal Error file not found");
		exit(-404);
	}

	if (debug) { printf("Read the original image.\n"); }
	if (displayIntermediateImages) {
		// Displaying the image read.
		cv::namedWindow("Original Image", windowSize);
		cv::imshow("Original Image", image);
	}

	// Remove any salt and pepper noise from the image.
	cv::Mat noSnPImg;
	cv::medianBlur(image, noSnPImg, 5);
	if (debug) { printf("Removed salt and pepper noise.\n"); }

	// Blur the image to smoothen the edges
	cv::Mat blurredImg;
	cv::GaussianBlur(noSnPImg, blurredImg, cv::Size(7, 7), 0.1);
	if (debug) { printf("Blurred the image.\n"); }

	// Thresholding based on the grayscale value above threshold using function from 
	cv::Mat binaryImg;
	thresholdImage(blurredImg, binaryImg, grayscaleThreshold);
	if (debug) { printf("Thresholded greyscale image to obtain binary image.\n"); }
	if (displayIntermediateImages) {
		cv::namedWindow("Binary Image", windowSize);
		cv::imshow("Binary Image", binaryImg);
	}

	// Morphological operations to clean the image
	// Erosion of binary image
	cv::Mat erroredImage;
	erosion(binaryImg, erroredImage, numberOfErosions, erosionConnectValue);
	if (debug) { printf("Erroded the binary image %d times following %d-connected technique\n"
		, numberOfErosions, erosionConnectValue); }
	if (displayIntermediateImages)
	{
		cv::namedWindow("Erorded Image", windowSize);
		cv::imshow("Eroded Image", erroredImage);
	}

	// Dilation of binary image to complete cleaning the binary image
	cv::Mat cleanImg;
	dilation(erroredImage, cleanImg, numberOfErosions, dilationConnectValue);
	if (debug) {
		printf("Dilated the binary image %d times following %d-connected technique\n"
			, numberOfErosions, dilationConnectValue);
		printf("Cleaned the binary image.\n");
	}
	if (displayIntermediateImages) {
		cv::namedWindow("Cleaned Image", windowSize);
		cv::imshow("Cleaned Image", cleanImg);
	}

	// Segment the detected foreground pixels into regions. 
	cv::Mat regionMap = cv::Mat::zeros(cleanImg.size(), CV_32SC1);
	regionGrowing(cleanImg, regionMap, 8);

	// Restrict the segmentation to top N regions only.
	cv::Mat segImg = cv::Mat::zeros(cleanImg.size(), CV_8UC1);
	int segments = topNSegments(regionMap, segImg, numberOfSegments);
	if (debug) { printf("Segmented the binary image to have top %d regions.\n", segments); }
	if (displayIntermediateImages) {
		cv::namedWindow("Top N segmented binary Image", windowSize);
		cv::imshow("Top N segmented binary Image", segImg);
	}

	// Get the features of the segmented Image.
	getFeatures(regionMap, featureVector, segments);
	return 0;
}