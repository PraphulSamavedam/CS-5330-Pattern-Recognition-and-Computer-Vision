/**
*   Written by: Samavedam Manikhanta Praphul
* This file has the function which process an image for its feature vector.
*/

#include "../include/utils.h"
#include <opencv2/opencv.hpp>

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

*/
int getFeaturesForImage(cv::Mat& image, std::vector<float>& featureVector,
	bool debug, bool displayIntermediateImages, bool saveImages, int numberOfSegments,
	int grayscaleThreshold, int numberOfErosions, int erosionConnectValue,
	int dilationConnectValue)
{
	int windowSize = cv::WINDOW_KEEPRATIO;

	if (displayIntermediateImages) {
		// Displaying the image read.
		cv::namedWindow("Original Image", windowSize);
		cv::imshow("Original Image", image);
	}

	/* This code didn't boost the performance of the system, so commenting out these steps.
	* Remove any salt and pepper noise from the image.
	cv::Mat noSnPImg;
	cv::medianBlur(image, noSnPImg, 5);
	if (debug) { printf("Removed salt and pepper noise.\n"); }

	// Blur the image to smoothen the edges
	cv::Mat blurredImg;
	cv::GaussianBlur(noSnPImg, blurredImg, cv::Size(7, 7), 0.1);
	if (debug) { printf("Blurred the image.\n"); }
	*/

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
	int segments = topNSegments(regionMap, segImg, numberOfSegments);
	if (debug) { printf("Segmented the binary image to have top %d regions.\n", segments); }
	if (displayIntermediateImages) {
		cv::namedWindow("Top N segmented binary Image", windowSize);
		cv::imshow("Top N segmented binary Image", segImg);
	}


	// Color the detected Segments
	cv::Mat segmentColoredImg = cv::Mat::zeros(cleanImg.size(), CV_32SC3);
	colorSegmentation(regionMap, segmentColoredImg);
	if (debug) { printf("Colored the segmented image having %d segments.\n", segments); }
	if (displayIntermediateImages) {
		cv::namedWindow("Colored Segmented Image", windowSize);
		cv::imshow("Colored Segmented Image", segmentColoredImg);
	}

	// Draw bounding boxes with orientation details.
	cv::Mat ImgWithBoxes;
	image.copyTo(ImgWithBoxes);
	drawBoundingBoxes(regionMap, ImgWithBoxes, segments);
	if (debug) { printf("Successfully drawn bouding boxes.\n"); }
	if (displayIntermediateImages) {
		cv::namedWindow("Bounding Boxes", windowSize);
		cv::imshow("Bounding Boxes", ImgWithBoxes);
	}
	if (saveImages) {
		cv::imwrite("Binary Image.jpg", binaryImg);
		cv::imwrite("Cleaned Image.jpg", cleanImg);
		cv::imwrite("Segmented Image.jpg", segImg);
		cv::imwrite("Segmented Color Image.jpg", segmentColoredImg);
		cv::imwrite("Bounded Image.jpg", ImgWithBoxes);
	}

	// Get the features of the segmented Image.
	getFeatures(regionMap, featureVector, segments);

	return 0;
}


/** This function returns only the fileName from the filePath provided.
@param filePath path of the file whose name needs to be obtained.
@param fileName placeholder for result.
@param label placeholder for the label read.
@return 0 for successfully obtaining the fileName.
@note Assumes that the filePath is valid (doesn't validate filePath)
	  Method: Parses the filePath to find the last folder separator like '/' or '\\' and
	  populates from that index to end.
*/
int getFileNameAndLabel(char*& filePath, char*& fileName, char*& label) {
	// Get the last \ index and then populate the fileName

	// Get the last '\' or '/' index in the filePath
	int length = strlen(filePath);
	int index = 0; // For marking the filePath begining
	for (int ind = length - 1; ind > -1; ind--)
	{	// Parse from the end as we are interested in last separator
		if (filePath[ind] == '\\' or filePath[ind] == '/') {
			index = ind + 1;
			break;
		}
	}

	fileName = new char[256]; // To Ensure no prepopulated data is being used.
	// Populating the fileName. 
	for (int ind = index; ind < length; ind++) {
		fileName[ind - index] = filePath[ind];
	}
	fileName[length - index] = '\0'; //To mark the end.

	label = new char[257]; // To Ensure no prepopulated data is being used.
	// Populating the label from the fileName
	for (int ind = 0; ind < strlen(fileName); ind++) {
		int value = int(fileName[ind]) - '0';
		if (value <= 9 and value >= 0)
		{
			label[ind] = '\0'; //To mark the end.
			break;
		}
		else {
			label[ind] = fileName[ind];
		}
	}

	return 0;
}