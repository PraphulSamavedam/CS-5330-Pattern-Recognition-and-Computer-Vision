/*
* Written by : Samavedam Manikhanta Praphul
* This file displays the live labelling of the objects.
*/

#define _CRT_SECURE_NO_WARNINGS
//#include "../include/RT2DOR.h"
#include <opencv2/opencv.hpp>

//#include "../include/utils.h" // Required for feature calculation.
#include "../include/match.h" // For getting the label


using namespace std;

int main(int argc, char* argv[])
{
	// Main configuration variables.
	int windowSize = cv::WINDOW_GUI_EXPANDED;
	int grayscaleThreshold = 124; // Value is based on the experimentation with sample images
	int numberOfErosions = 5;
	int numberOfSegments = 1;
	bool displaySteps = false;
	bool debug = false;

	// Setup the camera for Capture
	cv::VideoCapture* capture = new cv::VideoCapture(0);
	// Check if any video capture device is present.
	if (!capture->isOpened())
	{
		std::printf("Unable to open the primary video device.\n");
		return(-404);
	}
	cv::Size refs((int)capture->get(cv::CAP_PROP_FRAME_WIDTH),
		capture->get(cv::CAP_PROP_FRAME_HEIGHT));
	std::printf("Camera Capture size: %d x %d \n.", refs.width, refs.height);

	cv::Mat frame;
	while (true)
	{
		*capture >> frame;
		//get new frame from camera, treating as stream.

		if (frame.empty()) {
			std::printf("Frame is empty");
			break;
		}

		// Display the current stream of images in the video stream.
		// Displaying the image read.
		cv::namedWindow("Live Video", windowSize);
		cv::imshow("Live Video", frame);

		if (debug) { printf("Read the original image.\n"); }

		std::vector<float> featureVector;

		getFeaturesForImage(frame,featureVector);

		char vectorFilePath[100] = "../data/db/features.csv";
		char distanceMetric[100] = "euclidean";
		char predictedLabel[100];
		// To Do here after computed images to get the label from existing functions. 
		ComputingNearestLabelUsingKNN(frame, vectorFilePath, distanceMetric, predictedLabel, 5);

		placeLabel(frame, predictedLabel);
		char key = cv::waitKey(100);
		if (key == 'q')
		{
			cv::destroyAllWindows();
			break;
		}
		else if (key == 's')
		{
			cv::setBreakOnError(true);
		}
	}
}
