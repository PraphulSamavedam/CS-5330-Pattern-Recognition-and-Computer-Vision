/*
* Written by : Samavedam Manikhanta Praphul
* This file displays the live labelling of the objects.
*/

#define _CRT_SECURE_NO_WARNINGS // To suppress warnings

#include <opencv2/opencv.hpp>

//#include "../include/utils.h" // Required for feature calculation.
#include "../include/utils.h"
#include "../include/match_utils.h"  // For getting the label


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
	int K = 5;

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

		if (debug) { printf("Read the original image.\n"); }

		// Get the target feature vector
		std::vector<float> featureVector;
		getFeaturesForImage(frame, featureVector);

		char vectorFilePath[100] = "../data/db/features.csv";
		char distanceMetric[100] = "euclidean";
		char predictedLabel[100];

		ComputingNearestLabelUsingKNN(frame, vectorFilePath, distanceMetric, predictedLabel, K);
		printf("Label Predicted: %s", predictedLabel);
		placeLabel(frame, predictedLabel, 1, 2);

		// Display the current stream of images in the video stream.
		cv::namedWindow("Live Video", windowSize);
		cv::imshow("Live Video", frame);
		char key = cv::waitKey(100);
		if (key == 'q')
		{
			cv::destroyAllWindows();
			break;
		}
		else if (key == 's')
		{
			cv::setBreakOnError(true);
			cv::destroyAllWindows();
		}
	}
}