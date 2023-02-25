/*
* Written by : Samavedam Manikhanta Praphul
* This file defines the entry point for the application -- RT2DOR.cpp
*/

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include "../include/readfiles.h" // To get the list of the files in the directory
#include "../include/csv_util.h" // To work with csv file of feature vectors
#include "../include/utils.h" // Required for the standard function to obtain the feature vector


using namespace std;

int main(int argc, char* argv[])
{
	// Main configuration variables.
	int windowSize = cv::WINDOW_GUI_NORMAL;
	int grayscaleThreshold = 124; // Value is based on the experimentation with sample images
	int numberOfErosions = 1;
	int numberOfSegments = 3;
	bool displaySteps = true;

	if (argc<2)
	{
		printf("Missing required arguments.\n");
		printf("Usage: %s <folderPath> <csvFilePath[default='../data/db/features.csv']>\n", argv[0]);
		exit(-404);
	}
	char csvFilePath[50];

	// Get the image path from input
	char folderPath[1024];
	std::strcpy(folderPath, argv[1]);

	// Read the images from the folder path provided
	std::vector<char*> filesList;
	getFilesFromDirectory(folderPath, filesList, false);

	if (argc == 3)
	{
		strcpy(csvFilePath, argv[2]);
	}
	strcpy(csvFilePath, "../data/db/features.csv");
	printf("Processed for the files list\n");

	for (int index = 0; index < filesList.size(); index++)
	{
		// Get the features for each file
		vector<float> featureVector;
		getFeaturesForImage(filesList[index], featureVector, 124, 1, 4, 8, 1, false, false);

		cv::Mat image = cv::imread(filesList[index]);
		
		// Request for the label from the user
		char label[256];
		printf("Enter the label for %s:\n", filesList[index]);
		std::cin >> label;
		if (label == "") {
			printf("Invalid Emtpy Label\n");
			continue;
		}
		
		// Write the feature vector to csv file
		if (index == 0)
		{	// First entry should override the file contents to start afresh
			append_image_data_csv(csvFilePath, filesList[index], label, featureVector, true);
		}
		else { // Append to the file as it was created afresh for the first image.
			append_image_data_csv(csvFilePath, filesList[index], label, featureVector, false);
		}
	}
}
