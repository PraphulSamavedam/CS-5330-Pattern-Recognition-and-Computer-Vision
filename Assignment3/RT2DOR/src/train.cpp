/*
* Written by : Samavedam Manikhanta Praphul
*              Poorna Chandra Vemula
* This file trains the data for the features and correspondin labels.
* This program generated from this program requires mandatory param of database images directory.
* This file can be run in manual mode or automode. by providing the right parameters.
*
*/

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include "../include/readfiles.h" // To get the list of the files in the directory
#include "../include/csv_util.h" // To work with csv file of feature vectors
#include "../include/utils.h" // Required for the standard function to obtain the feature vector

/** The application requires mandatory parameter of the training images folder. 
*	Eg. <train.exe> "../data/train_images" 
*/
int main(int argc, char* argv[])
{
	// Main configuration variables.
	int windowSize = cv::WINDOW_GUI_NORMAL;
	int grayscaleThreshold = 124; // Value is based on the experimentation with sample images
	int numberOfErosions = 5;
	int erosionConnectValue = 4;
	int dilationConnectValue = 8;
	int numberOfSegments = 1;
	bool displaySteps = true;

	if (argc < 2)
	{
		printf("Missing required arguments.\n");
		printf("Usage: %s <folderPath> <csvFilePath[default='../data/db/features.csv']> <mode>[default='auto']\n", argv[0]);
		exit(-404);
	}
	char csvFilePath[50];

	// Get the image path from input
	char folderPath[1024];
	std::strcpy(folderPath, argv[1]);

	// Read the images from the folder path provided
	std::vector<char*> filesList;
	getFilesFromDirectory(folderPath, filesList, false);

	strcpy(csvFilePath, "../data/db/features.csv");
	printf("Args: %d", argc);

	if (argc > 2)
	{
		strcpy(csvFilePath, argv[2]);
	}
	printf("Processed for the files list\n");

	char mode[2] = "a";
	if (argc > 3)
	{
		strcpy(mode, argv[3]);
	}
	printf("Mode selected: %s", mode);

	for (int index = 0; index < filesList.size(); index++)
	{
		// Get the features for each file
		std::vector<float> featureVector;
		printf("Processing %s file for features\n", filesList[index]);
		cv::Mat image = cv::imread(filesList[index]);
		if (image.data == NULL)
		{
			printf("Fatal Error file not found");
			exit(-404);
		}

		getFeaturesForImage(image, featureVector);

		// Request for the label from the user
		char* label{};
		char* fileName;

		if (strcmp(mode,"a") == 0) {
			// Automatic mode
			getFileNameAndLabel(filesList[index], fileName, label);
			printf("Processed to have fileName:%s and label: %s\n", fileName, label);
		}
		else { // Manula mode loop for valid label.
			while (true) {
				printf("Enter the label for %s:\n", filesList[index]);
				std::cin >> label;
				if (label == "") {
					printf("Invalid Emtpy Label\n");
					continue;
				}
				break;
			}
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
