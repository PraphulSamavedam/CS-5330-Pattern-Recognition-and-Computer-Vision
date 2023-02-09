/** This file dictates the order of execution in the program

*/
#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/opencv.hpp> // Required for openCV functions
#include <vector> // Required to store the feature vector if the image passed.
#include <..\include\utils.h> // Required for the features, distance metrics processing. 
#include <..\include\readfiles.h> // Required for getting the list if files from the directory passed
#include <..\include\csv_util.h> // Required for reading and writing from a csv file
#include <fstream> // Required to check if the file paths are valid.
#include <cstdlib> // Required for parsing user input

/*This main function drives the complete program.*/
int main(int argc, char* argv[]) {
	bool echoStatus = false;
	bool resetFile = false;
	if (argc < 5)
	{
		printf("Usage: %s <fileName> <imagesDatabasePath> <featuerTechnique> <distanceMetrics> <NumberOfSimilarImages> <[optional]resetFile> <[optional]echoStatus>\n Aborting with exit code: -100\n", argv[0]);
		exit(-100);
	}

	// Get the target file path
	char targetFilePath[256];
	std::strcpy(targetFilePath, argv[1]);
	printf("Read the target file as '%s'\n", targetFilePath);

	// Get the images database folder
	char databasePath[512];
	std::strcpy(databasePath, argv[2]);
	printf("Read the i Database folder as '%s'\n", databasePath);

	// Get the method of extracting features from the image
	char featureTechnique[100];
	std::strcpy(featureTechnique, argv[3]);
	printf("Read the feature technique to be used as '%s'\n", featureTechnique);

	// Get the distance metric to be used
	char distanceMetric[100];
	std::strcpy(distanceMetric, argv[4]);
	printf("Read the distance metric to be used as '%s'\n", distanceMetric);

	// Get the number of similar images provided by the user.
	int sImgCnt;
	sImgCnt = std::atoi(argv[5]);
	printf("Read the number of similar images to be fetched as '%d'\n", sImgCnt);

	// Get if the user requires to recalculate the feature vectors and store in file or use existing as it is.
	/* Note: If the file does not exist, it is by default set as true. */
	if (argv[6])
	{
		if (strcmp(argv[6], "0") != 0)
		{
			printf("Read the reset condition as 'true'\n");
			resetFile = true;
		}
		else
		{
			printf("Read the reset condition as 'false'\n"); // Just to inform user
			// resetFile = false; Not required as this is default
		}
	}

	// Get if the user requires verbose of the operation
	if (argv[7])
	{
		if (strcmp(argv[7], "0") != 0)
		{
			printf("Read the verbose condition as 'true'\n");
			echoStatus = true;
		}
		else {
			printf("Read the verbose condition as 'false'\n"); // Just to inform user
			// echoStatus = true; Not required as this is default
		}
	}

	//Compute the features of the target image
	std::vector<float> targetImageFeatureVector;
	printf("Computing the feature of the target image\n...\n");
	computeFeature(targetFilePath, featureTechnique, targetImageFeatureVector, echoStatus);
	printf("Successfully computed the features of the target image.\n");

	// To store the features in the <featureTechnique>.csv file
	char fileName[512];
	strcpy(fileName, featureTechnique);
	strcat(fileName, ".csv");

	// Logic to check if the .csv file exists or not.
	std::ifstream filestream;
	filestream.open(fileName);
	std::vector<char*> filesList;
	// If file doesn't exist also then resetfile will be set irrespective if it has been set or not.
	if (!filestream) { resetFile = true; }

	// If feature vectors need to be recalculated.
	if (resetFile) {
		// Get the list of image filenames to process
		getFilesFromDirectory(databasePath, filesList, echoStatus);

		// Details of files which need to be process and the location of the feature vectors file.
		if (echoStatus) { printf("Files list size: %zd\n", filesList.size()); }
		if (echoStatus) { printf("Writing the feaures to %s\n", fileName); }

		for (int index = 0; index < filesList.size(); index++)
		{
			if (index % 20 == 0) { printf("."); }
			if (echoStatus) { printf("Processing for feature vector of %s file\n", filesList[index]); }
			//Compute the features of each image in the image database to write to the csv file
			std::vector<float> imageFeatureVector;
			computeFeature(filesList[index], featureTechnique, imageFeatureVector, echoStatus);
			if (echoStatus) { printf("Computed feature vector, proceeding to append data %s\n", filesList[index]); }

			// Write the feature vector to csv file
			if (index == 0)
			{	// First entry should override the file contents to start afresh
				append_image_data_csv(fileName, filesList[index], imageFeatureVector, true);
			}
			else { // Append to the file as it was created afresh for the first image.
				append_image_data_csv(fileName, filesList[index], imageFeatureVector, false);
			}
		}
		printf("\nProcessed all the files of folder:\n%s\n", databasePath);
	}

	//Read the feature vectors from the file.
	std::vector<std::vector<float>> featureVectors;
	filesList.clear();
	read_image_data_csv(fileName, filesList, featureVectors, 0);

	// Obtain the top K files
	std::vector<char*> kFilesList; // For the top files list
	std::vector<float> kDistancesList; // For the distances of the files from target image.
	int status = 0;
	getTopKElements(kFilesList, kDistancesList, sImgCnt, distanceMetric, targetImageFeatureVector, featureVectors, filesList);

	// Status check if operation is successful.
	if (status != 0)
	{
		printf("Error Occurred, Code: %d\n", status);
		return status;
	}

	// Checking if sufficient images have been fetched from files list provided. 
	if (kFilesList.size() < sImgCnt)
	{
		printf("Error as kFiles are %zd while requested %d", kFilesList.size(), sImgCnt);
		exit(-1000);
	}

	// Print and display the closely matching images.
	printf("Closely Matching images are:\n");
	std::string windowName;
	std::vector<std::string> windowNames; // Required to save the images displayed
	for (int index = 0; index < sImgCnt; index++)
	{
		windowName.clear();
		printf("Id: %d is %s with distance: %0.4f\n", index, kFilesList[index], kDistancesList[index]);
		windowName.append("Result ");
		windowName.append(std::to_string(index));
		windowName.append(": ");
		windowName.append(kFilesList[index]);
		cv::Mat tmp = cv::imread(kFilesList[index], cv::WINDOW_AUTOSIZE);
		cv::imshow(windowName, tmp);
	}

	// Display the target image for comparision reference
	cv::Mat targetImage = cv::imread(targetFilePath, cv::WINDOW_AUTOSIZE);
	cv::imshow("Target image", targetImage);

	// Wait for user to press 'q' to exit program or close the target image.
	//while (cv::getWindowProperty("Target image", 0) !=-1 )
	while (true)
	{
		char key = cv::waitKey();
		if (key == 'q')
		{
			cv::destroyAllWindows();
			break;
		}
		if (key == 's')
		{
			for (int index = 0; index < sImgCnt; index++)
			{
				char fileName[1024];
				char* tmpFileName;
				cv::Mat tmp = cv::imread(kFilesList[index]);
				int status = getOnlyFileName(kFilesList[index], tmpFileName);
				printf("TmpFileName: %s\n", tmpFileName);
				sprintf_s(fileName, "output/ft_%s_DstMetric_%s_ID_%d_File_%s\n",
					featureTechnique, distanceMetric, index,tmpFileName);
				cv::imwrite(fileName, tmp);
				printf("Saving %s .....\n", fileName);
			}
			printf_s("Successfully saved all the %d files fetched using CBIR", sImgCnt);
		}
	}
	return 0;

}
