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

int main(int argc, char* argv[]) {
	bool echoStatus = false;
	bool resetFile = false;
	if (argc<4)
	{
		printf("Usage: %s <fileName> <imagesDatabasePath> <featuerTechnique> <distanceMetrics> <NumberOfSimilarImages> <resetFile> <echoStatus>\n Aborting with exit code: -100\n", argv[0]);
		exit(-100);
	}
	if (argv[6] !=0)
	{
		resetFile = true;		
	}
	if (argv[7] != 0)
	{
		echoStatus = true;
	}
	
	// Get the target file path
	char targetFilePath[256];
	std::strcpy(targetFilePath ,argv[1]);
	printf("Read the target file as %s\n", targetFilePath);

	// Get the images database folder
	char databasePath[512];
	std::strcpy(databasePath, argv[2]);
	printf("Read the imaegs Database folder as '%s'\n", databasePath);

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

	//Compute the features of the target image
	std::vector<float> targetImageFeatureVector;
	printf("Computing the feature of the target Image...\n");
	computeFeature(targetFilePath, featureTechnique, targetImageFeatureVector, echoStatus);
	printf("\nSuccessfully computed the features of the target Image");


	// To store the features in the <featureTechnique>.csv file
	char fileName[512];
	strcpy(fileName, featureTechnique);
	strcat(fileName, ".csv");

	// Logic to check if the .csv file exists or not.
	std::ifstream filestream;
	filestream.open(fileName);
	std::vector<char*> filesList;
	// If file doesn't exist also then resetfile will be set irrespective if ti has been set or not.
	if (!filestream) {
		resetFile = true;
	}

	// If feature vectors need to be recalculated.
	if(resetFile){
		// Get the list of image filenames to process
		getFilesFromDirectory(databasePath, filesList, echoStatus);

		if (echoStatus) { printf("Files list size: %zd\n", filesList.size());}
		if (echoStatus) { printf("Writing the feaures to %s\n", fileName); }
		if (echoStatus) { printf("Processing for feature vector\n.."); }
		for (int index = 0; index < filesList.size(); index++)
		{
			printf(".");
			if (echoStatus) { printf("Processing %s file\n", filesList[index]); }
			//Compute the features of each image in the image database to write to the csv file
			std::vector<float> imageFeatureVector;
			computeFeature(filesList[index], featureTechnique, imageFeatureVector, echoStatus);
			if (echoStatus) { printf("\n****\Computed feature Vector, proceeding to append data %s\n", filesList[index]); }
			// Write the feature vector to csv file
			append_image_data_csv(fileName, filesList[index], imageFeatureVector, false);
		}
		printf("\nProcessed all the files of folder:\n%s\n", databasePath);

	}
	
	//Read the feature vectors from the file.
	std::vector<std::vector<float>> featureVectors;
	filesList.clear();
	read_image_data_csv(fileName, filesList, featureVectors, 0);

	// Obtain the top K files
	std::vector<char*> kFilesList;
	int status = 0;
	getTopKElements(kFilesList, sImgCnt, distanceMetric, targetImageFeatureVector, featureVectors, filesList);
	if (status != 0)
	{
		return status;
	}

	printf("Closely Matching images are:");
	if (kFilesList.size() < sImgCnt)
	{
		printf("Error as kFiles are %zd while requested %d", kFilesList.size(), sImgCnt); 
		exit(-1000);
	}
	std::string windowName;
	for (int index = 0; index < sImgCnt; index++)
	{
		/*windowName.clear();*/
		printf("\nId: %d is %s", index,kFilesList[index]);
		/*windowName.append("Result ");
		windowName.append(std::to_string(index));
		windowName.append(": ");
		windowName.append(kFilesList[index]);*/
		//cv::Mat tmp = cv::imread(kFilesList[index], cv::WINDOW_AUTOSIZE);
		//cv::imshow(windowName, tmp);
	}

	cv::Mat targetImage = cv::imread(targetFilePath, cv::WINDOW_AUTOSIZE);
	cv::imshow("Target Image", targetImage);
	while (true)
	{
		char key = cv::waitKey();
		if (key == 'q')
		{
			cv::destroyAllWindows();
			break;
		}
	}
	return 0;

}
