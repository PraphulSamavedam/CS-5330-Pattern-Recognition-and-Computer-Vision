/** This file dictates the order of execution in the program

*/
#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/opencv.hpp> //Required for openCV functions
#include <vector> // Required to store the feature vector if the image passed.
#include <..\include\utils.h> // Required for the features, distance metrics processing. 
#include <..\include\readfiles.h> // Required for getting the list if files from the directory passed
#include <..\include\csv_util.h> // Required for reading and writing from a csv file

int main(int argc, char* argv[]) {
	bool echoStatus = false;
	if (argc<4)
	{
		printf("Usage: %s <fileName> <imagesDatabasePath> <featuerTechnique> <distanceMetrics>\n Aborting with exit code: -100\n", argv[0]);
		exit(-100);
	}
	if (argv[5])
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

	//Compute the features of the target image
	std::vector<float> targetImageFeatureVector;
	computeFeature(targetFilePath, featureTechnique, targetImageFeatureVector, echoStatus);

	// To store the features in the <featureTechnique>.csv file
	char fileName[256];
	strcpy(fileName, featureTechnique);
	strcat(fileName, ".csv");

	//Logic to check if the .csv file exists or not.
	
	//Get the list of image filenames to process
	std::vector<char*> filesList;
	getFilesFromDirectory(databasePath, filesList, echoStatus);

	
	printf("Files list size: %zd\n", filesList.size());
	printf("Writing the feaures to %s\n", fileName);
	int bucket_size = filesList.size() / 50;
	printf("Processing for feature vector\n..");
	for (int index = 0; index < filesList.size(); index++)
	{	
		if (index % bucket_size == 0)
		{
			printf("..");
		}
		if (echoStatus) { printf("Processing %s file\n", filesList[index]); }
		//Compute the features of each image in the image database to write to the csv file
		std::vector<float> imageFeatureVector;
		computeFeature(filesList[index], featureTechnique, imageFeatureVector, echoStatus);
		if (echoStatus) {printf("\n****\Computed feature Vector, proceeding to append data %s\n", filesList[index]);}
		// Write the feature vector to csv file
		append_image_data_csv(fileName, filesList[index], imageFeatureVector, 0);
	}
	printf("\nProcessed all the files of folder:\n%s\n", databasePath);
	

	std::vector<std::vector<float>> featureVectors;
	filesList.clear();
	read_image_data_csv(fileName, filesList, featureVectors, 0);

	std::vector<char*> kFilesList;
	int K = 3;
	int status = 
	getTopKElements(kFilesList, K, distanceMetric, targetImageFeatureVector, featureVectors, filesList);
	if (status != 0)
	{
		return status;
	}

	printf("Closely Matching images are:");
	for (int index = 0; index < K; index++)
	{
		printf("\n%s", kFilesList[index]);
		cv::Mat tmp = cv::imread(kFilesList[index], cv::WINDOW_AUTOSIZE);
		cv::imshow(kFilesList[index], tmp);
	}

	cv::Mat targetImage = cv::imread(targetFilePath, cv::WINDOW_AUTOSIZE);
	cv::imshow("Target Image", targetImage);
	char key = cv::waitKey();
	if (key == 'q')
	{
		cv::destroyAllWindows();
	}
	return 0;

}
