/** Written by: Samavedam Manikhanta Praphul
This file is GUI based program with some default values built.
which are targetFilePath, imagesDatabasePath, featureTechnique, distanceMetrics, NumberOfSimilarImages.
Based on the parameters passed similar images are displayed to the user. 
*/
 
#define _CRT_SECURE_NO_WARNINGS // To Supress warnings
#define CVUI_IMPLEMENTATION // To Enable GUI 
//Reference for GUI: https://dovyski.github.io/cvui/usage/

#include <opencv2/opencv.hpp> // Required for openCV functions
#include <vector> // Required to store the feature vector if the image passed.
#include <..\include\utils.h> // Required for the features, distance metrics processing. 
#include <..\include\readfiles.h> // Required for getting the list if files from the directory passed
#include <..\include\csv_util.h> // Required for reading and writing from a csv file
#include <fstream> // Required to check if the file paths are valid.
#include <cstdlib> // Required for parsing user input
#include <..\include\cvui.h> //Required for GUI interface.

/*This main function drives the complete program.*/
int main(int argc, char* argv[]) {
	
	bool echoStatus = false;
	bool resetFile = false;

	if (argc < 3) // Did not remove this due to time.
	{
		printf("Usage: %s <fileName> <imagesDatabasePath> \n Aborting with exit code: -100\n", argv[0]);
		exit(-100);
	}

	// Get the target file path
	char targetFilePath[256];
	std::strcpy(targetFilePath, argv[1]);
	printf("Read the target file as '%s'\n", targetFilePath);

	// Get the images database folder
	char databasePath[512];
	std::strcpy(databasePath, argv[2]);
	printf("Read the image Database folder as '%s'\n", databasePath);
	
	// Initialize GUI 
	char ftSettingsWindowName[17] = "Feature Settings";
	cv::namedWindow(ftSettingsWindowName, cv::WINDOW_AUTOSIZE);
	cvui::init(ftSettingsWindowName);

	cv::Mat frame = cv::Mat(300, 400, CV_8UC3);
	frame = cv::Scalar(49, 52, 49);
	//frame = cv::Scalar(214, 191, 92);
	// Get the method of extracting features from the image
	char featureTechnique[100];
	bool selectedFeature = false;

	while (!selectedFeature)
	{
		cv::imshow(ftSettingsWindowName, frame);

		// Checkbox if features have to be re evaluated
		cvui::checkbox(frame, 80, 25, "Re-evaluate ft vectors", &resetFile,'0xff');

		if (cvui::button(frame, 80, 60, "9x9Baseline")) {
			std::strcpy(featureTechnique, "Baseline");
			selectedFeature = true;
		}

		if (cvui::button(frame, 80, 90, "2DHistogram")) {
			std::strcpy(featureTechnique, "2DHistogram");
			selectedFeature = true;
		}

		if (cvui::button(frame, 80, 120, "3DHistogram")) {
			std::strcpy(featureTechnique, "3DHistogram");
			selectedFeature = true;
		}

		if (cvui::button(frame, 80, 150, "MultiHistogram - UB")) {
			std::strcpy(featureTechnique, "2HalvesUBHistogram");
			selectedFeature = true;
		}

		if (cvui::button(frame, 80, 180, "MultiHistogram - LR")) {
			std::strcpy(featureTechnique, "2HalvesLRHistogram");
			selectedFeature = true;
		}

		if (cvui::button(frame, 80, 210, "TextureHistogram")) {
			std::strcpy(featureTechnique, "TACHistogram");
			selectedFeature = true;
		}

		if (cvui::button(frame, 80, 240, "Q4TextureHistogram")) {
			std::strcpy(featureTechnique, "Q4TextureHistogram");
			selectedFeature = true;
		}

		if (cvui::button(frame, 80, 270, "CentreAndTextureHistogram")) {
			std::strcpy(featureTechnique, "CustomHistogram");
			selectedFeature = true;
		}

		cvui::update();
		// Option to exit if 
		if (cv::waitKey(10) == 27) {
			printf("Terminating the program as 'Esc' is pressed.\n");
			exit(-100);
		}
	}
	cv::destroyWindow(ftSettingsWindowName);
	printf("Read the feature technique to be used as '%s'\n", featureTechnique);

	printf("Read the reset feature vectors as '%s'\n", resetFile ? "true": "false");

	// Get the number of similar images provided by the user.
	int sImgCnt = 3;//Default = 3

	// Get the distance metric to be used
	char distanceMetric[100];
	bool selectedDistanceMetric = false;

	// Initialize GUI for distance and Number of images
	char distSettingsWindowName[32] = "Distance & # of Images settings";
	cv::Mat frame2 = cv::Mat(340, 400, CV_8UC3);
	frame2 = cv::Scalar(49, 52, 49);
	cv::namedWindow(distSettingsWindowName, cv::WINDOW_AUTOSIZE);
	cvui::init(distSettingsWindowName);

	while (!selectedDistanceMetric)
	{
		cv::imshow(distSettingsWindowName, frame2);

		//TrackBar for selecting the number of images.
		cvui::trackbar(frame2, 80, 20, 200, &sImgCnt, 3, 15);

		//Create trackbars in "Distance Metrics" window
		//cv::createTrackbar("Images:", "distSettingsWindowName", &sImgCnt, 15); //Hue (0 - 179)

		if (cvui::button(frame2, 80, 80, "Sum of squared error")) {
			std::strcpy(distanceMetric, "AggSquareError");
			selectedDistanceMetric = true;
		}

		if (cvui::button(frame2, 80, 120, "HistogramError")) {
			std::strcpy(distanceMetric, "HistogramError");
			selectedDistanceMetric = true;
		}

		if (cvui::button(frame2, 80, 160, "EntropyError")) {
			std::strcpy(distanceMetric, "EntropyError");
			selectedDistanceMetric = true;
		}

		if (cvui::button(frame2, 80, 200, "Weighted_80_20_HistogramError")) {
			std::strcpy(distanceMetric, "W82HistogramError");
			selectedDistanceMetric = true;
		}

		if (cvui::button(frame2, 80, 240, "MeanSquareError")) {
			std::strcpy(distanceMetric, "MeanSquareError");
			selectedDistanceMetric = true;
		}

		if (cvui::button(frame2, 80, 280, "MaskedBoundError")) {
			std::strcpy(distanceMetric, "MaskedBoundError");
			selectedDistanceMetric = true;
		}

		cvui::update();
		// Option to exit if 
		if (cv::waitKey(10) == 27) {
			printf("Terminating the program as 'Esc' is pressed.\n");
			exit(-100);
		}
	}
	cv::destroyWindow(distSettingsWindowName);
	printf("Read the distance metric to be used as '%s'\n", distanceMetric);
	printf("Read the number of similar images to be fetched as '%d'\n", sImgCnt);
	


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

		printf("Processing feature vectors of the files in %s folder\n", databasePath);
		//Compute the feature vectors of images in the folder and store in the <featureVector.csv> file.
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
	// while (cv::getWindowProperty("Target image", 0) !=-1 )
	while (true)
	{
		char key = cv::waitKey();
		if (key == 'q')
		{
			printf("Termintating the program as 'q' has been pressed.\n");
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
					featureTechnique, distanceMetric, index, tmpFileName);
				cv::imwrite(fileName, tmp);
				printf("Saving %s .....\n", fileName);
			}
			printf_s("Successfully saved all the %d files fetched using CBIR", sImgCnt);
		}
	}

}