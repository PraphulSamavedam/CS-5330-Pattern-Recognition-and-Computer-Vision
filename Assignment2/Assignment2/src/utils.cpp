/**
* Written by: Samavedam Manikhanta Praphul
* This file has the primary functions which are called from the main function.
*/

#define _CRT_SECURE_NO_WARNINGS // To Suppress the strcpy issues.
#include <opencv2/opencv.hpp> //For openCV operations like imread
#include <fstream> // For io operations like checking if file exists.
#include <cmath>  // Required for the pow function
#include <vector> // Required to process the vector features data
#include <queue> //Required for priority_queue
#include <tuple> //Required for creating tuple.


struct PairedData {

	float value;

	char* fileName;

	// this will used to initialize the variables
	// of the structure
	PairedData(float difference, char* file_name)
		: value(difference), fileName(file_name)
	{
	}
};
// this is an structure which implements the
// operator overloading
struct CompareValue {
	bool operator()(PairedData const& p1, PairedData const& p2)
	{
		// return "true" if "p1" is ordered
		// before "p2", for example:
		return p1.value > p2.value;
	}
};


/*This function loads the feature vector for the image passed.
 * @param imagePath path of the image to be processed for this feature
 * @param featuerVector vector of the features
 * @returns	   0 if the processing is successful
 *			-100 if the image reading is unsuccessful
 *			-400 if the image is too small to process for this technique
*/
int baselineTechnique(char* imagePath, std::vector<float>& featureVector) {
	cv::Mat image = cv::imread(imagePath);
	if (image.data ==NULL)
	{
		printf("%s file is corrupted, kindly check.\n", imagePath);
		exit(-100);
	}
	int midRow = image.rows/2;
	int midCol = image.cols/2;
	if (midRow < 4 or midCol < 4)
	{
		printf("Cannot process baseline technique for the image %s.\n", imagePath);
		exit(-400);
	}
	featureVector.clear();
	for (int rowIncr = -4; rowIncr < 5; rowIncr++)
	{
		cv::Vec3b *rowPtr = image.ptr<cv::Vec3b>(midRow + rowIncr);
		for (int colIncr = -4; colIncr < 5; colIncr++)
		{
			featureVector.push_back(rowPtr[midCol + colIncr][0]); // Blue Channel value
			featureVector.push_back(rowPtr[midCol + colIncr][1]); // Green Channel value
			featureVector.push_back(rowPtr[midCol + colIncr][2]); // Red Channel value
		}
	}
	return 0;
}

/*This function provides the feature vector of the image passed based on the feature requested.
	@param imagePath path of the image for which the features needs to be extracted
	@param featureTechnique feature extraction technique
	@param featureVector array of the feature considered
	@returns   0 if features are computed
			non-zero value if an error occured like -404 if file doesn't exist at ImagePath
*/
int computeFeature(char* imagePath,char* featureTechnique, std::vector<float>& featureVector, int echoStatus) {
	int status = -404;
	std::ifstream filestream;
	filestream.open(imagePath);
	if (!filestream) {
		printf("File does not exist");
		return status;
	}
	
	if (strcmp(featureTechnique, "Baseline") == 0)
	{
		 status = baselineTechnique(imagePath, featureVector);
	}
	else
	{
		status = -500;
		printf("Erorr code: -500\nInvalid Feature technique '%s'\n", featureTechnique);
	}
	if (echoStatus) { printf("Feature Vector Calculated for %s\n", imagePath); }
	if (status != 0)
	{
		printf("Error code :'%d'\n while calculating the feature '%s' for\nimage: %s\n",
			status, featureTechnique, imagePath);
		return status;
	}
	return 0;
}


/* This function provides the float value of the sum of squared errors for all the entries in the feature vectors.
* @param featureVector1 first feature vector
* @param featureVector2 second feature vector
* @returns -100 if the lengths of the feature vectors do not match
*			float value of the sum of squared errors of all features in the feature vectors provided.
* @note: return value = sum of (square(featureVector1[i] - featureVector2[i]))
*/
float aggSquareError(std::vector<float>& featureVector1, std::vector<float>& featureVector2) {
	// Assuming the featureVectors are of same size
	float result = 0.0;
	int length = featureVector1.size();
	for (int index = 0; index < length; index++)
	{
		// Aggregrate the square of the error of the feature vectors
		result += ((featureVector1[index] - featureVector2[index])*(featureVector1[index] - featureVector2[index]));
	}
	return result;
}

/*This function provides the distance metrics of the 2 images passed.
*	@param distanceMetric distance metric which needs to be computed
*	@param featureVector1 vector of the features to compare
* *	@param featureVector2 vector of the features to be compared with
*	@returns   0 if features are computed
*			-100 if feature vectors do not match
*/
float computeMetric(char* distanceMetric, std::vector<float>& featureVector1, std::vector<float>& featureVector2) {
	int status = -100;
	if (featureVector1.size() != featureVector2.size())
	{
		// Cannot compute the distance metric for feature vectors of different sizes.
		return status;
	}
	if (strcmp(distanceMetric,"AggSquareError")==0)
	{
		return aggSquareError(featureVector1, featureVector2);
	}
	if (status != 0)
	{
		printf("Error processing %s for feature vectors", distanceMetric);
		status = -500;
	}
	return status;
}


/** This function returns the topK filesList which are closest to the targetFeatureVector provided
	@param kFilesList vector of top K files closest to the targetfeatureVector
	@param k the number of files which needs to selected.
	@param targetFeatureVector feature Vector against which the data needs to be compared.
	@param featureVectors vector of all potential featureVectors
	@param allFilesList  the list of files from which top K needs to selected.
	@returns non-zero value if top K files are available.
*/
int getTopKElements(std::vector<char*>& kFilesList, int k, char* distanceMetric, std::vector<float>& targetfeatureVector, std::vector<std::vector<float>>& featureVectors, std::vector<char*>& allFilesList) {
	if (k> allFilesList.size())
	{	// There can be atmost be all the files in the target vector kFilesList.
		return -100;
	}
	if (featureVectors.size() != allFilesList.size())
	{
		//Mismatch in the feature Vectors provided and the list of files processed.
		return -101;
	}
	std::vector<float> imageFtVector;

	// Priority Queue to get the top K elements tuple
	std::priority_queue<PairedData, std::vector<PairedData>,CompareValue> diffQueue;
	float difference = 0.0;
	for (int index = 0; index < featureVectors.size(); index++)
	{
		imageFtVector = featureVectors[index];
		difference = computeMetric(distanceMetric, targetfeatureVector, imageFtVector);
		char* currentFileName = allFilesList[index];
		diffQueue.push(PairedData(difference,currentFileName));
		//printf("\nDifference was calculated as :%.02f for %s", difference, currentFileName);
	}

	// Get the top K files names based on the difference
	kFilesList.clear();
	for (int count = 0; count < k; count++)
	{
		PairedData data = diffQueue.top();
		kFilesList.push_back(data.fileName);
		diffQueue.pop();
	}
	return 0;
}