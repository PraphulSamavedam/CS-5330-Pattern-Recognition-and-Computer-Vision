/**
* Written by: Samavedam Manikhanta Praphul
* This file has the primary functions which are called from the main function.
*/

#define _CRT_SECURE_NO_WARNINGS // To Suppress the strcpy issues.
#include <fstream> // For io operations like checking if file exists.
#include <opencv2/opencv.hpp> // For openCV operations like imread to check for file corruption.
#include <vector> // Required to process the vector features data
#include <queue> // Required for sorting
#include <..\include\featureCalculations.h> // Required for feature vector calculations
#include <..\include\distanceCalculations.h> // Required for distance calculations between feature vectors.


/** This custom structure is required to store the distance metric 
associated with the filename.*/
struct PairedData {	
	float value;
	char* fileName;

	// Constructor to initialize values
	PairedData(float difference, char* file_name)
	: value(difference), fileName(file_name)
	{}
};
/* This structure implements the operator overloading
 Used to sort the files along with distance metric */
struct CompareValue {
	bool operator()(PairedData const& p1, PairedData const& p2)
	{
		// return "true" if "p1" has higher value than "p2".
		return p1.value > p2.value; 
	}
};

/** This function returns only the fileName from the filePath provided.
@param filePath path of the file whose name needs to be obtained. 
@param fileName placeholder for result. 
@return 0 for successfully obtaining the fileName.
@note Assumes that the filePath is valid (doesn't validate filePath)
	  Method: Parses the filePath to find the last folder separator like '/' or '\\' and
	  populates from that index to end.
*/
int getOnlyFileName(char* &filePath, char* &fileName) {
	// Get the last \ index and then populate the fileName

	// Get the last '\' or '/' index in the filePath
	int length = strlen(filePath);
	int index = 0;
	for (int ind = length-1; ind > -1; ind--)
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
	return 0;
}

/*This function provides the feature vector of the image passed based on the feature requested.
	@param imagePath path of the image for which the features needs to be extracted
	@param featureTechnique feature extraction technique
	@param featureVector array of the feature considered
	@returns   0 if features are computed
			non-zero value if an error occured like -404 if file doesn't exist at ImagePath
*/
int computeFeature(char* imagePath, char* featureTechnique, std::vector<float>& featureVector, bool echoStatus = false) {
	int status = -404;
	std::ifstream filestream;
	filestream.open(imagePath);

	if (!filestream) {
		printf("File does not exist.\n");
		return status;
	}
	cv::Mat image = cv::imread(imagePath);
	if (image.data == NULL) {
		printf("%s file exists but it is corrupted, kindly check.\n", imagePath);
		exit(-100);
	}
	if (echoStatus) { printf("\n%s file exists.\n", imagePath); }
	// Calculate the feature based on the selection
	if (strcmp(featureTechnique, "Baseline") == 0)
	{
		status = baselineTechnique(image, featureVector);
	}
	else if (strcmp(featureTechnique, "2DHistogram") == 0)
	{
		status = rgHistogramTechnique(image, featureVector, 16);
	}
	else if (strcmp(featureTechnique, "Q2DHistogram") == 0)
	{
		status = modRGHistogramTechnique(image, featureVector, 16);
	}
	else if (strcmp(featureTechnique, "3DHistogram") == 0)
	{
		//printf("Calculating the histogram feature....");
		status = rgbHistogramTechnique(image, featureVector, 8, false);
	}
	else if (strcmp(featureTechnique, "2HalvesHistogram") == 0)
	{
		status = twoHalvesApproach(image, featureVector, 8, false);
	}
	else if (strcmp(featureTechnique, "TACHistogram") == 0)
	{
		status = textureAndColorHistApproach(image, featureVector, 16, false);
	}
	else
	{
		status = -500;
		printf("Erorr code: -500\nInvalid Feature technique '%s'\n", featureTechnique);
	}
	if (echoStatus) { printf("\nFeature vector calculated for %s\n", imagePath); }
	if (status != 0)
	{
		printf("\nError code :'%d'\n while calculating the feature '%s' for\nimage: %s\n",
			status, featureTechnique, imagePath);
		return status;
	}
	return 0;
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

	// Cannot compute the distance metric for feature vectors of different sizes.
	assert(featureVector1.size(), featureVector2.size());

	// Switch to appropriate distance Metric
	if (strcmp(distanceMetric, "AggSquareError") == 0)
	{
		return aggSquareError(featureVector1, featureVector2);
	}
	else if (strcmp(distanceMetric, "HistogramError") == 0)
	{
		return histogramIntersectionError(featureVector1, featureVector2);
	}
	else if (strcmp(distanceMetric, "EntropyeError") == 0)
	{
		return histogramIntersectionError(featureVector1, featureVector2);
	}
	if (status != 0)
	{
		printf("Unsupported distance Metric:%s\n", distanceMetric);
		printf("Error code: -500 (for distance Metric)");
		exit(- 500);
	}
	return status;
}

/** This function returns the topK filesList which are closest to the targetFeatureVector provided
	@param kFilesList vector of top K files closest to the target feature vector.
	@param kDistancesList vector of top K distances closest to the target feature vector.
	@param k the number of files which needs to selected.
	@param targetFeatureVector feature Vector against which the data needs to be compared.
	@param featureVectors vector of all potential featureVectors
	@param allFilesList  the list of files from which top K needs to selected.
	@returns non-zero value if top K files are available.
*/
int getTopKElements(std::vector<char*>& kFilesList, std::vector<float>& kDistancesList, int k, char* distanceMetric, std::vector<float>& targetfeatureVector, std::vector<std::vector<float>>& featureVectors, std::vector<char*>& allFilesList) {
	if (k > allFilesList.size())
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
	std::priority_queue<PairedData, std::vector<PairedData>, CompareValue> diffQueue;

	float difference = 0.0;
	for (int index = 0; index < featureVectors.size(); index++)
	{
		imageFtVector = featureVectors[index];
		difference = computeMetric(distanceMetric, targetfeatureVector, imageFtVector);
		char* currentFileName = allFilesList[index];
		diffQueue.push(PairedData(difference, currentFileName));
		//printf("\nDifference was calculated as :%.02f for %s", difference, currentFileName);
	}

	// Get the top K files names based on the difference
	kFilesList.clear(); // To ensur that we do not append the data errornously.
	kDistancesList.clear(); // To ensure that we do not append the data errorneously. 
	for (int count = 0; count < k; count++)
	{
		PairedData data = diffQueue.top();
		kFilesList.push_back(data.fileName);
		kDistancesList.push_back(data.value);
		diffQueue.pop();
	}
	return 0;
}