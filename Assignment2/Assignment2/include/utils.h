/**
* This file has the utility functions definitions.
* Written by: Samavedam Manikhanta Praphul
*/

#include <vector>

/**This function asserts if number of bins passed is not positive and returns the perfect binsize to cover all values of 255.
	@note set echoStatus for print.
*/
int getBinSize(int numberOfBins, bool echoStatus = false);


int getOnlyFileName(char*& filePath,char* &fileName);

/*This function loads the feature vector for the image passed.
 * @param imagePath path of the image to be processed for this feature
 * @param featuerVector vector of the features
 * @returns	   0 if the processing is successful
 *			-100 if the image reading is unsuccessful
 *			-400 if the image is too small to process for this technique
*/
int baselineTechnique(cv::Mat& image, std::vector<float>& featureVector);

/** This function calculates the rg chromaticity of the image based on the imagePath and stores the
*/
int rghistogramTechnique(cv::Mat& image, std::vector<float>& featureVector, int histBins);

/** This function calculates the rg chromaticity of the image using linear intermediate feature vector and direct update
*/
int modRGHistogramTechnique(cv::Mat& image, std::vector<float>& featureVector, int histBins);

/** This function calculates the RGB histogram of the image using linear intermediate feature vector and direct update
*/
int rgbHistogramTechnique(cv::Mat& image, std::vector<float>& featureVector, int histBins, bool echoStatus);

int twoHalvesApproaches(cv::Mat& image, std::vector<float>& featureVector, int histBins = 8, bool echoStatus = false);

int textureAndColorHistApproach(cv::Mat& image, std::vector<float>& featureVector, int histBins = 16, bool echoStatus = false);

/*This function provides the feature vector of the image passed based on the feature requested.
	@param imagePath path of the image for which the features needs to be extracted
	@param featureTechnique feature extraction technique
	@param featureVector array of the feature considered
	@returns   0 if features are computed
			-404 if file doesn't exist at ImagePath
*/
int computeFeature(char* imagePath, char* featureTechnique, std::vector<float>& featureVector, bool echoStatus = false);


/* This function provides the float value of the sum of squared errors for all the entries in the feature vectors.
* @param featureVector1 first feature vector
* @param featureVector2 second feature vector
* @returns -100 if the lengths of the feature vectors do not match
*			float value of the sum of squared errors of all features in the feature vectors provided.
* @note: return value = sum of (square(featureVector1[i] - featureVector2[i]))
*/
float aggSquareError(std::vector<float>& featureVector1, std::vector<float>& featureVector2);

/*This function provides the distance metrics of the 2 images passed.
*	@param distanceMetric distance metric which needs to be computed
*	@param featureVector1 vector of the features to compare
* *	@param featureVector2 vector of the features to be compared with
*	@returns   0 if features are computed
*			-100 if feature vectors do not match
*/
float computeMetric(char* distanceMetric, std::vector<float>& featureVector1, std::vector<float>& featureVector2);

/** This function returns the topK filesList which are closest to the targetFeatureVector provided
	@param kFilesList vector of top K files closest to the targetfeatureVector
	@param k the number of files which needs to selected.
	@param distanceMetric based on which the files have to be selected. 
	@param targetFeatureVector feature Vector against which the data needs to be compared. 
	@param featureVectors vector of all potential featureVectors
	@param allFilesList  the list of files from which top K needs to selected. 
	@returns non-zero value if top K files are available.
*/
int getTopKElements(std::vector<char*>& kFilesList, std::vector<float>& kDistancesList, int k, char* distanceMetric, std::vector<float>& targetfeatureVector, std::vector<std::vector<float>>& featureVectors, std::vector<char*>& allFilesList);