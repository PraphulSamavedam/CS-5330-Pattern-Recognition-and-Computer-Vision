/**
* Written by: Samavedam Manikhanta Praphul
*                   Poorna Chandra Vemula
* This file provides the signatures of several functions required in the project.
*/

#include <opencv2/opencv.hpp>

/**This function computes  the standard deviations of the features in the data
* @param data data is passed as a matrix
* @param standardDeviations st-dev to be updated
* @returns 0 if the feature is properly extracted.
*        non zero if the operation is failure.
*/
int computeStandardDeviations(std::vector<std::vector<float>>& data, std::vector<float>& standardDeviations);


/**This function computes the sum squared error as a distance metric
* @param x feature vector 1
* @param y feature vector 2
* @param distance to be updated
* @returns 0 if the feature is properly extracted.
*        non zero if the operation is failure.
*/
int sumSquaredError(std::vector<float>& x, std::vector<float>& y, float& distance);


/**This function computes the scaled eucleadian distance as a distance metric
* @param x feature vector 1
* @param y feature vector 2
* @param standardDeviations std-devs of the features
* @param distance to be updated
* @returns 0 if the feature is properly extracted.
*        non zero if the operation is failure.
*/
int eucledianDistance(std::vector<float>& x, std::vector<float>& y, std::vector<float>& standardDeviations, float& distance);

/**This function shows top matched images
* @param nMatches vector of top 'n' matched filenames
*/
void showTopMatchedImages(std::vector<char*>& nMatches);

/**This function identfies the top 'n' matches for a target image
* @param targetImage
* @param featureVectorFile
* @param distanceMetric
* @param nMatches vector of matched filenames to be updated
* @param nLabels vector of matched labels to be updated
* @returns 0 if the feature is properly extracted.
*        non zero if the operation is failure.
*/
int identifyMatches(cv::Mat& targetImage, char* featureVectorFile, char* distanceMetric, int N, std::vector<char*>& nMatches, std::vector<char*>& nLabels);



/**This function predicts the label using KNN
* @param targetImage
* @param featureVectorFile
* @param distanceMetric
* @param Label char* to be updated
* @param K parameter for KNN
* @returns 0 if the feature is properly extracted.
*        non zero if the operation is failure.
*/
int ComputingNearestLabelUsingKNN(cv::Mat& targetImage, char* featureVectorFile, char* distanceMetric, char* Label, int K);


/**This function identfies the top 'n' matches for a target image
* @param targetImage
* @param featureVectorData
* @param distanceMetric
* @param nMatches vector of matched filenames to be updated
* @param nLabels vector of matched labels to be updated
* @returns 0 if the feature is properly extracted.
*        non zero if the operation is failure.
* @note: overloaded function passing 2D vector data instead of feature Vector file
*/
int identifyMatches(cv::Mat& targetImage, std::vector<std::vector<float>> data,  std::vector<char*> filenames, std::vector<char*> labels, char* distanceMetric, int N, std::vector<char*>& nMatches, std::vector<char*>& nLabels);



/**This function predicts the label using KNN
* @param targetImage
* @param featureVectorData
* @param distanceMetric
* @param Label char* to be updated
* @param K parameter for KNN
* @returns 0 if the feature is properly extracted.
*        non zero if the operation is failure.
* @note: overloaded function passing 2D vector data instead of feature Vector file
*/
int ComputingNearestLabelUsingKNN(cv::Mat& targetImage, std::vector<std::vector<float>> data,  std::vector<char*> filenames, std::vector<char*> labels, char* featureVectorFile, char* distanceMetric, char* Label, int K);

/**This function places label on the image
* @param image
* @param label
* @param fontSize
* @param fontWeight
*/
void placeLabel(cv::Mat& image, char* label, int fontSize = 8, int fontWeight = 3);
