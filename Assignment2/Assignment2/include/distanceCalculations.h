/** Written by: Samavedam Manikhanta Praphul
* This file has several method signatures used to calculate the distance between two feature vectors.
* For implementations, kindly look at the distanceCalculations.cpp file. 
*/

#include <vector> //Requierd to hold, process and store feature vectors. 

/* This function provides the float value of the sum of squared errors for all the entries in the feature vectors.
* @param featureVector1 first feature vector.
* @param featureVector2 second feature vector.
* @returns float value of the sum of squared errors of all features in the feature vectors provided.
* @note: function assume that the feature vectors are of same length.
*		 return value = sum of {square(featureVector1[i] - featureVector2[i])), for all i} i.e. all entries of feature vector.
*
*/
float aggSquareError(std::vector<float>& featureVector1, std::vector<float>& featureVector2);

/* This function provides the float value of mean squared errors for all the entries in the feature vectors.
* @param featureVector1 first feature vector.
* @param featureVector2 second feature vector.
* @returns float value of the sum of squared errors of all features in the feature vectors provided.
* @note: function assume that the feature vectors are of same length.
*		 return value = root(mean(sum of {square(featureVector1[i] - featureVector2[i])), for all i})) i.e. all entries of feature vector
*
*/
float meanSquaredError(std::vector<float>& featureVector1, std::vector<float>& featureVector2);

/* This function provides the float value of the histogram error between two feature vectors.
* @param featureVector1 normalized histogram feature vector of image1.
* @param featureVector2 normalized histogram feature vector of image2.
* @returns float value of the histogram error between two feature vectors.
* @note: function assume that the feature vectors are of same length.
*		 function assumes the feature vectors are normalized.
*		 return value = 1 - sum{min(featureVector1[i], featureVector2[i]), for all i} i.e. all entries of feature vector.
*/
float histogramIntersectionError(std::vector<float>& featureVector1, std::vector<float>& featureVector2);

/** This function provides the entropy difference of the feature vectors as the result.
* @param featureVector1 normalized histogram feature vector of image1.
* @param featureVector2 normalized histogram feature vector of image2.
* @returns the Entropy difference between the feature vectors
* @note function assume that the feature vectors are of same length.
*		function assumes the feature vectors are normalized
*		Method: Entropy = sum{-probability[i]* log(probability[i]), for all i} i.e. all entries of feature vector.
*/
float entropyError(std::vector<float>& featureVector1, std::vector<float>& featureVector2);

/** This function provides the difference of the feature vectors as the result assuming that the 2 features are
* normalized contour area of the image and normalized height/width ratio.
* @param featureVector1 normalized feature vector of image1.
* @param featureVector2 normalized feature vector of image2.
* @returns the maskedBoundaryError as sum of the maximum overlap of the common areas.
* @note function assume that the feature vectors are of same length.
*		function assumes the feature vectors are 2 features: normalized contour area, normalized height/width ratio.
* Method: Error  = 2- sum{min(feature overlap), for 2 features in the feature vector.
*/
float maskedBoundaryError(std::vector<float>& featureVector1, std::vector<float>& featureVector2);

/** This function provides the weighted histogram difference of the feature vectors as the result.
* Useful when feature vectors having histograms need to be weighted differently.
* @param featureVector1 normalized histogram feature vector of image1.
* @param featureVector2 normalized histogram feature vector of image2.
* @param featuresLengths length of histograms used in the feature vectors.
* @param weights weights of the histogram values in the final result.
* @returns weighted histogram difference between the feature vectors
* @note function assume that the feature vectors are of same length.
*		function assumes the feature vectors are normalized
*		Method: result = 1 - sum{weight * histogramIntersectionError{for each length of partitioned feature vector}
								, for all partitioned feature vector} where partition is done on the basis lengths provided.
*/
float weightedHistogramIntersectionError(std::vector<float>& featureVector1, std::vector<float>& featureVector2, std::vector<int>& featuresLengths, std::vector<float>& weights);
