/** Written by: Samavedam Manikhanta Praphul
* This file has several methods to calculate the distance between two feature vectors.
* All the functions assume that the feature vectors are of same length
*/

#include <vector> //Requierd to hold, process and store feature vectors. 
#include <cassert> //Required for asserts
#include <cmath> //Required for logarithm function

/* This function provides the float value of the sum of squared errors for all the entries in the feature vectors.
* @param featureVector1 first feature vector.
* @param featureVector2 second feature vector.
* @returns float value of the sum of squared errors of all features in the feature vectors provided.
* @note: function assume that the feature vectors are of same length.
*		 return value = sum of {square(featureVector1[i] - featureVector2[i])), for all i} i.e. all entries of feature vector.
*
*/
float aggSquareError(std::vector<float>& featureVector1, std::vector<float>& featureVector2) {
	// Assumes the featureVectors are of same size
	float result = 0.0;
	int length = featureVector1.size();
	for (int index = 0; index < length; index++)
	{
		// Aggregrate the square of the error of the feature vectors
		result += ((featureVector1[index] - featureVector2[index]) * (featureVector1[index] - featureVector2[index]));
	}
	return result;
}

/* This function provides the float value of mean squared errors for all the entries in the feature vectors.
* @param featureVector1 first feature vector.
* @param featureVector2 second feature vector.
* @returns float value of the sum of squared errors of all features in the feature vectors provided.
* @note: function assume that the feature vectors are of same length.
*		 return value = root(mean(sum of {square(featureVector1[i] - featureVector2[i])), for all i})) i.e. all entries of feature vector
*
*/
float meanSquaredError(std::vector<float>& featureVector1, std::vector<float>& featureVector2) {
	// Assumes the featureVectors are of same size
	float result = 0.0;
	int length = featureVector1.size();

	for (int index = 0; index < length; index++)
	{
		// Aggregrate the square of the error of the feature vectors
		result += ((featureVector1[index] - featureVector2[index]) * (featureVector1[index] - featureVector2[index]));
	}
	//Normalize the error and obtain the square root.
	return std::sqrtf(result / length);
}

/* This function provides the float value of the histogram error between two feature vectors.
* @param featureVector1 normalized histogram feature vector of image1.
* @param featureVector2 normalized histogram feature vector of image2.
* @returns float value of the histogram error between two feature vectors.
* @note: function assume that the feature vectors are of same length.
*		 function assumes the feature vectors are normalized.
*		 return value = 1 - sum{min(featureVector1[i], featureVector2[i]), for all i} i.e. all entries of feature vector.
*/
float histogramIntersectionError(std::vector<float>& featureVector1, std::vector<float>& featureVector2) {
	// Assuming the featureVectors are of same size
	float result = 0.0;
	int length = featureVector1.size();
	for (int index = 0; index < length; index++)
	{
		// Aggregrate the square of the error of the feature vectors
		// Ternary operator to obtain the minimum value
		result += (featureVector1[index] > featureVector2[index] ? featureVector2[index] : featureVector1[index]);
	}
	return 1 - result;
}

/** This function provides the absolutle entropy difference of the feature vectors as the result.
* @param featureVector1 normalized histogram feature vector of image1.
* @param featureVector2 normalized histogram feature vector of image2.
* @returns the Entropy difference between the feature vectors
* @note function assume that the feature vectors are of same length.
*		function assumes the feature vectors are normalized histograms
* Method: Entropy = sum{-probability[i]* log(probability[i]), for all i} i.e. all entries of feature vector.
*/
float entropyError(std::vector<float>& featureVector1, std::vector<float>& featureVector2) {
	int length = featureVector1.size();
	double entropy1 = 0.0;
	double entropy2 = 0.0;
	for (int index = 0; index < length; index++)
	{
		double probability1 = featureVector1[index];
		if (probability1 != 0.0) {
			entropy1 += (probability1*log(probability1));
		}
		double probability2 = featureVector2[index];
		if (probability2 != 0.0) {
			entropy2 += (probability2 * log(probability2));
		}
	}
	return abs(entropy1 - entropy2);
}

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
float weightedHistogramIntersectionError(std::vector<float>& featureVector1, std::vector<float>& featureVector2, std::vector<int>& featuresLengths, std::vector<float>& weights) {
	// Assuming the featureVectors are of same size
	float result = 0.0;
	//FeaturesLengths and weights must have the same length
	assert(featuesLengths.size() == weights.size());
	float totalWeight = 0.0;
	// For each feature get the weight and calculate the weight of the histogram intersection.
	for (int index = 0; index < featuresLengths.size(); index++)
	{
		int length = featuresLengths[index];
		float weight = weights[index];
		float featureResult = 0.0;
		for (int index = 0; index < length; index++)
		{
			// Aggregrate the square of the error of the feature vectors
			featureResult += (featureVector1[index] > featureVector2[index] ? featureVector2[index] : featureVector1[index]);
		}
		result += (weight * featureResult);
	}
	return 1 - (result / totalWeight);
}