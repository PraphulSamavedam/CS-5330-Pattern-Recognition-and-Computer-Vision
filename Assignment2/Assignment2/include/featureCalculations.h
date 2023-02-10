/** Written by: Samavedam Manikhanta Praphul
This file has all the function signatures used to calculate several features of the image.
For implementation kindly look into featureCalculations.cpp file.
*/

#include <vector> // Require to hold, process and store the feature vectors.
#include <opencv2/opencv.hpp> // Required to hold, process the image objects passed to the functions.
#include <filters.h> // Required to calculate the SobelX, SobelY and gradient magnitude of the image passed.


/** This function provides the valid binSize after validates the number of bins passed as parameter.
@param numberOfBins number of bins to be used to segment 256 values.
@param echoStatus @default = false, set this boolean to enable verbose.
@returns binSize for each color.
*/
int getBinSize(int numberOfBins, bool echoStatus = false);

/*This function provides the nxn square pixel values at the center as the feature vector for the image passed.
 * @param image address of the cv::Mat object to be processed for this feature
 * @param featuerVector vector to store the center n**2 pixel values.
 * @param numOfCntrPixels[default=9] dimension of the square which needs to be extracted. i.e. for 9x9 it is 9.
 * @param echoStatus[default=false] set this bool to enable verbose.
 * @note: numOfCntrPixels needs to be odd.
 *		  featureVector will be cleared before loading the new featureVector details.
 *		  featureVector holds the value in RGB order. and not the openCV's BGR order.
 * @returns	   0 if the processing is successful
 *			-400 if the image is too small to process for this technique
*/
int baselineTechnique(cv::Mat& image, std::vector<float>& featureVector, int numOfCntrPixels = 9, bool echoStatus = false);


/*This function provides the normalized histogram of rg chromaticity of complete image passed as the feature vector.
 * @param image address of the cv::Mat object to be processed for this feature
 * @param featuerVector vector to store the normalized histogram of the image in rg chromaticity space.
 * @param histBins [default=16] number of bins to be used for each color.
 * @param echoStatus [default=false] set this bool to enable verbose.
 * @note: featureVector will be cleared before loading the new featureVector details.
 *		  featureVector holds the value in r, g values
 *		i.e. order will be in (0,0), (0,1),...,(1,0), (1,1)...
 *		  r_index is incremented when color lies in that bin based on bin calculated by
 *			r_index  = (R * histBins)/(B+G+R+0.0000001). 1.0e-7 is offset to ensure non-zero denominator.
 *			g_index  = (G * histBins)/(B+G+R+0.0000001). 1.0e-7 is offset to ensure non-zero denominator.
 *			where R, G, B are values of the pixel in Red, Green and Blue Channels.
 * @returns	   0 if the processing is successful
*/
int rgHistogramTechnique(cv::Mat& image, std::vector<float>& featureVector, int histBins = 16, bool echoStatus = false);


/** This function provides the normalized histogram in RGB space of complete image passed as the feature vector.
 * @param image address of the cv::Mat object to be processed for this feature
 * @param featuerVector vector to store the normalized histogram of the image in RGB space.
 * @param histBins number of bins to be used for each color.
 * @param echoStatus set this bool to enable verbose
 * @note: featureVector will be cleared before loading the new featureVector details.
 *		  featureVector holds the value in r, g, b values
 *			i.e. order will be in (0,0, 0), (0,0,1),...,(0,1,0),(0,1,1),...(1,0,0), (1,0,1),....
 *		  This function utilizes the getBinSize(histBins) function to get appropriate binSize.
 * Method: for each pixel, the bin to which the value belongs is determined by
 *			r_index  = R/binSize.
 *			g_index  = G/binSize.
 *			b_index  = B/binSize.
 *			where R, G, B are values of the pixel in Red, Green and Blue Channels.
 *	Finally the normalize the histogram before pushing to feature vector.
 * @returns	   0 if the processing is successful
*/
int rgbHistogramTechnique(cv::Mat& image, std::vector<float>& featureVector, int histBins = 8, bool echoStatus = false);


/*This function provides the normalized histogram as 2 separate halves in RGB space of complete image passed as the feature vector.
 * @param image address of the cv::Mat object to be processed for this feature
 * @param featuerVector vector to store the normalized histogram of the image in RGB space of two halves.
 * @param histBins number of bins to be used for each color.
 * @param echoStatus set this bool to enable verbose
 * @note featureVector will be cleared before loading the featureVector details.
 *		 This function will append the normalized histograms of two halves to the featureVector.
 *		 First half will have RGB histogram of the top half of the image.
 *		 Next half will have RGB histogram of bottom half of the image.
 *		 Feature Vector for either half will hold the value in r, g, b values
 *			i.e. order will be in (0,0, 0), (0,0,1),...,(0,1,0),(0,1,1),...(1,0,0), (1,0,1),....
 *		 This function utilizes the getBinSize(histBins) function to get appropriate binSize.
 *		 This implementation is independent of the rgbHistogramTechnique, rgHistogramTechniques.
 * Method: break the image into two halves, one for the top and other for the bottom.
 * for each half of the image,
 *		for each pixel, the bin to which the value belongs is determined by
 *			r_index  = R/binSize.
 *			g_index  = G/binSize.
 *			b_index  = B/binSize. where R, G, B are values of the pixel in Red, Green and Blue Channels.
 *	Finally the normalize the histogram before pushing to feature vector in the order of top half first and bottom half last.
 * @returns	   0 if the processing is successful
*/
int twoHalvesApproach(cv::Mat& image, std::vector<float>& featureVector, int histBins = 8, bool echoStatus = false);

/*This function provides the normalized histogram as 2 separate halves in RGB space of complete image passed as the feature vector.
 * @param image address of the cv::Mat object to be processed for this feature
 * @param featuerVector vector to store the normalized histogram of the image in RGB space of two halves.
 * @param histBins[default=8] number of bins to be used for each color.
 * @param echoStatus[default=false] set this bool to enable verbose
 * @note featureVector will be cleared before loading the featureVector details.
 *		 This function will append the normalized histograms of two halves to the featureVector.
 *		 First half will have RGB histogram of the top half of the image.
 *		 Next half will have RGB histogram of bottom half of the image.
 *		 Feature Vector for either half will hold the value in r, g, b values
 *			i.e. order will be in (0,0, 0), (0,0,1),...,(0,1,0),(0,1,1),...(1,0,0), (1,0,1),....
 *		 This function utilizes the getBinSize(histBins) function to get appropriate binSize.
 * Method: This function reuses the rgbHistogramTechnique by processing it for first half and second half
 * and pushing the results back to the feature vector.
 * @returns	   0 if the processing is successful
*/
int twoHalvesRGBApproach(cv::Mat& image, std::vector<float>& featureVector, int histBins = 8, bool echoStatus = false);


/*This function provides the normalized histogram as 2 separate halves in RGB space of complete image passed as the feature vector.
 * @param image address of the cv::Mat object to be processed for this feature
 * @param featuerVector vector to store the normalized histogram of the image in RGB space of two halves.
 * @param histBins[default=16] number of bins to be used for each color.
 * @param echoStatus[default=false] set this bool to enable verbose
 * @note featureVector will be cleared before loading the featureVector details.
 *		 This function will append the normalized histograms of two halves to the featureVector.
 *		 First half will have RGB histogram of the top half of the image.
 *		 Next half will have RGB histogram of bottom half of the image.
 *		 Feature Vector for either half will hold the value in r, g, b values
 *			i.e. order will be in (0,0, 0), (0,0,1),...,(0,1,0),(0,1,1),...(1,0,0), (1,0,1),....
 *		 This function utilizes the getBinSize(histBins) function to get appropriate binSize.
 * Method: This function reuses the rgHistogramTechnique by processing it for first half and second half
 * and pushing the results back to the feature vector.
 * @returns	   0 if the processing is successful
*/
int twoHalvesRGChromApproach(cv::Mat& image, std::vector<float>& featureVector, int histBins = 8, bool echoStatus = false);

/** This function computes and provides the normalized values of
internally computed 2 histograms as feature vectors.
This function provides equal weights to both feature vectors.
* 1. gradient magnitude to find texture in the image and
* 2. color RGB histogram to find the color distribution in the image.
 * @param image address of the cv::Mat object to be processed for this feature
 * @param featuerVector vector to store the center normalized histogram of the image in RGB space
		  for the computed histograms.
 * @param histBins number of bins to be used for each color.
 * @param echoStatus set this bool to enable verbose
 * @note featureVector will be cleared before loading the featureVector details.
 *		 This function will append the normalized histograms of gradient magnitude and RGB colorspace to the featureVector.
 *		 First half will have RGB histogram of the gradient magnitude of the image.
 *		 Next half will have RGB histogram of the image.
 *		 Feature Vector for either half will hold the value in r, g, b values
 *			i.e. order will be in (0,0, 0), (0,0,1),...,(0,1,0),(0,1,1),...(1,0,0), (1,0,1),....
 *		 This function utilizes the getBinSize(histBins) function to get appropriate binSize.
 *		 This function also utilizes the filters SobelX, SobelY, gradientMagnitude from previous assignment.
 * Method: Generate the gradient magnitude of the image passed and obtain its histogram in rg chromaticity space
 * using existing function, obtain the color histogram of the original image in RGB color space using half nuber of bins.
 * Push these two feature vectors in the same order with weighs as 0.5 and 0.5 respectively.
 * @returns	   0 if the processing is successful
*/
int textureAndColorHistApproach(cv::Mat& image, std::vector<float>& featureVector, int histBins = 16, bool echoStatus = false);

/** This */
int modRGHistogramTechnique(cv::Mat& image, std::vector<float>& featureVector, int histBins = 16);