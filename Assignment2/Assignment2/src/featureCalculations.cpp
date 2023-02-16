/** Written by: Samavedam Manikhanta Praphul
* This file has the different ways to calculate the feature vector of the image.
*/

#include <vector> // Require to hold, process and store the feature vectors.
#include <opencv2/opencv.hpp> // Required to hold, process the image objects passed to the functions.
#include <filters.h> // Required to calculate the SobelX, SobelY and gradient magnitude of the image passed.

/** This function provides the valid binSize after validates the number of bins passed as parameter.
@param numberOfBins number of bins to be used to segment 256 values.
@param echoStatus @default = false, set this boolean to enable verbose.
@returns binSize for each color.
*/
int getBinSize(int numberOfBins, bool echoStatus = false) {
	// Number of Bins must be positive
	assert(numberOfBins > 0);

	// Inform the user
	if (echoStatus) { printf("Number of bins passed: %d\n", numberOfBins); }
	if (256 % numberOfBins == 0)
	{
		// Perfect BinSize is possible to cover all scenarios
		if (echoStatus) { printf("Perfect BinSizing: %d\n", 256 / numberOfBins); }
		return 256 / numberOfBins;
	}
	else
	{
		if (echoStatus) { printf("Adjusted BinSizing: %d\n", 1 + (256 / numberOfBins)); }
		// Ensure all levels are covered for this binSize
		return 1 + (256 / numberOfBins);
	}
}

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
int baselineTechnique(cv::Mat& image, std::vector<float>& featureVector, int numOfCntrPixels = 9, bool echoStatus = false) {

	if (echoStatus) { printf("Dimension of central space: %d x %d\n", numOfCntrPixels, numOfCntrPixels); }
	// Ensure only odd dimension is passed. 
	assert(numOfCntrPixels % 2 == 1);

	// Calculate the center of the image.
	if (echoStatus) { printf("Image Dimensions: %d x %d\n", image.rows, image.cols); }
	int midRow = image.rows / 2;
	int midCol = image.cols / 2;
	if (echoStatus) { printf("Center of the image is: %d x %d\n", midRow, midCol); }

	//Ensure that we have sufficient pixels to capture.
	if ((midRow < numOfCntrPixels / 2) or (midCol < numOfCntrPixels / 2))
	{
		printf("Cannot process baseline technique for this image due to small size of the image.\n");
		exit(-400);
	}

	// Ensure that feature vector is overwritten. 
	featureVector.clear();
	for (int rowIncr = -numOfCntrPixels / 2; rowIncr < (numOfCntrPixels / 2) + 1; rowIncr++)
	{
		// Row pointer for easy of acceess
		cv::Vec3b* rowPtr = image.ptr<cv::Vec3b>(midRow + rowIncr);
		for (int colIncr = -numOfCntrPixels / 2; colIncr < (numOfCntrPixels / 2) + 1; colIncr++)
		{
			// Push the RGB channel values of the pixel to the feature vector. 
			featureVector.push_back(rowPtr[midCol + colIncr][2]); // Red Channel value
			featureVector.push_back(rowPtr[midCol + colIncr][1]); // Green Channel value
			featureVector.push_back(rowPtr[midCol + colIncr][0]); // Blue Channel value	
		}
	}

	// Mark the process as successful. 
	if (echoStatus) { printf("Successfully populated the feature vector.\n"); }
	return 0;
}


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
int rg2DHistogramTechnique(cv::Mat& image, std::vector<float>& featureVector, int histBins = 16, bool echoStatus = false) {
	if (echoStatus) { printf("Dimensions of the image: %d x %d\n", image.rows, image.cols); }
	// Histogram Configuration
	int numberOfPixels = image.rows * image.cols;
	if (echoStatus) { printf("Number of pixels in this image: %d\n", numberOfPixels); }

	// Temporarily store the histogram values. 
	std::vector<std::vector<float>> histogramVector(histBins + 1, std::vector<float>(histBins + 1, 0.0));

	// Calculate r_value, g_value, b go for the image pixels. 
	for (int row = 0; row < image.rows; row++)
	{	// Row pointer for faster access an iterate overall rows. 
		cv::Vec3b* rowPtr = image.ptr<cv::Vec3b>(row);
		for (int col = 0; col < image.cols; col++)
		{
			// Histogram in rg chromaticity for each pixel
			int B = rowPtr[col][0]; // Blue Channel 
			int G = rowPtr[col][1]; // Green Channel
			int R = rowPtr[col][2]; // Red Channel
			// Adding 10e-7 to ensure non-zero denominator.
			float r_value = R / (B + G + R + 0.0000001);
			float g_value = G / (B + G + R + 0.0000001);
			int r_index = r_value * (histBins);
			int g_index = g_value * (histBins);
			if (echoStatus) {
				printf("Original pixel values(R,G,B) are (%d,%d,%d), inds are (%d, %d)\n", R, G, B, r_index, g_index);
			}
			// Update the frequency of the color in the histogramVector
			histogramVector[r_index][g_index] += 1.0;
		}
	}

	featureVector.clear(); // To ensure that we load properly. 
	// Get histogram from the histogramVector, normalize and store in the featureVector
	for (int row = 0; row < histBins + 1; row++)
	{
		for (int col = 0; col < histBins + 1; col++) {
			if (echoStatus) {
				printf("Histogram value for color with r_bin:%d, g_bin:%d is %.04f", row, col, histogramVector[row][col]);
			}
			//Normalize the value before pushing back
			featureVector.push_back(histogramVector[row][col] / numberOfPixels);
		}
	}
	// Mark the process as successful. 
	return 0;
}


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
int rgb3DHistogramTechnique(cv::Mat& image, std::vector<float>& featureVector, int histBins = 8, bool echoStatus = false) {

	// Normalization parameter
	if (echoStatus) { printf("Dimensions of the image: %d x %d\n", image.rows, image.cols); }
	float numberOfPixels = image.rows * image.cols;
	if (echoStatus) { printf("Number of pixels in this image: %.00f\n", numberOfPixels); }

	// Histogram Configuration
	// Calcluate the Bin size to use
	if (echoStatus) { printf("\nUsing %d bins for each color.\n", histBins); }
	int binSize = getBinSize(histBins);
	if (echoStatus) { printf("\nBinsize:%d", binSize); }

	// 3D Array to store the frequencies of the color
	std::vector<std::vector<std::vector<float>>> histogramVector(histBins, std::vector<std::vector<float>>(histBins, std::vector<float>(histBins, 0)));

	// Iterate over all pixels for the frequency
	for (int rowIncr = 0; rowIncr < image.rows; rowIncr++)
	{	// 
		cv::Vec3b* rowPtr = image.ptr<cv::Vec3b>(rowIncr);
		for (int col = 0; col < image.cols; col++)
		{
			// Calculating the bin for the color
			int r = rowPtr[col][2]; // Red Channel
			int g = rowPtr[col][1]; // Green Channel
			int b = rowPtr[col][0]; // Blue Channel 
			if (echoStatus) { printf("Channel values: r=%d, g=%d, b=%d\n", r, g, b); }
			int R = (r / binSize);  // red bin
			int G = (g / binSize);  // green bin
			int B = (b / binSize);  // blue bin
			if (echoStatus) { printf("Channel bin values: R=%d, G=%d, B=%d\n", R, G, B); }

			//Update the frequency of the color.
			histogramVector[R][G][B] += 1.0;
		}
	}

	featureVector.clear(); // To ensure that we load properly. 

	// Get histogram from the histogramVector, normalize and store in the featureVector
	for (int red_bin = 0; red_bin < histBins; red_bin++) {
		for (int green_bin = 0; green_bin < histBins; green_bin++) {
			for (int blue_bin = 0; blue_bin < histBins; blue_bin++) {
				featureVector.push_back(histogramVector[red_bin][green_bin][blue_bin] / numberOfPixels);
			}
		}
	}
	// Mark the storing of the feature vector as successful. 
	return 0;
}


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
 *		 This implementation is independent of the rgb3DHistogramTechnique, rgHistogramTechniques.
 * Method: break the image into two halves, one for the top and other for the bottom.
 * for each half of the image,
 *		for each pixel, the bin to which the value belongs is determined by
 *			r_index  = R/binSize.
 *			g_index  = G/binSize.
 *			b_index  = B/binSize. where R, G, B are values of the pixel in Red, Green and Blue Channels.
 *	Finally the normalize the histogram before pushing to feature vector in the order of top half first and bottom half last.
 * @returns	   0 if the processing is successful
*/
int twoHalvesUpperBottomApproach(cv::Mat& image, std::vector<float>& featureVector, int histBins = 8, bool echoStatus = false) {
	// Calcluate the Bin size to use
	if (echoStatus) { printf("Using %d bins for each color.\n", histBins); }
	int binSize = getBinSize(histBins);
	if (echoStatus) { printf("Binsize:%d\n", binSize); }

	// Histogram Configuration
	float normalizationFactor = (image.rows * image.cols); //

	// 3D Array to store the frequencies of the color
	std::vector<std::vector<std::vector<float>>> histogramVector1(histBins, std::vector<std::vector<float>>(histBins, std::vector<float>(histBins, 0)));

	// Iterate over hirst half for histogram
	for (int rowIncr = 0; rowIncr < (image.rows / 2); rowIncr++)
	{	// Row pointer for quicker access 
		cv::Vec3b* rowPtr = image.ptr<cv::Vec3b>(rowIncr);
		for (int col = 0; col < image.cols; col++)
		{
			// Calculating the bin for the color
			int r = rowPtr[col][2]; // Red Channel
			int g = rowPtr[col][1]; // Green Channel
			int b = rowPtr[col][0]; // Blue Channel 
			if (echoStatus) { printf("Channel values: r=%d, g=%d, b=%d\n", r, g, b); }
			int R = (r / binSize);  // red bin
			int G = (g / binSize);  // green bin
			int B = (b / binSize);  // blue bin
			if (echoStatus) { printf("Channel bin values: R=%d, G=%d, B=%d\n", R, G, B); }
			histogramVector1[R][G][B] += 1.0;
		}
	}

	// Get histogram from the vector and store in the featureVector
	for (int red_bin = 0; red_bin < histBins; red_bin++) {
		for (int green_bin = 0; green_bin < histBins; green_bin++) {
			for (int blue_bin = 0; blue_bin < histBins; blue_bin++) {
				// Normalize before pushing into the featureVector
				featureVector.push_back(histogramVector1[red_bin][green_bin][blue_bin] / normalizationFactor);
			}
		}
	}

	// Another frequency vector as frequency needs to be recalculated.
	std::vector<std::vector<std::vector<float>>> histogramVector2(histBins, std::vector<std::vector<float>>(histBins, std::vector<float>(histBins, 0)));

	// Iterate over second half for the histogram
	for (int rowIncr = image.rows / 2; rowIncr < image.rows; rowIncr++)
	{	// Row pointer for quicker access 
		cv::Vec3b* rowPtr = image.ptr<cv::Vec3b>(rowIncr);
		for (int col = 0; col < image.cols; col++)
		{
			// Calculating the bin for the color
			int r = rowPtr[col][2]; // Red Channel
			int g = rowPtr[col][1]; // Green Channel
			int b = rowPtr[col][0]; // Blue Channel 
			int R = (r / binSize);  // red bin
			int G = (g / binSize);  // green bin
			int B = (b / binSize);  // blue bin
			//printf("\nR=%d, G=%d, B=%d", R, G, B);
			histogramVector2[R][G][B] += 1.0;
		}
	}

	// Get histogram from the vector and store in the featureVector
	for (int red_bin = 0; red_bin < histBins; red_bin++) {
		for (int green_bin = 0; green_bin < histBins; green_bin++) {
			for (int blue_bin = 0; blue_bin < histBins; blue_bin++) {
				featureVector.push_back(histogramVector2[red_bin][green_bin][blue_bin] / normalizationFactor);
			}
		}
	}

	return 0;
}


/*This function provides the normalized histogram as 2 separate halves in RGB space of complete image passed as the feature vector.
 * @param image address of the cv::Mat object to be processed for this feature
 * @param featuerVector vector to store the normalized histogram of the image in RGB space of two halves.
 * @param histBins number of bins to be used for each color.
 * @param echoStatus set this bool to enable verbose
 * @note featureVector will be cleared before loading the featureVector details.
 *		 This function will append the normalized histograms of two halves to the featureVector.
 *		 First half will have RGB histogram of the left half of the image.
 *		 Next half will have RGB histogram of the right half of the image.
 *		 Feature Vector for either half will hold the value in r, g, b values
 *			i.e. order will be in (0,0, 0), (0,0,1),...,(0,1,0),(0,1,1),...(1,0,0), (1,0,1),....
 *		 This function utilizes the getBinSize(histBins) function to get appropriate binSize.
 *		 This implementation is independent of the rgb3DHistogramTechnique, rgHistogramTechniques.
 * Method: break the image into two halves, one for the top and other for the bottom.
 * for each half of the image,
 *		for each pixel, the bin to which the value belongs is determined by
 *			r_index  = R/binSize.
 *			g_index  = G/binSize.
 *			b_index  = B/binSize. where R, G, B are values of the pixel in Red, Green and Blue Channels.
 *	Finally the normalize the histogram before pushing to feature vector in the order of left half first and right half last.
 * @returns	   0 if the processing is successful
*/
int twoHalvesLeftRightApproach(cv::Mat& image, std::vector<float>& featureVector, int histBins = 8, bool echoStatus = false) {
	// Calcluate the Bin size to use
	if (echoStatus) { printf("Using %d bins for each color.\n", histBins); }
	int binSize = getBinSize(histBins);
	if (echoStatus) { printf("Binsize:%d\n", binSize); }

	// Histogram Configuration
	float normalizationFactor = (image.rows * image.cols);
	// normalization factor is number of Pixels to ensure min(histograms) do not exceed 0.5

	// 3D Array to store the frequencies of the color
	std::vector<std::vector<std::vector<float>>> histogramVector1(histBins, std::vector<std::vector<float>>(histBins, std::vector<float>(histBins, 0)));

	// Iterate over left half for histogram
	for (int rowIncr = 0; rowIncr < image.rows; rowIncr++)
	{	// Row pointer for quicker access 
		cv::Vec3b* rowPtr = image.ptr<cv::Vec3b>(rowIncr);
		for (int col = 0; col < image.cols / 2; col++)
		{
			// Calculating the bin for the color
			int r = rowPtr[col][2]; // Red Channel
			int g = rowPtr[col][1]; // Green Channel
			int b = rowPtr[col][0]; // Blue Channel 
			if (echoStatus) { printf("Channel values: r=%d, g=%d, b=%d\n", r, g, b); }
			int R = (r / binSize);  // red bin
			int G = (g / binSize);  // green bin
			int B = (b / binSize);  // blue bin
			if (echoStatus) { printf("Channel bin values: R=%d, G=%d, B=%d\n", R, G, B); }
			histogramVector1[R][G][B] += 1.0;
		}
	}

	// Get histogram from the vector and store in the featureVector
	for (int red_bin = 0; red_bin < histBins; red_bin++) {
		for (int green_bin = 0; green_bin < histBins; green_bin++) {
			for (int blue_bin = 0; blue_bin < histBins; blue_bin++) {
				// Normalize before pushing into the featureVector
				featureVector.push_back(histogramVector1[red_bin][green_bin][blue_bin] / normalizationFactor);
			}
		}
	}

	// Another frequency vector as frequency needs to be recalculated.
	std::vector<std::vector<std::vector<float>>> histogramVector2(histBins, std::vector<std::vector<float>>(histBins, std::vector<float>(histBins, 0)));

	// Iterate over right half for the histogram
	for (int rowIncr = 0; rowIncr < image.rows; rowIncr++)
	{	// Row pointer for quicker access 
		cv::Vec3b* rowPtr = image.ptr<cv::Vec3b>(rowIncr);
		for (int col = image.cols / 2; col < image.cols; col++)
		{
			// Calculating the bin for the color
			int r = rowPtr[col][2]; // Red Channel
			int g = rowPtr[col][1]; // Green Channel
			int b = rowPtr[col][0]; // Blue Channel 
			int R = (r / binSize);  // red bin
			int G = (g / binSize);  // green bin
			int B = (b / binSize);  // blue bin
			//printf("\nR=%d, G=%d, B=%d", R, G, B);
			histogramVector2[R][G][B] += 1.0;
		}
	}

	// Get histogram from the vector and store in the featureVector
	for (int red_bin = 0; red_bin < histBins; red_bin++) {
		for (int green_bin = 0; green_bin < histBins; green_bin++) {
			for (int blue_bin = 0; blue_bin < histBins; blue_bin++) {
				featureVector.push_back(histogramVector2[red_bin][green_bin][blue_bin] / normalizationFactor);
			}
		}
	}

	return 0;
}


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
int gradientAndColorHistApproach(cv::Mat& image, std::vector<float>& featureVector, int histBins = 8, bool echoStatus = false) {
	// Get the gradient magnitude of the image provided
	// Get the vertical edges in the image using SobelX filter
	cv::Mat sobelXImg = cv::Mat(image.size(), CV_16SC3);
	sobelX3x3(image, sobelXImg);
	// Get the horizontal edges in the image using SobelY filter
	cv::Mat sobelYImg = cv::Mat(image.size(), CV_16SC3);
	sobelY3x3(image, sobelYImg);
	// Get the gradient magnitude of the image based on SobelX and SobelY filtered images
	cv::Mat magnitudeImage = cv::Mat(image.size(), CV_8UC3);
	magnitude(sobelXImg, sobelYImg, magnitudeImage);

	// 1st feature vector capturing the texture of the complete image.
	std::vector<float> firstFeatureVector;
	rgb3DHistogramTechnique(magnitudeImage, firstFeatureVector, histBins);

	// 2nd feature vector capturing the color of the complete image.
	std::vector<float> secondFeatureVector;
	rgb3DHistogramTechnique(image, secondFeatureVector, histBins, false);

	// Collect the features of the image into the final feature vector 
	for (int indx = 0; indx < firstFeatureVector.size(); indx++)
	{	// Push with weight of the feature vector as 0.5
		featureVector.push_back(firstFeatureVector[indx] / 2);
	}
	for (int indx = 0; indx < secondFeatureVector.size(); indx++)
	{	// Push with weight of the feature vector as 0.5
		featureVector.push_back(secondFeatureVector[indx] / 2);
	}
	// Mark this function as successful.
	return 0;
}


/** This function utilizes both the hitograms of 4 quarters and also the Laplaciant for the whole image.
 * @param image address of the cv::Mat object to be processed for this feature
 * @param featuerVector vector to store the normalized histogram of the 4 quarters of the image in RGB space
 * @param histBins number of bins to be used for each color.
 * @param echoStatus set this bool to enable verbose
 * @note featureVector will be cleared before loading the featureVector details.
 *		 This function will append the normalized histograms of gradient magnitude and RGB colorspace to the featureVector.
 *		 Quarters to process will start from top left , traverse to right and then moves down to reach the bottom right.
 *		 Feature Vector for either half will hold the value in r, g, b values
 *			i.e. order will be in (0,0, 0), (0,0,1),...,(0,1,0),(0,1,1),...(1,0,0), (1,0,1),....
 *		 This function utilizes the getBinSize(histBins) function to get appropriate binSize.
 *		 This function also utilizes the filters openCV's Laplacian for the texture estimation.
*/
int quartersAndTextureApproach(cv::Mat& image, std::vector<float>& featureVector, int histBins = 8, bool echoStatus = false) {

	// Calcluate the Bin size to use
	if (echoStatus) { printf("Using %d bins for each color.\n", histBins); }
	int binSize = getBinSize(histBins);
	if (echoStatus) { printf("Binsize:%d\n", binSize); }

	float normalizationFactor = (image.rows * image.cols) * 2;

	// 3D Array to store the frequencies of the color
	std::vector<std::vector<std::vector<float>>> histogramVector1(histBins, std::vector<std::vector<float>>(histBins, std::vector<float>(histBins, 0)));
	// Iterate over first quarter for histogram
	for (int rowIncr = 0; rowIncr < (image.rows / 2); rowIncr++)
	{	// Row pointer for quicker access 
		cv::Vec3b* rowPtr = image.ptr<cv::Vec3b>(rowIncr);
		for (int col = 0; col < image.cols / 2; col++)
		{
			// Calculating the bin for the color
			int r = rowPtr[col][2]; // Red Channel
			int g = rowPtr[col][1]; // Green Channel
			int b = rowPtr[col][0]; // Blue Channel 
			if (echoStatus) { printf("Channel values: r=%d, g=%d, b=%d\n", r, g, b); }
			int R = (r / binSize);  // red bin
			int G = (g / binSize);  // green bin
			int B = (b / binSize);  // blue bin
			if (echoStatus) { printf("Channel bin values: R=%d, G=%d, B=%d\n", R, G, B); }
			histogramVector1[R][G][B] += 1.0;
		}
	}

	// Get histogram from the vector and store in the featureVector
	for (int red_bin = 0; red_bin < histBins; red_bin++) {
		for (int green_bin = 0; green_bin < histBins; green_bin++) {
			for (int blue_bin = 0; blue_bin < histBins; blue_bin++) {
				// Normalize before pushing into the featureVector
				featureVector.push_back(histogramVector1[red_bin][green_bin][blue_bin] / normalizationFactor);
			}
		}
	}

	// 3D Array to store the frequencies of the color
	std::vector<std::vector<std::vector<float>>> histogramVector2(histBins, std::vector<std::vector<float>>(histBins, std::vector<float>(histBins, 0)));
	// Iterate over first quarter for histogram
	for (int rowIncr = 0; rowIncr < (image.rows / 2); rowIncr++)
	{	// Row pointer for quicker access 
		cv::Vec3b* rowPtr = image.ptr<cv::Vec3b>(rowIncr);
		for (int col = image.cols / 2; col < image.cols; col++)
		{
			// Calculating the bin for the color
			int r = rowPtr[col][2]; // Red Channel
			int g = rowPtr[col][1]; // Green Channel
			int b = rowPtr[col][0]; // Blue Channel 
			if (echoStatus) { printf("Channel values: r=%d, g=%d, b=%d\n", r, g, b); }
			int R = (r / binSize);  // red bin
			int G = (g / binSize);  // green bin
			int B = (b / binSize);  // blue bin
			if (echoStatus) { printf("Channel bin values: R=%d, G=%d, B=%d\n", R, G, B); }
			histogramVector2[R][G][B] += 1.0;
		}
	}
	// Get histogram from the vector and store in the featureVector
	for (int red_bin = 0; red_bin < histBins; red_bin++) {
		for (int green_bin = 0; green_bin < histBins; green_bin++) {
			for (int blue_bin = 0; blue_bin < histBins; blue_bin++) {
				// Normalize before pushing into the featureVector
				featureVector.push_back(histogramVector2[red_bin][green_bin][blue_bin] / normalizationFactor);
			}
		}
	}

	// 3D Array to store the frequencies of the color
	std::vector<std::vector<std::vector<float>>> histogramVector3(histBins, std::vector<std::vector<float>>(histBins, std::vector<float>(histBins, 0)));
	// Iterate over second quarter for histogram
	for (int rowIncr = image.rows / 2; rowIncr < image.rows; rowIncr++)
	{	// Row pointer for quicker access 
		cv::Vec3b* rowPtr = image.ptr<cv::Vec3b>(rowIncr);
		for (int col = 0; col < image.cols / 2; col++)
		{
			// Calculating the bin for the color
			int r = rowPtr[col][2]; // Red Channel
			int g = rowPtr[col][1]; // Green Channel
			int b = rowPtr[col][0]; // Blue Channel 
			if (echoStatus) { printf("Channel values: r=%d, g=%d, b=%d\n", r, g, b); }
			int R = (r / binSize);  // red bin
			int G = (g / binSize);  // green bin
			int B = (b / binSize);  // blue bin
			if (echoStatus) { printf("Channel bin values: R=%d, G=%d, B=%d\n", R, G, B); }
			histogramVector3[R][G][B] += 1.0;
		}
	}
	// Get histogram from the vector and store in the featureVector
	for (int red_bin = 0; red_bin < histBins; red_bin++) {
		for (int green_bin = 0; green_bin < histBins; green_bin++) {
			for (int blue_bin = 0; blue_bin < histBins; blue_bin++) {
				// Normalize before pushing into the featureVector
				featureVector.push_back(histogramVector2[red_bin][green_bin][blue_bin] / normalizationFactor);
			}
		}
	}

	// 3D Array to store the frequencies of the color
	std::vector<std::vector<std::vector<float>>> histogramVector4(histBins, std::vector<std::vector<float>>(histBins, std::vector<float>(histBins, 0)));
	// Iterate over third quarter for histogram
	for (int rowIncr = image.rows / 2; rowIncr < image.rows; rowIncr++)
	{	// Row pointer for quicker access 
		cv::Vec3b* rowPtr = image.ptr<cv::Vec3b>(rowIncr);
		for (int col = image.cols / 2; col < image.cols; col++)
		{
			// Calculating the bin for the color
			int r = rowPtr[col][2]; // Red Channel
			int g = rowPtr[col][1]; // Green Channel
			int b = rowPtr[col][0]; // Blue Channel 
			if (echoStatus) { printf("Channel values: r=%d, g=%d, b=%d\n", r, g, b); }
			int R = (r / binSize);  // red bin
			int G = (g / binSize);  // green bin
			int B = (b / binSize);  // blue bin
			if (echoStatus) { printf("Channel bin values: R=%d, G=%d, B=%d\n", R, G, B); }
			histogramVector4[R][G][B] += 1.0;
		}
	}
	// Get histogram from the vector and store in the featureVector
	for (int red_bin = 0; red_bin < histBins; red_bin++) {
		for (int green_bin = 0; green_bin < histBins; green_bin++) {
			for (int blue_bin = 0; blue_bin < histBins; blue_bin++) {
				// Normalize before pushing into the featureVector
				featureVector.push_back(histogramVector4[red_bin][green_bin][blue_bin] / normalizationFactor);
			}
		}
	}

	// Get the edges in the image.
	cv::Mat textureImage;
	cv::Laplacian(image, textureImage, 5);

	std::vector<float> lastFeatureVector;
	rgb3DHistogramTechnique(textureImage, lastFeatureVector, histBins, false);

	for (int index = 0; index < lastFeatureVector.size(); index++)
	{
		featureVector.push_back(lastFeatureVector[index] / normalizationFactor);
	}
	return 0;
}


/** This function utilizes histograms of the center image 15x15 area and textures with 30%, 70% weightage.
 * @param image address of the cv::Mat object to be processed for this feature
 * @param featureVector vector to store the normalized histogram of the 4 quarters of the image in RGB space
 * @param histBins number of bins to be used for each color.
 * @param echoStatus set this bool to enable verbose
 * @note featureVector will be cleared before loading the featureVector details.
 *		 This function will append the normalized histograms of center part of image and RGB colorspace to the featureVector.
 *		 Feature Vector for either half will hold the value in r, g, b values
 *			i.e. order will be in (0,0, 0), (0,0,1),...,(0,1,0), (0,1,1),...(1,0,0), (1,0,1),....
 *		 This function utilizes the getBinSize(histBins) function to get appropriate binSize.
 *		 This function uses OpenCV's laplacian function for edge detection as the texture.
*/
int centerColorAndTextureApproach(cv::Mat& image, std::vector<float>& featureVector, int histBins = 8, bool echoStatus = false) {

	std::vector<float> centerColorFeatureVector;

	float normalizationFactor = (image.rows * image.cols) * 2;
	baselineTechnique(image, centerColorFeatureVector, 15, 0);

	featureVector.clear();
	for (int indx = 0; indx < centerColorFeatureVector.size(); indx++)
	{
		featureVector.push_back((0.3 * centerColorFeatureVector[indx]) / normalizationFactor);
	}

	// Get the gradient magnitude of the image provided
	// Get the edges using the Laplacian images.
	cv::Mat laplacianImg;
	cv::Laplacian(image, laplacianImg, 5);

	// 1st feature vector capturing the texture of the complete image.
	std::vector<float> textureHistogramFeatureVector;
	rgb3DHistogramTechnique(laplacianImg, textureHistogramFeatureVector, histBins);

	for (int indx = 0; indx < textureHistogramFeatureVector.size(); indx++)
	{
		featureVector.push_back((0.7 * textureHistogramFeatureVector[indx]) / normalizationFactor);
	}
	return 0;
}


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
int edgesAndColorHistApproach(cv::Mat& image, std::vector<float>& featureVector, int histBins = 8, bool echoStatus = false) {

	// Obtain greyscale  of the image
	cv::Mat grayscaleImage = cv::Mat(image.size(), CV_8UC1);
	cv::cvtColor(image, grayscaleImage, cv::COLOR_BGR2GRAY);

	//Reduce the noise in the image using blur operation
	cv::Mat blurImg;
	cv::Size blurSize;
	blurSize.height = 5;
	blurSize.width = 5;
	cv::GaussianBlur(grayscaleImage, blurImg, blurSize, 0.0);

	// Get the gradient magnitude of the image provided
	cv::Mat edgesDetected;
	cv::Canny(image, edgesDetected, 86, 173);

	// 1st feature vector capturing the texture of the complete image.
	std::vector<float> firstFeatureVector;
	rgb3DHistogramTechnique(edgesDetected, firstFeatureVector, histBins);

	// 2nd feature vector capturing the color of the complete image.
	std::vector<float> secondFeatureVector;
	rgb3DHistogramTechnique(image, secondFeatureVector, histBins, false);

	// Collect the features of the image into the final feature vector 
	for (int indx = 0; indx < firstFeatureVector.size(); indx++)
	{	// Push with weight of the feature vector as 0.5
		featureVector.push_back(firstFeatureVector[indx] / 2);
	}
	for (int indx = 0; indx < secondFeatureVector.size(); indx++)
	{	// Push with weight of the feature vector as 0.5
		featureVector.push_back(secondFeatureVector[indx] / 2);
	}
	// Mark this function as successful.
	return 0;
}


/**This function will be calculated by thresholding based on H,S,V values provided to find particular color
and then calculated the histogram of the image in RGB space to obtain the featureVector.
 * @param image address of the cv::Mat object to be processed for this feature
 * @param featuerVector vector to store the normalized histogram of the image in RGB space of two halves.
 * @param histBins number of bins to be used for each color.
 * @param echoStatus set this bool to enable verbose
 * @note featureVector will be cleared before loading the featureVector details.
 *		 This function will append the normalized histograms of the masked image obtained.
 *		 Feature Vector will hold the value in r, g, b values
 *			i.e. order will be in (0,0, 0), (0,0,1),...,(0,1,0),(0,1,1),...(1,0,0), (1,0,1),....
 *		 This function utilizes the getBinSize(histBins) function to get appropriate binSize.
 * Method: This function first converts the image from BGR to HSV colorspace and then depending upon
 * minimum and maximum values provided to obtain mask image, then dilate the masked image to fill any holes.
 * Finally the function reuses the rgb3DHistogramTechnique to feature vector and
 * push the results back to the feature vector.
 * @returns	   0 if the processing is successful.

*/
int specificColorApproach(cv::Mat& image, std::vector<float>& featureVector, int hueMin = 0, int hueMax = 180, int satMin = 0, int satMax = 255, int valMin = 0, int valMax = 255, int histbins = 8, bool echoStatus = false) {

	// Obtain the HSV format of the image 
	cv::Mat hsvImg = cv::Mat(image.rows, image.cols, CV_8UC3);
	cv::cvtColor(image, hsvImg, cv::COLOR_BGR2HSV);

	// Min and Max thresholds to be used to extract the specific color
	cv::Scalar minValues = cv::Scalar(hueMin, satMin, valMin);
	cv::Scalar maxValues = cv::Scalar(hueMax, satMax, valMax);

	//Obtain the mask of the image. 
	cv::Mat maskImg;
	cv::inRange(hsvImg, minValues, maxValues, maskImg);

	// Using Dilation to fill holes by 3x3 matrix
	cv::Mat dilatedImg;
	maskImg.copyTo(dilatedImg);
	cv::dilate(dilatedImg, maskImg, cv::Mat());

	// Get the contours of the images in the threshold image
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(dilatedImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	////Add the 3D Histogram distribution 
	rgb3DHistogramTechnique(image, featureVector, histbins, echoStatus);

	// Add the normalized area, normalized lengths, normalized widths in the contours.
	float area = 0; // Total area matching in the contours
	float height = 0; // Total height of the contours
	float width = 0; // Total wdith of the contours
	for (size_t i = 0; i < contours.size(); i++)
	{
		cv::Rect boundRect = boundingRect(contours[i]);
		if (boundRect.area() > 10000)
		{
			area += boundRect.area();
			height += boundRect.height;
			width += boundRect.width;
		}
	}

	// Adding the normalized area, lengths and widths as features
	featureVector.push_back(area / (image.rows * image.cols)); //Normalized Area
	height /= image.rows;
	width /= image.cols;
	//featureVector.push_back(height / image.rows); // Normalized height
	//featureVector.push_back(width / image.cols);  // Normalized width
	//width = std::max(1.0f, width); // Ensure non zero division
	featureVector.push_back(height / width); // Width to height ratio is another feature.

	return 0;
}


/**This function uses Yellow coor values on the specificColorApproach.
* @param image address of the image
* @param featureVector address of the feature vector.
* @param histBins[default=8] number of bins to be used for each bin.
* @param echoStatus[default=false] set this bool to enable verbose.
* @returns 0 if featureVector is properly populated
*		  non zero values in case of error.
* @note the values used to obtain the yellowColor of the banana is obtained by experimenting with pic.0343.jpg
*/
int customApproachforBanana(cv::Mat& image, std::vector<float>& featureVector, int histbins = 8, bool echoStatus = false) {
	// Min and Max thresholds to be used to extract the yellow Color based on experiment with pic.0343.jpg
	return specificColorApproach(image, featureVector, 22, 50, 110, 255, 120, 255, histbins, false);
}


/**This function uses Blue coor values on the specificColorApproach.
* @param image address of the image
* @param featureVector address of the feature vector.
* @param histBins[default=8] number of bins to be used for each bin.
* @param echoStatus[default=false] set this bool to enable verbose.
* @returns 0 if featureVector is properly populated
*		  non zero values in case of error.
* @note the values used to obtain the Blue color of the blue bins is obtained by experimenting with pic.0287.jpg
*/
int customApproachforBlueBins(cv::Mat& image, std::vector<float>& featureVector, int histbins = 8, bool echoStatus = false) {
	// Min and Max thresholds to be used to extract the Blue Color based on experiment with pic.0287.jpg
	return specificColorApproach(image, featureVector, 95, 160, 110, 255, 110, 255, histbins, false);
}


/**This function uses Green color values on the specificColorApproach.
* @param image address of the image
* @param featureVector address of the feature vector.
* @param histBins[default=8] number of bins to be used for each bin.
* @param echoStatus[default=false] set this bool to enable verbose.
* @returns 0 if featureVector is properly populated
*		  non zero values in case of error.
* @note the values used to obtain the Green color of the blue bins is obtained by experimenting with pictures ranging 0746 - 0755
*/
int customApproachforGreenBins(cv::Mat& image, std::vector<float>& featureVector, int histbins = 8, bool echoStatus = false) {
	// Min and Max thresholds to be used to extract the Blue Color based on experiment with pic.0287.jpg
	return specificColorApproach(image, featureVector, 63, 94, 51, 236, 41, 243, histbins, false);
}


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
 * Method: This function reuses the rgb3DHistogramTechnique by processing it for first half and second half
 * and pushing the results back to the feature vector.
 * @returns	   0 if the processing is successful
*/
int twoHalvesRGBApproach(cv::Mat& image, std::vector<float>& featureVector, int histBins = 8, bool echoStatus = false) {
	// Calcluate the Bin size to use
	if (echoStatus) { printf("Using %d bins for each color.\n", histBins); }
	int binSize = getBinSize(histBins);
	if (echoStatus) { printf("Binsize:%d\n", binSize); }

	// Get the featureVector for the top half image
	cv::Mat TopHalf = image(cv::Range(0, image.rows / 2), cv::Range(0, image.cols));
	std::vector<float> ftVecTopHalf;
	rgb3DHistogramTechnique(TopHalf, ftVecTopHalf, histBins, echoStatus);

	// Get histogram from the vector and store in the featureVector
	for (int index = 0; index < ftVecTopHalf.size(); index++) {
		featureVector.push_back(ftVecTopHalf[index]);
	}

	// Get the featureVector for the bottom half image
	cv::Mat BottomHalf = image(cv::Range(image.rows / 2, image.rows), cv::Range(0, image.cols));
	// Another frequency vector as frequency needs to be recalculated.
	std::vector<float> ftVecBottomHalf;
	rgb3DHistogramTechnique(TopHalf, ftVecBottomHalf, histBins, echoStatus);

	// Get histogram from the vector and store in the featureVector
	for (int index = 0; index < ftVecBottomHalf.size(); index++) {
		featureVector.push_back(ftVecBottomHalf[index]);
	}

	// Mark as successful
	return 0;
}


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
 * Method: This function reuses the rg2DHistogramTechnique by processing it for first half and second half
 * and pushing the results back to the feature vector.
 * @returns	   0 if the processing is successful
*/
int twoHalvesRGChromApproach(cv::Mat& image, std::vector<float>& featureVector, int histBins = 8, bool echoStatus = false) {
	// Calcluate the Bin size to use
	if (echoStatus) { printf("Using %d bins for each color.\n", histBins); }
	int binSize = getBinSize(histBins);
	if (echoStatus) { printf("Binsize:%d\n", binSize); }

	// Get the featureVector for the top half image
	cv::Mat TopHalf = image(cv::Range(0, image.rows / 2), cv::Range(0, image.cols));
	std::vector<float> ftVecTopHalf;
	rg2DHistogramTechnique(TopHalf, ftVecTopHalf, histBins, echoStatus);

	// Get histogram from the vector and store in the featureVector
	for (int index = 0; index < ftVecTopHalf.size(); index++) {
		featureVector.push_back(ftVecTopHalf[index]);
	}

	// Get the featureVector for the bottom half image
	cv::Mat BottomHalf = image(cv::Range(image.rows / 2, image.rows), cv::Range(0, image.cols));
	// Another frequency vector as frequency needs to be recalculated.
	std::vector<float> ftVecBottomHalf;
	rg2DHistogramTechnique(TopHalf, ftVecBottomHalf, histBins, echoStatus);

	// Get histogram from the vector and store in the featureVector
	for (int index = 0; index < ftVecBottomHalf.size(); index++) {
		featureVector.push_back(ftVecBottomHalf[index]);
	}

	// Mark as successful
	return 0;
}
