/** This file has the implementations of the extension functions done in this filtering project.
Wrtitten by Samavedam Manikhanta Praphul.
 */

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>


/**This function convolves with positive Laplacian filter 
	*							  | 0 -1  0 |
	* Positive Laplacian filter = |-1  4 -1 | to detect edges in an image. 
	*							  | 0 -1  0 |
	* Laplacian filter is the derivative of the Gaussian filter.
	* @param address of source image (Assumed to be UChar)
	* @param address of destination image (Assumed to be Uchar) with edges detected.
	* @return   0 if the function is successful. 
	*		 -100 if the source and destinations are of different sizes.
	*		 -101 if the source and destinations are of different types.
	* @note: I have divided by 6 instead of 4 in the convolution. 
	*/
int PositiveLaplacianFilter(cv::Mat& src, cv::Mat& dst) {
	
	if (src.size() != dst.size()){return -100;}
	if (src.type() != dst.type()){return -101;}

	for (int row = 1; row < src.rows - 1; row++)
	{  // Looping over rows.
		cv::Vec3b* srcRowP1Ptr = src.ptr<cv::Vec3b>(row + 1);
		cv::Vec3b* srcRowPtr = src.ptr<cv::Vec3b>(row);
		cv::Vec3b* srcRowM1Ptr = src.ptr<cv::Vec3b>(row - 1);
		cv::Vec3b* dstRowPtr = dst.ptr<cv::Vec3b>(row);
		for (int col = 1; col < src.cols - 1; col++)
		{	// Looping over cols. single line for each channel. @note divided by 6 instead of 4. 
			// Reference To-Do Link of the page.
			dstRowPtr[col][0] = ((4 * srcRowPtr[col][0]) -srcRowM1Ptr[col][0] - srcRowPtr[col - 1][0] - srcRowPtr[col + 1][0] - srcRowP1Ptr[col][0]) / 6;
			dstRowPtr[col][1] = ((4 * srcRowPtr[col][1]) -srcRowM1Ptr[col][1] - srcRowPtr[col - 1][1] - srcRowPtr[col + 1][1] - srcRowP1Ptr[col][1]) / 6;
			dstRowPtr[col][2] = ((4 * srcRowPtr[col][2]) -srcRowM1Ptr[col][2] - srcRowPtr[col - 1][2] - srcRowPtr[col + 1][2] - srcRowP1Ptr[col][2]) / 6;
		}
	}
	return 0;
}

/** This function convolves with negative Laplacian filter
	*							  | 0  1  0 |
	* Positive Laplacian filter = | 1 -4  1 | to detect edges in an image.
	*							  | 0  1  0 |
	* Laplacian filter is the derivative of the Gaussian filter.
	* @param address of source image (Assumed to be UChar)
	* @param address of destination image (Assumed to be Uchar) with edges detected.
	* @return   0 if the function is successful.
	*		 -100 if the source and destinations are of different sizes.
	*		 -101 if the source and destinations are of different types.
	* @note: I have divided by 6 instead of 4 in the convolution.
	*/
int NegativeLaplacianFilter(cv::Mat& src, cv::Mat& dst) {
	
	if (src.size() != dst.size()) { return -100; }
	if (src.type() != dst.type()) { return -101; }

	for (int row = 1; row < src.rows - 1; row++)
	{  // Looping over rows.
		cv::Vec3b* srcRowP1Ptr = src.ptr<cv::Vec3b>(row + 1);
		cv::Vec3b* srcRowPtr = src.ptr<cv::Vec3b>(row);
		cv::Vec3b* srcRowM1Ptr = src.ptr<cv::Vec3b>(row - 1);
		cv::Vec3b* dstRowPtr = dst.ptr<cv::Vec3b>(row);
		for (int col = 1; col < src.cols - 1; col++)
		{	// Looping over cols. single line for each channel. @note divided by 6 instead of 4. 
			// Reference To-Do Link of the page.
			dstRowPtr[col][0] = (srcRowM1Ptr[col][0] + srcRowPtr[col - 1][0] + (-4 * srcRowPtr[col][0]) + srcRowPtr[col + 1][0] + srcRowP1Ptr[col][0]) / 6;
			dstRowPtr[col][1] = (srcRowM1Ptr[col][1] + srcRowPtr[col - 1][1] + (-4 * srcRowPtr[col][1]) + srcRowPtr[col + 1][1] + srcRowP1Ptr[col][1]) / 6;
			dstRowPtr[col][2] = (srcRowM1Ptr[col][2] + srcRowPtr[col - 1][2] + (-4 * srcRowPtr[col][2]) + srcRowPtr[col + 1][2] + srcRowP1Ptr[col][2]) / 6;
		}
	}
	return 0;
}

/** This function applies box blur of 3x3 dimenstion to the image using 2 separable filters approach.
	*		               | 1 1 1 |
	*	 5x5 blur filter = | 1 1 1 |
	*		               | 1 1 1 |
	* @param address of source image (Assumed to be UChar)
	* @param address of destination image (Assumed to be Uchar) with blur effect.
	* @return   0 if the function is successful.
	*		 -100 if the source and destinations are of different sizes.
	*		 -101 if the source and destinations are of different types.
	* @note This function Will be implemented as 2 separable 1 - D filters
	* 1. (1 x 3) filter and 2. (3 x 1) filter
	*						    | 1 |
	*	[ 1, 1, 1]      and     | 1 |.
	*						    | 1 |
	*/
int boxBlur3x3(cv::Mat& src, cv::Mat& dst) {
	
	if (src.size() != dst.size()) { return -100; }
	if (src.type() != dst.type()) { return -101; }
	cv::Mat interim;
	src.copyTo(interim);

	/* Convolving with [1, 1, 1] filter*/
	for (int row = 0; row < src.rows; row++)
	{
		cv::Vec3b* srcRowPtr = src.ptr<cv::Vec3b>(row);
		// Destination is interim
		cv::Vec3b* dstRowPtr = interim.ptr<cv::Vec3b>(row);
		for (int col = 1; col < src.cols - 1; col++)
		{
			dstRowPtr[col][0] = (srcRowPtr[col - 1][0] + srcRowPtr[col][0] + srcRowPtr[col + 1][0]) / 3;
			dstRowPtr[col][1] = (srcRowPtr[col - 1][1] + srcRowPtr[col][1] + srcRowPtr[col + 1][1]) / 3;
			dstRowPtr[col][2] = (srcRowPtr[col - 1][2] + srcRowPtr[col][2] + srcRowPtr[col + 1][2]) / 3;
		}
	}

	/*					| 1 |
		Convolving with | 1 | filter.
						| 1 |
	*/

	for (int row = 1; row < src.rows - 1; row++)
	{
		// Source is interim as it has updated values
		cv::Vec3b* srcRowM1Ptr = interim.ptr<cv::Vec3b>(row - 1);
		cv::Vec3b* srcRowPtr = interim.ptr<cv::Vec3b>(row);
		cv::Vec3b* srcRowP1Ptr = interim.ptr<cv::Vec3b>(row + 1);
		cv::Vec3b* dstRowPtr = dst.ptr<cv::Vec3b>(row);
		for (int col = 0; col < src.cols; col++)
		{
			dstRowPtr[col][0] = (srcRowM1Ptr[col][0] + srcRowPtr[col][0] + srcRowP1Ptr[col][0]) / 3;
			dstRowPtr[col][1] = (srcRowM1Ptr[col][1] + srcRowPtr[col][1] + srcRowP1Ptr[col][1]) / 3;
			dstRowPtr[col][2] = (srcRowM1Ptr[col][2] + srcRowPtr[col][2] + srcRowP1Ptr[col][2]) / 3;
		}
	}
	return 0;
}

/** This function convolves with Median filter to remove salt and pepper noise (almost).
	* Idea: is salt and pepper noises are peak noises which tremendously increase/decrease the pixel value,
	* median value is unaffected by these sorts of salt and pepper noise.
	* Hence median can be used to rid of salt and pepper noise.
	* 
	* @param address of source image to remove salt pepper noise. (Assumed to be UChar)
	* @param address of destination image with out salt pepper noise (almost). (Assumed to be Uchar)
	* @return   0 if the function is successful.
	*		 -100 if the source and destinations are of different sizes.
	*		 -101 if the source and destinations are of different types.
	* @note: Used inefficient way O(NlogN) to find the median value.
	*/
int medianFilter3x3(cv::Mat& src, cv::Mat& dst){
	/* This function removes salt pepper noise in the source image.*/
	if (src.size() != dst.size()) { return -100; }
	if (src.type() != dst.type()) { return -101; }
	for (int row = 1; row < src.rows-1; row++)
	{
		cv::Vec3b* srcRowM1Ptr = src.ptr<cv::Vec3b>(row - 1);
		cv::Vec3b* srcRowPtr = src.ptr<cv::Vec3b>(row);
		cv::Vec3b* srcRowP1Ptr = src.ptr<cv::Vec3b>(row + 1);
		cv::Vec3b* dstRowPtr = dst.ptr<cv::Vec3b>(row);
		for (int col = 1; col < src.cols - 1; col++)
		{
			for (int color = 0; color < 3; color++)
			{ // Create a temporary vector to sort for the middle/median value.
				std::vector<uchar> tmp;
				tmp.push_back(srcRowM1Ptr[col - 1][color]);
				tmp.push_back(srcRowM1Ptr[col][color]);
				tmp.push_back(srcRowM1Ptr[col+1][color]);
				tmp.push_back(srcRowPtr[col - 1][color]);
				tmp.push_back(srcRowPtr[col][color]);
				tmp.push_back(srcRowPtr[col + 1][color]);
				tmp.push_back(srcRowP1Ptr[col - 1][color]);
				tmp.push_back(srcRowP1Ptr[col][color]);
				tmp.push_back(srcRowP1Ptr[col + 1][color]);
				std::sort(tmp.begin(), tmp.end());
				dstRowPtr[col][color] = tmp[5];
			}	
		}
	}
	return 0;
}

/** This function blurs the image appplying the box filter on the 
	image following the brute force mechanism for cross checking the implementation.
			 | 1 1 1| 
	filter = | 1 1 1|
			 | 1 1 1|
	* @param address of source image (Assumed to be UChar)
	* @param address of destination image (Assumed to be Uchar) with blur effect.
	* @return   0 if the function is successful.
	*		 -100 if the source and destinations are of different sizes.
	*		 -101 if the source and destinations are of different types.
	*/
int boxBlur3x3Brute(cv::Mat& src, cv::Mat& dst) {
	if (src.size() != dst.size()) { return -100; }
	if (src.type() != dst.type()) { return -101; }
	
	for (int row = 1; row < src.rows-1; row++)
	{	//Looping over rows in the image
		cv::Vec3b* srcRowPtr = src.ptr<cv::Vec3b>(row);
		cv::Vec3b* srcRowM1Ptr = src.ptr<cv::Vec3b>(row-1);
		cv::Vec3b* srcRowP1Ptr = src.ptr<cv::Vec3b>(row+1);
		cv::Vec3b* dstRowPtr = dst.ptr<cv::Vec3b>(row);
		for (int col = 1; col < src.cols-1; col++)
		{	//Looping over columns in the image
			dstRowPtr[col][0] = (srcRowM1Ptr[col - 1][0] + srcRowM1Ptr[col][0] + srcRowM1Ptr[col + 1][0]
				+ srcRowPtr[col - 1][0] + srcRowPtr[col][0] + +srcRowPtr[col + 1][0]
				+ srcRowP1Ptr[col - 1][0] + srcRowP1Ptr[col][0] + +srcRowP1Ptr[col + 1][0])/9;
			dstRowPtr[col][1] = (srcRowM1Ptr[col - 1][1] + srcRowM1Ptr[col][1] + srcRowM1Ptr[col + 1][1]
				+ srcRowPtr[col - 1][1] + srcRowPtr[col][1] + +srcRowPtr[col + 1][1]
				+ srcRowP1Ptr[col - 1][1] + srcRowP1Ptr[col][1] + +srcRowP1Ptr[col + 1][1]) / 9;
			dstRowPtr[col][2] = (srcRowM1Ptr[col - 1][2] + srcRowM1Ptr[col][2] + srcRowM1Ptr[col + 1][2]
				+ srcRowPtr[col - 1][2] + srcRowPtr[col][2] + +srcRowPtr[col + 1][2]
				+ srcRowP1Ptr[col - 1][2] + srcRowP1Ptr[col][2] + +srcRowP1Ptr[col + 1][2]) / 9;
		}	
	}
	return 0;
}

/** This function applies box blur of 5x5 dimenstion to the image using 2 separable filters approach.
	*		               |1 1 1 1 1|
	*		               |1 1 1 1 1|
	*	 5x5 blur filter = |1 1 1 1 1|
	*		               |1 1 1 1 1|
	*		               |1 1 1 1 1|
	* @param address of source image (Assumed to be UChar)
	* @param address of destination image (Assumed to be Uchar) with blur effect.
	* @return   0 if the function is successful.
	*		 -100 if the source and destinations are of different sizes.
	*		 -101 if the source and destinations are of different types.
	* @note This function Will be implemented as 2 separable 1 - D filters
	* 1. (1 x 5) filter and 2. (5 x 1) filter
	*						| 1 |
	*						| 1 |
	*  [ 1, 1, 1, 1, 1 ] and| 1 |
	*						| 1 |
	*						| 1 |.
	*/
int boxBlur5x5(cv::Mat& src, cv::Mat& dst) {
	
	cv::Mat interim;
	src.copyTo(interim);

	/* Convolving with [1, 1, 1, 1, 1] filter*/
	for (int row = 0; row < src.rows; row++)
	{
		cv::Vec3b* srcRowPtr = src.ptr<cv::Vec3b>(row);
		// Destination is interim
		cv::Vec3b* dstRowPtr = interim.ptr<cv::Vec3b>(row);
		for (int col = 2; col < src.cols - 2; col++)
		{
			dstRowPtr[col][0] = (srcRowPtr[col - 2][0] + srcRowPtr[col - 1][0] + srcRowPtr[col][0] + srcRowPtr[col + 1][0] + srcRowPtr[col + 2][0]) / 5;
			dstRowPtr[col][1] = (srcRowPtr[col - 2][1] + srcRowPtr[col - 1][1] + srcRowPtr[col][1] + srcRowPtr[col + 1][1] + srcRowPtr[col + 2][1]) / 5;
			dstRowPtr[col][2] = (srcRowPtr[col - 2][2] + srcRowPtr[col - 1][2] + srcRowPtr[col][2] + srcRowPtr[col + 1][2] + srcRowPtr[col + 2][2]) / 5;
		}
	}

	/*					| 1 |
						| 1 |
		Convolving with | 1 | filter.
						| 1 |
						| 1 |
	*/

	for (int row = 2; row < src.rows - 2; row++)
	{
		// Source is interim as it has updated values
		cv::Vec3b* srcRowM2Ptr = interim.ptr<cv::Vec3b>(row - 2);
		cv::Vec3b* srcRowM1Ptr = interim.ptr<cv::Vec3b>(row - 1);
		cv::Vec3b* srcRowPtr = interim.ptr<cv::Vec3b>(row);
		cv::Vec3b* srcRowP1Ptr = interim.ptr<cv::Vec3b>(row + 1);
		cv::Vec3b* srcRowP2Ptr = interim.ptr<cv::Vec3b>(row + 2);
		cv::Vec3b* dstRowPtr = dst.ptr<cv::Vec3b>(row);
		for (int col = 0; col < src.cols; col++)
		{
			dstRowPtr[col][0] = (srcRowM2Ptr[col][0] + srcRowM1Ptr[col][0] + srcRowPtr[col][0] + srcRowP1Ptr[col][0] + srcRowP2Ptr[col][0]) / 5;
			dstRowPtr[col][1] = (srcRowM2Ptr[col][1] + srcRowM1Ptr[col][1] + srcRowPtr[col][1] + srcRowP1Ptr[col][1] + srcRowP2Ptr[col][1]) / 5;
			dstRowPtr[col][2] = (srcRowM2Ptr[col][2] + srcRowM1Ptr[col][2] + srcRowPtr[col][2] + srcRowP1Ptr[col][2] + srcRowP2Ptr[col][2]) / 5;
		}
	}
	return 0;
}

/** This function blurs the image appplying the box filter on the image.
	* 			| 1 1 1 1 1|
	* 			| 1 1 1 1 1|
	* filter =  | 1 1 1 1 1|
	*			| 1 1 1 1 1|
	*			| 1 1 1 1 1|
	*/
int boxBlur5x5Brute(cv::Mat& src, cv::Mat& dst) {

	for (int row = 2; row < src.rows - 2; row++)
	{ //Loop over all the rows
		cv::Vec3b* srcRowPtr = src.ptr<cv::Vec3b>(row);
		cv::Vec3b* srcRowM1Ptr = src.ptr<cv::Vec3b>(row - 1);
		cv::Vec3b* srcRowM2Ptr = src.ptr<cv::Vec3b>(row - 2);
		cv::Vec3b* srcRowP1Ptr = src.ptr<cv::Vec3b>(row + 1);
		cv::Vec3b* srcRowP2Ptr = src.ptr<cv::Vec3b>(row + 2);
		cv::Vec3b* dstRowPtr = dst.ptr<cv::Vec3b>(row);
		for (int col = 2; col < src.cols - 2; col++)
		{ // Loop over all the cols
			for (int color = 0; color < 3; color++)
			{ // Loop over all colors
				dstRowPtr[col][color] = (srcRowM2Ptr[col - 2][color] + srcRowM2Ptr[col - 1][color] + srcRowM2Ptr[col][color] + srcRowM2Ptr[col + 1][color] + srcRowM2Ptr[col + 2][color] 
					+ srcRowM1Ptr[col - 2][color] + srcRowM1Ptr[col - 1][color] + srcRowM1Ptr[col][color] + srcRowM1Ptr[col + 1][color] + srcRowM1Ptr[col + 2][color]
					+ srcRowPtr[col - 2][color] + srcRowPtr[col - 1][color] + srcRowPtr[col][color] + srcRowPtr[col + 1][color] + srcRowPtr[col + 2][color]
					+ srcRowP1Ptr[col - 2][color] + srcRowP1Ptr[col - 1][color] + srcRowP1Ptr[col][color] + srcRowP1Ptr[col + 1][color] + srcRowP1Ptr[col + 2][color]
					+ srcRowP2Ptr[col - 2][color] + srcRowP2Ptr[col - 1][color] + srcRowP2Ptr[col][color] + srcRowP2Ptr[col + 1][color] + srcRowP2Ptr[col + 2][color]) / 25;
			}
		}
	}
	return 0;
}

/** This function is better at recognizing the edges in an image.
	* Famously known as Laplacian over the Gaussian Filter (LoG  filter).
	* Idea: Get rid of noise using Gaussian blur, then detect edges using Laplacian.
	* Laplacian filter is the derivative of the Gaussian filter.
	* @param address of source image (Assumed to be UChar)
	* @param address of destination image (Assumed to be Uchar) with edges detected.
	* @return   0 if the function is successful. 
	*		 -100 if the source and destinations are of different sizes.
	*		 -101 if the source and destinations are of different types.
	* @note: I have OpenCV functions for this purpose.
	*/
int gaussianLaplacian(cv::Mat& src, cv::Mat& dst) {
	if (src.size() != dst.size()){return -100;}
	if (src.type() != dst.type()){return -101;}
	cv::Mat interim;
	src.copyTo(interim);
	cv::Size kernelDim;
	kernelDim.height = 3;
	kernelDim.width = 3;
	cv::GaussianBlur(src, interim,kernelDim,1.0);
	cv::Laplacian(interim, dst, 5);
	return 0;
}

/** This function provides a negative image of the image provided.
* 
*/
cv::Mat negativeImage(cv::Mat &given_image) {
    cv::Mat negative_copy;
    negative_copy = cv::Mat::zeros(given_image.size(), CV_16SC3);
    for (int row = 0; row < given_image.rows; row++) {
        cv::Vec3b *rowptr = given_image.ptr<cv::Vec3b>(row);
        // uchar *charptr = negative_copy.ptr<uchar>(row);
        for (int column = 0; column < given_image.cols; column++) {
            /* Inefficient way
            src.at<uchar>(i,j) = 255 - src.at<uchar>(i,j);
            */
            /* Quicker method*/
            rowptr[column][0] = 255 - rowptr[column][0];
            rowptr[column][1] = 255 - rowptr[column][1];
            rowptr[column][2] = 255 - rowptr[column][2];
            /* Alternative way for quick method
            charptr[3*j + 0] = 255 - charptr[3*j + 0];
            charptr[3*j + 1] = 255 - charptr[3*j + 1];
            charptr[3*j + 2] = 255 - charptr[3*j + 2];
            */
        }
    }
    return negative_copy;
}

