#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

/** This function provides the greyscale of the image passed
*/
int greyscale(cv::Mat& src, cv::Mat& dst) {
	for (int row = 0; row < src.rows; row++) {
		cv::Vec3b* srcRPtr = src.ptr<cv::Vec3b>(row);
		cv::Vec3b* dstRPtr = dst.ptr<cv::Vec3b>(row);
		for (int col = 0; col < src.cols; col++)
		{
			uchar agg_val = (0.6*srcRPtr[col][1]) + (0.2*srcRPtr[col][0]) + (0.2*srcRPtr[col][2]);
			dstRPtr[col][0] = agg_val;
			dstRPtr[col][1] = agg_val;
			dstRPtr[col][2] = agg_val;
		}
	}
	return 0;
}

/** 
* This function convolves the given image with 5x5 Gaussian Blur filter.
* 
*						|1  2   4  2  1 |
*						|2  4   8  4  2 |
* 5x5 blur filter  =	|4  8  16  8  4 |
*						|2  4   8  4  2 |
*						|1  2   4  2  1 |
* 
* @param src is the image which needs to be convolved. 
* @param dst is the destination image which is the blurred version of the src. 
* @return 0 if the operation is successful. 
*		 -100 if the src and dst are not of same size.
* 
* @note This function Will be implemented as 2 separable 1-D filters 
* 1. (1 x 5) filter and 2. (5 x 1) filter
*							| 1 |
*							| 2 |
*	   [1, 2, 4, 2, 1] and  | 4 |
* 							| 2 |
* 							| 1 |.
*/
int blur5x5(cv::Mat &src, cv::Mat &dst) {
	
	cv::Mat interim;
	src.copyTo(interim);

	// Applying Convolution by [1, 2, 4, 2, 1] (1 x 5 filter)
	for (int row = 0; row < src.rows; row++)
	{
		// Row pointer for quicker access to rows.
		cv::Vec3b* srcRowPtr = src.ptr<cv::Vec3b>(row);
		cv::Vec3b* interRowPtr = interim.ptr<cv::Vec3b>(row);
		for (int col = 2; col < src.cols - 2; col++)
		{
			// Update the interim matrix with appropriate 
			interRowPtr[col][0] = (srcRowPtr[col - 2][0] + (2 * srcRowPtr[col - 1][0]) + (4 * srcRowPtr[col][0])
				+ (2 * srcRowPtr[col + 1][0]) + srcRowPtr[col + 2][0]) / 10;
			interRowPtr[col][1] = (srcRowPtr[col - 2][1] + (2 * srcRowPtr[col - 1][1]) + (4 * srcRowPtr[col][1])
				+ (2 * srcRowPtr[col + 1][1]) + srcRowPtr[col + 2][1]) / 10;
			interRowPtr[col][2] = (srcRowPtr[col - 2][2] + (2 * srcRowPtr[col - 1][2]) + (4 * srcRowPtr[col][2])
				+ (2 * srcRowPtr[col + 1][2]) + srcRowPtr[col + 2][2]) / 10;
		}
	}

	/* Applying Convolution by (5 x 1 filter)
		| 1 |
		| 2 |
		| 4 |
		| 2 |
		| 1 |
	*/
	for (int row = 2; row < src.rows - 2; row++)
	{
		cv::Vec3b* rowPtr = interim.ptr<cv::Vec3b>(row);
		cv::Vec3b* rowM2Ptr = interim.ptr<cv::Vec3b>(row - 2);
		cv::Vec3b* rowM1Ptr = interim.ptr<cv::Vec3b>(row - 1);
		cv::Vec3b* rowP1Ptr = interim.ptr<cv::Vec3b>(row + 1);
		cv::Vec3b* rowP2Ptr = interim.ptr<cv::Vec3b>(row + 2);
		cv::Vec3b* dstRowPtr = dst.ptr<cv::Vec3b>(row);
		for (int col = 0; col < src.cols; col++)
		{
			dstRowPtr[col][0] = (rowM2Ptr[col][0] + (2 * rowM1Ptr[col][0]) + (4 * rowPtr[col][0]) + (2 * rowP1Ptr[col][0]) + rowP2Ptr[col][0]) / 10;
			dstRowPtr[col][1] = (rowM2Ptr[col][1] + (2 * rowM1Ptr[col][1]) + (4 * rowPtr[col][1]) + (2 * rowP1Ptr[col][1]) + rowP2Ptr[col][1]) / 10;
			dstRowPtr[col][2] = (rowM2Ptr[col][2] + (2 * rowM1Ptr[col][2]) + (4 * rowPtr[col][2]) + (2 * rowP1Ptr[col][2]) + rowP2Ptr[col][2]) / 10;
		}
	}
	return 0;
}

int blur5x5_brute(cv::Mat &src, cv::Mat &dst) {
	/* 5x5 blur filter
	   |1  2   4  2  1 |
	   |2  4   8  4  2 |
	   |4  8  16  8  4 |
	   |2  4   8  4  2 |
	   |1  2   4  2  1 |
	*/

	// Applying 5 x 5 filter 
	for (int row = 2; row < src.rows - 2; row++)
	{
		cv::Vec3b* rowM2Ptr = src.ptr<cv::Vec3b>(row - 2);
		cv::Vec3b* rowM1Ptr = src.ptr<cv::Vec3b>(row - 1);
		cv::Vec3b* rowPtr = src.ptr<cv::Vec3b>(row);
		cv::Vec3b* rowP1Ptr = src.ptr<cv::Vec3b>(row + 1);
		cv::Vec3b* rowP2Ptr = src.ptr<cv::Vec3b>(row + 2);
		cv::Vec3s* dstRowPtr = dst.ptr<cv::Vec3s>(row);
		for (int col = 2; col < src.cols - 2; col++)
		{
			for (int color = 0; color < 3; color++)
			{
				dstRowPtr[col][color] =
					(1 * rowM2Ptr[col - 2][color] + 2 * rowM2Ptr[col - 1][color] + 4 * rowM2Ptr[col][color] + 2 * rowM2Ptr[col + 1][color] + rowM2Ptr[col + 2][color]
						+ 2 * rowM1Ptr[col - 2][color] + 4 * rowM1Ptr[col - 1][color] + 8 * rowM1Ptr[col][color] + 4 * rowM1Ptr[col + 1][color] + 2 * rowM1Ptr[col + 2][color]
						+ 4 * rowPtr[col - 2][color] + 8 * rowPtr[col - 1][color] + 16 * rowPtr[col][color] + 8 * rowPtr[col + 1][color] + 4 * rowPtr[col + 2][color]
						+ 2 * rowP1Ptr[col - 2][color] + 4 * rowP1Ptr[col - 1][color] + 8 * rowP1Ptr[col][color] + 4 * rowP1Ptr[col + 1][color] + 2 * rowP1Ptr[col + 2][color]
						+ 1 * rowP2Ptr[col - 2][color] + 2 * rowP2Ptr[col - 1][color] + 4 * rowP2Ptr[col][color] + 2 * rowP2Ptr[col + 1][color] + rowP2Ptr[col + 2][color]) / 100;
			}
		}
	}
	return 0;
}

int sobelX3x3(cv::Mat &src, cv::Mat &dst) {

	/* This function performs transformation to find vertical edges in an source image.
	 @param src image source which needs to be transformed. 
	 @param dst processed/convolved image.

	 Implementing SobelX filter 
	 | -1 0 1|
	 | -2 0 2|
	 | -1 0 1|
	 as two separable filters of dimensions (1x3) say A and (3x1) say B.
	 |  1 |
	 |  2 | and [-1 0 1]
	 |  1 |
	*/

	/* Implementing convolution with [-1, 0, 1] filter. */
	cv::Mat intermediate = cv::Mat(src.size(), CV_16SC3);
	for (int row = 0; row < src.rows; row++)
	{
		// Pointers for better performance and reduced memory operation.
		cv::Vec3b* srcPtr = src.ptr<cv::Vec3b>(row);
		cv::Vec3s* interPtr = intermediate.ptr<cv::Vec3s>(row);
		for (int col = 0; col < src.cols; col++)
		{
			// For these columns, will copy the data from source
			if (col == 0 or col == src.cols-1)
			{
				interPtr[col][0] = srcPtr[col][0];
				interPtr[col][1] = srcPtr[col][1];
				interPtr[col][2] = srcPtr[col][2];
			}
			else {
				// Applying the convolution with [-1, 0, 1], dropping the col record as its weight is 0
				interPtr[col][0] = (srcPtr[col + 1][0] - srcPtr[col - 1][0]);
				interPtr[col][1] = (srcPtr[col + 1][1] - srcPtr[col - 1][1]);
				interPtr[col][2] = (srcPtr[col + 1][2] - srcPtr[col - 1][2]);
			}
		} // Processed for all columns of the row
	}// Processed all the rows --> Image is processed with this functionality.

	/* 
								  |  1 |
	Implementing convolution with |	 2 | filter.
								  |  1 |
	*/
	for (int row = 1; row < src.rows-1; row++)
	{
		//Intermediate Mat is the source as it has updated values of first convolution.
		cv::Vec3s* rowM1Ptr = intermediate.ptr<cv::Vec3s>(row - 1);
		cv::Vec3s* rowPtr = intermediate.ptr<cv::Vec3s>(row);
		cv::Vec3s* rowP1Ptr = intermediate.ptr<cv::Vec3s>(row + 1);
		cv::Vec3s* dstPtr = dst.ptr<cv::Vec3s>(row);
		for (int col = 0; col < src.cols; col++)
		{
			/*for (int color = 0; color < 3; color++)
			{
				dstPtr[col][color] = (rowM1Ptr[col][color] + (2 * rowPtr[col][color]) + rowP1Ptr[col][color])/4;
			}*/
			dstPtr[col][0] = (rowM1Ptr[col][0] + (2 * rowPtr[col][0]) + rowP1Ptr[col][0]) / 4;
			dstPtr[col][1] = (rowM1Ptr[col][1] + (2 * rowPtr[col][1]) + rowP1Ptr[col][1]) / 4;
			dstPtr[col][2] = (rowM1Ptr[col][2] + (2 * rowPtr[col][2]) + rowP1Ptr[col][2]) / 4;
		} // Processed for all columns of the row
	}// Processed all the rows --> Image is processed with this functionality.	

	return 0;
}

int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
	/* This function performs transformation to find horizontal edges in an source image.
	 @param src source image which needs to be transformed.
	 @param dst processed image.
	 Implementing SobelY filter
	 |  1  2  1|
	 |  0  0  0|
	 | -1 -2 -1|
	 as two separable filters of dimensions (1x3) and (3x1).
	 |  1 |
	 |  0 | and [1 2 1]
	 | -1 |
	*/
	
	/* Implementing convolution with [1, 2, 1] filter. */
	cv::Mat intermediate = cv::Mat(src.size(), CV_16SC3);
	for (int row = 0; row < src.rows; row++)
	{
		// Destination for this single operation is interim Mat
		cv::Vec3b* srcPtr = src.ptr<cv::Vec3b>(row);
		cv::Vec3s* dstPtr = intermediate.ptr<cv::Vec3s>(row);
		for (int col = 0; col < src.cols; col++)
		{
			if (col == 0 or col == src.cols -1)
			{
				dstPtr[col][0] = srcPtr[col][0];
				dstPtr[col][1] = srcPtr[col][1];
				dstPtr[col][2] = srcPtr[col][2];
			}
			else {
				dstPtr[col][0] = (srcPtr[col - 1][0] + (2 * srcPtr[col][0]) + srcPtr[col + 1][0]) / 4;
				dstPtr[col][1] = (srcPtr[col - 1][1] + (2 * srcPtr[col][1]) + srcPtr[col + 1][1]) / 4;
				dstPtr[col][2] = (srcPtr[col - 1][2] + (2 * srcPtr[col][2]) + srcPtr[col + 1][2]) / 4;
			}
		} // Processed for all columns of the row
	}// Processed all the rows --> Image is processed with this functionality.

	/*
								  |  1 |
	Implementing convolution with |	 0 | filter.
								  | -1 |
	*/
	for (int row = 1; row < src.rows - 1; row++)
	{
		// Source for this single operation is interim as it has the convoluted values
		cv::Vec3s* rowM1Ptr = intermediate.ptr<cv::Vec3s>(row - 1);
		cv::Vec3s* rowP1Ptr = intermediate.ptr<cv::Vec3s>(row + 1);
		cv::Vec3s* dstPtr = dst.ptr<cv::Vec3s>(row);
		for (int col = 0; col < src.cols; col++)
		{
			dstPtr[col][0] = rowM1Ptr[col][0] - rowP1Ptr[col][0];
			dstPtr[col][1] = rowM1Ptr[col][1] - rowP1Ptr[col][1];
			dstPtr[col][2] = rowM1Ptr[col][2] - rowP1Ptr[col][2];
		} // Processed for all columns of the row
	}// Processed all the rows --> Image is processed with this functionality.

	return 0;
}

int absolute_image(cv::Mat& src, cv::Mat& dst) {
	src.copyTo(dst);
	for (int row = 0; row < src.rows; row++)
	{
		cv::Vec3s *srcRowPtr = src.ptr<cv::Vec3s>(row);
		cv::Vec3s *dstRowPtr = dst.ptr<cv::Vec3s>(row);
		for (int col = 0; col < src.cols; col++)
		{
			for (int color = 0; color < 3; color++)
			{
				if (srcRowPtr[col][color] < 0) {
					printf("%d, %d, %d has %d", row, col, color, srcRowPtr[col][color]);
					dstRowPtr[col][color] = -srcRowPtr[col][color];
				}
				else {
					dstRowPtr[col][color] = srcRowPtr[col][color];
				}
			}
		}
	}
	return 0;
}

int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst) {
	/* This function calculates the gradient magnitude based on the sobel transformations parameters passed.
	 @param sx SobelX3x3 convolved image address. (Type: CV_16SC3)
	 @param sy SobelY3x3 convolved image address. (Type: CV_16SC3
	 @param dst processsed image address. (Type: CV_8UC3)
	 Implementing Gradient Magnitude as Euclidean distance for magnitude: I = sqrt( sx*sx + sy*sy )
	 @return 0 if the function is successfully applied and param dst is populated.	
	*/
	cv::Mat intermediate = cv::Mat(sx.size(), CV_16SC3);
	for (int row = 0; row < sx.rows; row++)
	{
		cv::Vec3s* sxRowPtr = sx.ptr<cv::Vec3s>(row);
		cv::Vec3s* syRowPtr = sy.ptr<cv::Vec3s>(row);
		cv::Vec3s* dstRowPtr = intermediate.ptr<cv::Vec3s>(row);
		for (int col = 0; col < sx.cols; col++) {
			for (int color = 0; color < 3; color++) {
				dstRowPtr[col][color] = sqrt((sxRowPtr[col][color] * sxRowPtr[col][color]) + (syRowPtr[col][color] * syRowPtr[col][color]));
			}
		}
	}

	cv::convertScaleAbs(intermediate, dst);
	return 0;
}

int sobelX3x3_brute(cv::Mat& src, cv::Mat& dst) {

	/* This function performs transformation to find vertical edges in an source image.
	 arg: src = image source which needs to be transformed.
	 arg: dst = processed image storage.

	 Implementing SobelX filter
	 | -1 0 1|
	 | -2 0 2|
	 | -1 0 1|
	 */

	for (int row = 1; row < src.rows-1; row++)
	{
		cv::Vec3b* srcRowPtr = src.ptr<cv::Vec3b>(row);
		cv::Vec3b* srcRowM1Ptr = src.ptr<cv::Vec3b>(row - 1);
		cv::Vec3b* srcRowP1Ptr = src.ptr<cv::Vec3b>(row + 1);

		cv::Vec3s* dstRowPtr = dst.ptr<cv::Vec3s>(row);

		for (int col = 1; col < src.cols-1; col++)
		{
			for (int color = 0; color < 3; color++)
			{
				dstRowPtr[col][color] = ((srcRowM1Ptr[col + 1][color] - srcRowM1Ptr[col-1][color]) 
					+ (2*(srcRowPtr[col + 1][color] - srcRowPtr[col - 1][color]))
					+ (srcRowP1Ptr[col + 1][color] - srcRowP1Ptr[col - 1][color])) / 4;
			}
		}
	}
	return 0;
}

int quantize(cv::Mat& src, cv::Mat& dst, int levels) {
	if (levels <= 0)
	{
		printf("Levels must be positive.");
		return -1;
	}
	int bucket_size = ceil(255.0 / levels);
	// Ensure that we get proper bin size. 
	// Ceil for levels = 10, we need bin size to be 26 instead of 25 to accomodate all values
	for (int row = 0; row < src.rows; row++)
	{
		cv::Vec3b* srcRowPtr = src.ptr<cv::Vec3b>(row);
		cv::Vec3b* dstRowPtr = dst.ptr<cv::Vec3b>(row);
		for (int col = 0; col < src.cols; col++)
		{
			dstRowPtr[col][0] = floor(srcRowPtr[col][0] / bucket_size) * bucket_size;
			dstRowPtr[col][1] = floor(srcRowPtr[col][1] / bucket_size) * bucket_size;
			dstRowPtr[col][2] = floor(srcRowPtr[col][2] / bucket_size) * bucket_size;
		}
	}
	return 0;
}

/* This function takes src image and depending upon levels 
selected image will be blurred and quantized to 
have desired number of levels.

returns -1 if levels is not a positive value.
*/
int blurQuantize(cv::Mat& src, cv::Mat& dst, int levels) {
	cv::Mat blurred_image = cv::Mat(src.size(), src.type());
	int status  = blur5x5(src, blurred_image);

	if (status != 0)
	{
		exit(status);
	}
	if (levels <= 0)
	{
		printf("Levels must be positive.");
		return -1;
	}
	int bucket_size = ceil(255.0 / levels);
	// Ensure that we get proper bin size. 
	// Ceil for levels = 10, we need bin size to be 26 instead of 25 to accomodate all values
	for (int row = 0; row < src.rows; row++)
	{
		cv::Vec3b* srcRowPtr = blurred_image.ptr<cv::Vec3b>(row);
		cv::Vec3b* dstRowPtr = dst.ptr<cv::Vec3b>(row);
		for (int col = 0; col < src.cols; col++)
		{
			dstRowPtr[col][0] = floor(srcRowPtr[col][0] / bucket_size) * bucket_size;
			dstRowPtr[col][1] = floor(srcRowPtr[col][1] / bucket_size) * bucket_size;
			dstRowPtr[col][2] = floor(srcRowPtr[col][2] / bucket_size) * bucket_size;
		}
	}
	return 0;
}

int cartoon(cv::Mat& src, cv::Mat& dst, int levels, int magThreshold) {
	
	// Calculating the SobelX of the image
	cv::Mat sx_image = cv::Mat::zeros(src.size(), CV_16SC3);
	sobelX3x3(src, sx_image);	
	
	// Calculating the SobelY of the image
	cv::Mat sy_image = cv::Mat(src.size(), CV_16SC3);
	sobelY3x3(src, sy_image);
	
	/* Calculate the Gradient of the image using the SobelX, SobelY data. */
	cv::Mat gradient_magnitude_image = cv::Mat(src.size(), CV_8UC3);
	magnitude(sx_image, sy_image, gradient_magnitude_image);

	/*Blur and Quantizing the source image */
	cv::Mat blur_quantized_image = cv::Mat(src.size(), CV_8UC3);
	blurQuantize(src, blur_quantized_image, levels);

	for (int row = 0; row < src.rows; row++)
	{
		cv::Vec3b* srcRowPtr = gradient_magnitude_image.ptr<cv::Vec3b>(row);
		cv::Vec3b* dstRowPtr = blur_quantized_image.ptr<cv::Vec3b>(row);
		for (int col = 0; col < src.cols; col++)
		{
			for (int color = 0; color < 3; color++)
			{
				if (srcRowPtr[col][color] > magThreshold)
				{
					dstRowPtr[col][color] = 0;
				}
			}
			/*if (srcRowPtr[col][0] > magThreshold 
				or srcRowPtr[col][1]> magThreshold 
				or srcRowPtr[col][2] > magThreshold)
				{
					dstRowPtr[col][0] = 0;
					dstRowPtr[col][1] = 0;
					dstRowPtr[col][2] = 0;
				}*/
		}
	}

	blur_quantized_image.copyTo(dst);
	return 0;
}


int negativeImage(cv::Mat& src, cv::Mat& dst) {
	for (int row = 0; row < src.rows; row++)
	{
		cv::Vec3b* srcRowPtr = src.ptr<cv::Vec3b>(row);
		cv::Vec3b* dstRowPtr = dst.ptr<cv::Vec3b>(row);
		for (int col = 0; col < src.cols; col++)
		{
			dstRowPtr[col][0] = 255 - srcRowPtr[col][0];
			dstRowPtr[col][1] = 255 - srcRowPtr[col][1];
			dstRowPtr[col][2] = 255 - srcRowPtr[col][2];
		}
	}
	return 0;
}


int strongBlur(cv::Mat& src, cv::Mat& dst) {
	cv::Mat intermediate = cv::Mat(src.size(), src.type());

	// Applying Convolution by [1,4,15,4, 1] (1 x 5 filter)
	for (int row = 0; row < src.rows; row++)
	{
		// Row pointer for quicker access to rows.
		cv::Vec3b* srcRowPtr = src.ptr<cv::Vec3b>(row);
		cv::Vec3b* interRowPtr = intermediate.ptr<cv::Vec3b>(row);
		for (int col = 2; col < src.cols - 2; col++)
		{
			// Update the interim matrix with appropriate 
			interRowPtr[col][0] = (srcRowPtr[col - 2][0] + (4 * srcRowPtr[col - 1][0]) + (15 * srcRowPtr[col][0])
				+ (4 * srcRowPtr[col + 1][0]) + srcRowPtr[col + 2][0]) / 25;
			interRowPtr[col][1] = (srcRowPtr[col - 2][1] + (4 * srcRowPtr[col - 1][1]) + (15 * srcRowPtr[col][1])
				+ (4 * srcRowPtr[col + 1][1]) + srcRowPtr[col + 2][1]) / 25;
			interRowPtr[col][2] = (srcRowPtr[col - 2][2] + (4 * srcRowPtr[col - 1][2]) + (15 * srcRowPtr[col][2])
				+ (4 * srcRowPtr[col + 1][2]) + srcRowPtr[col + 2][2]) / 25;
		}
	}

	/* Applying Convolution by (5 x 1 filter)
		|  1 |
		|  4 |
		| 15 |
		|  4 |
		|  1 |
	*/
	for (int row = 2; row < src.rows - 2; row++)
	{
		cv::Vec3b* rowPtr = intermediate.ptr<cv::Vec3b>(row);
		cv::Vec3b* rowM2Ptr = intermediate.ptr<cv::Vec3b>(row - 2);
		cv::Vec3b* rowM1Ptr = intermediate.ptr<cv::Vec3b>(row - 1);
		cv::Vec3b* rowP1Ptr = intermediate.ptr<cv::Vec3b>(row + 1);
		cv::Vec3b* rowP2Ptr = intermediate.ptr<cv::Vec3b>(row + 2);
		cv::Vec3b* dstRowPtr = dst.ptr<cv::Vec3b>(row);
		for (int col = 0; col < src.cols; col++)
		{
			dstRowPtr[col][0] = (rowM2Ptr[col][0] + (4 * rowM1Ptr[col][0]) + (15 * rowPtr[col][0]) + (4 * rowP1Ptr[col][0]) + rowP2Ptr[col][0]) / 25;
			dstRowPtr[col][1] = (rowM2Ptr[col][1] + (4 * rowM1Ptr[col][1]) + (15 * rowPtr[col][1]) + (4 * rowP1Ptr[col][1]) + rowP2Ptr[col][1]) / 25;
			dstRowPtr[col][2] = (rowM2Ptr[col][2] + (4 * rowM1Ptr[col][2]) + (15 * rowPtr[col][2]) + (4 * rowP1Ptr[col][2]) + rowP2Ptr[col][2]) / 25;
		}
	}
	return 0;
}