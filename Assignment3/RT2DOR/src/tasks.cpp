/**
* Written by: Samavedam Manikhanta Praphul
* This file provides the signatures of several functions required in the project.
*/

#include <opencv2/opencv.hpp>

/* This image is thresholded in the range of hueMin, hueMax; satMin, satMax;
and valMin , valMax; from the source image.
@param srcImg address of the source image
@param
*/
int thresholdImage(cv::Mat& srcImg, int hueMin, int hueMax, int satMin, int satMax, int valMin, int valMax, cv::Mat& thresholdedImg) {

	cv::Mat hsvImage;
	cv::cvtColor(srcImg, hsvImage, cv::COLOR_BGR2HSV);
	for (int row = 0; row < srcImg.rows; row++)
	{
		cv::Vec3b* srcPtr = hsvImage.ptr<cv::Vec3b>(row);
		cv::Vec3b* dstPtr = thresholdedImg.ptr<cv::Vec3b>(row);
		for (int col = 0; col < srcImg.cols; col++)
		{
			if ((hueMin <= srcPtr[col][0] && srcPtr[col][0] <= hueMax)
				&& (satMin <= srcPtr[col][1] && srcPtr[col][1] <= satMax)
				&& (valMin <= srcPtr[col][2] && srcPtr[col][2] <= valMax)
				)
			{
				dstPtr[col][0] = 0;
				dstPtr[col][1] = 0;
				dstPtr[col][2] = 0;
			}
			else {
				dstPtr[col][0] = 255;
				dstPtr[col][1] = 255;
				dstPtr[col][2] = 255;
			}
		}
	}
	return 0;
}

/*This function masks the source image by marking the image above threshold as black below as white.
*/
int thresholdImage(cv::Mat& srcImg, cv::Mat& thresholdedImg, int greyScaleThreshold) {

	// Thresholded Image must be binary image.
	thresholdedImg = cv::Mat::zeros(srcImg.size(), CV_8UC1);

	cv::Mat grayImg;
	cv::cvtColor(srcImg, grayImg, cv::COLOR_BGR2GRAY);

	// Checking if the image has value above threshold in the greyscale.
	for (int row = 0; row < srcImg.rows; row++)
	{
		cv::Vec3b* srcPtr = srcImg.ptr<cv::Vec3b>(row);
		uchar* srtPtr = grayImg.ptr<uchar>(row);
		uchar* dstPtr = thresholdedImg.ptr<uchar>(row);
		for (int col = 0; col < srcImg.cols; col++)
		{
			dstPtr[col] = ((srcPtr[col][0] * 0.206)  // 0.206 weight for blue channel
				+ (srcPtr[col][1] * 0.588) // 0.588 weight for green channel
				+ (srcPtr[col][2] * 0.206) // 0.206 weight for red channel
		> greyScaleThreshold) ? 0 : 255; // white color if below threshold else black
			dstPtr[col] = srtPtr[col] > greyScaleThreshold ? 0 : 255; // white color if below threshold else black
		}
	}
	return 0;

}

/** This function returns only the fileName from the filePath provided.
@param filePath path of the file whose name needs to be obtained.
@param fileName placeholder for result.
@return 0 for successfully obtaining the fileName.
@note Assumes that the filePath is valid (doesn't validate filePath)
	  Method: Parses the filePath to find the last folder separator like '/' or '\\' and
	  populates from that index to end.
*/
int getOnlyFileName(char*& filePath, char*& fileName) {
	// Get the last \ index and then populate the fileName

	// Get the last '\' or '/' index in the filePath
	int length = strlen(filePath);
	int index = 0;
	for (int ind = length - 1; ind > -1; ind--)
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

/**  This function does the Grass Fire transformation to obtain the distance of the pixel from the background.
* Assumes the foreground to be white (255), background to be black (0)
* @param srcImg address of the source binary image
* @param dstImg address of the destination image which needs to have distance values
* @param connectValue[default=4] set value as 4 or 8 to mark 4-connected, 8-connected technique
* @param foreGround[default=255] value of the foreground pixel value.
* @param backGround[default=0] value of the background pixel value.
* @returns 0 if the values are computed completed.
* @note AssertionError if connectValue not in (4,8)
*		AssertionError if foreGround or backGround values are not in range [0,255].
*/
int grassFireAlgorithm(cv::Mat& srcImg, cv::Mat& dstimg, int connectValue = 4, int foreGround = 255, int backGround = 0) {

	//Supports only 4-connected or 8-connected approach
	assert(connectValue == 4 || connectValue == 8);

	// Defaultly zero
	cv::Mat tmp = cv::Mat::zeros(srcImg.size(), CV_8UC1);

	//Cover for edges cases of first row and first column
	// First row
	for (int col = 1; col < srcImg.cols; col++)
	{
		tmp.at<uchar>(0, col) = srcImg.at<uchar>(0, col) == foreGround ? 1 : 0;
	}

	// First Column
	for (int row = 0; row < srcImg.rows; row++)
	{
		tmp.at<uchar>(row, 0) = srcImg.at<uchar>(row, 0) == foreGround ? 1 : 0;
	}

	// Defaultly Foreground is white and Background is black
	if (connectValue == 4)
	{
		// First pass
		for (int row = 1; row < srcImg.rows; row++)
		{
			// Additional Rowpointer for previous row.
			uchar* srcPtr = srcImg.ptr<uchar>(row);
			uchar* aboveRowPtr = tmp.ptr<uchar>(row - 1);
			uchar* currRowPtr = tmp.ptr<uchar>(row);
			for (int col = 1; col < srcImg.cols; col++)
			{
				//If the current pixel is foreground then minimum of top and left value.
				if (srcPtr[col] == foreGround)
				{
					// Minimum of neighbouring cells -> cell before or cell above
					currRowPtr[col] = currRowPtr[col - 1] > aboveRowPtr[col] ? aboveRowPtr[col] + 1 : currRowPtr[col - 1] + 1;
				}
				// Else the pixel is background then no change default value is 0 (already handled during initialization
			}
		}


		// Second pass
		for (int row = srcImg.rows - 2; row > -1; row--)
		{
			// Additional pointer for next row
			uchar* srcPtr = srcImg.ptr<uchar>(row);
			uchar* dstNextRowPtr = tmp.ptr<uchar>(row + 1);
			uchar* dstPtr = tmp.ptr<uchar>(row);
			for (int col = srcImg.cols - 2; col > -1; col--)
			{
				//If the current pixel is foreground then minimum of current value or min(bottom,right) + 1.
				if (srcPtr[col] == foreGround)
				{
					// Minimum of neighbouring cells -> cell right or cell below
					uchar minValue = dstNextRowPtr[col] > dstPtr[col + 1] ? dstPtr[col + 1] : dstNextRowPtr[col];
					dstPtr[col] = dstPtr[col] > minValue + 1 ? minValue + 1 : dstPtr[col];
				}
				// Else the pixel is background then no change.
			}
		}
	}

	else // Connect value can only be 8
	{

		// First pass
		for (int row = 1; row < srcImg.rows; row++)
		{
			// Additional Rowpointer for previous row.
			uchar* srcPtr = srcImg.ptr<uchar>(row);
			uchar* aboveRowPtr = tmp.ptr<uchar>(row - 1);
			uchar* currRowPtr = tmp.ptr<uchar>(row);
			for (int col = 1; col < srcImg.cols; col++)
			{
				//If the current pixel is foreground then minimum of top and left value.
				if (srcPtr[col] == foreGround)
				{
					// Minimum of neighbouring cells -> cell before or cell above
					currRowPtr[col] = MIN(aboveRowPtr[col - 1], aboveRowPtr[col], aboveRowPtr[col + 1]
						, currRowPtr[col - 1]) + 1;
				}
				// Else the pixel is background then no change default value is 0 (already handled during initialization
			}
		}


		// Second pass
		for (int row = srcImg.rows - 2; row > -1; row--)
		{
			// Additional rowPointer for next row
			uchar* srcPtr = srcImg.ptr<uchar>(row);
			uchar* belowRowPtr = tmp.ptr<uchar>(row + 1);
			uchar* currRowPtr = tmp.ptr<uchar>(row);
			for (int col = srcImg.cols - 2; col > -1; col--)
			{
				//If the current pixel is foreground then minimum of current value or
																// min(neighbours below and right) + 1.
				if (srcPtr[col] == foreGround)
				{
					// Minimum of current value or neighbouring cells + 1
					currRowPtr[col] = MIN(currRowPtr[col],
						MIN(belowRowPtr[col - 1], belowRowPtr[col], belowRowPtr[col + 1], currRowPtr[col + 1])
						+ 1);
				}
				// Else the pixel is background then no change.
			}
		}

		tmp.copyTo(dstimg);
	}
	return 0;
}

/** This function makes a foreground pixel into a background pixel based on the connect method chosen.
* Assumes the foreground to be white (255), background to be black (0)
* @param srcImg address of the source binary image
* @param erodedimg address of the destination binary image
* @param numberOftimes times the erosion operation needs to be performed.
* @param connectValue[default=4] set value as 4 or 8 to mark 4-connected, 8-connected technique
* @param foreGround[default=255] value of the foreground pixel value.
* @param backGround[default=0] value of the background pixel value.
* @returns 0 if the erosion is success.
* @note This function internally uses grassFireAlgorithm to obtain the number of erosions required to erode the specific pixel.
*		AssertionError if connectValue not in (4,8)
*		AssertionError if foreGround or backGround values are not in range [0,255].
*/
int erosion(cv::Mat& srcImg, cv::Mat& erodedImg, int numberOfTimes, int connectValue = 4, int foreGround = 255, int backGround = 0) {
	// Supports only 4-connected or 8-connected erosion. 
	assert(connectValue == 4 || connectValue == 8);

	// Only foreGround, backGround are not in range [0, 255]
	assert(foreGround >= 0 and foreGround <= 255);
	assert(backGround >= 0 and backGround <= 255);

	// Copy the image from the source to erodedImage
	srcImg.copyTo(erodedImg);

	// Run the grassFire Algorithm to get the number of erosions required to erode into background
	cv::Mat gFireImg;
	grassFireAlgorithm(srcImg, gFireImg, connectValue, foreGround);

	// Iterate overall the pixels to mark the pixels as BackGround
	for (int row = 0; row < srcImg.rows; row++)
	{
		uchar* srcPtr = gFireImg.ptr<uchar>(row);
		uchar* dstPtr = erodedImg.ptr<uchar>(row);
		for (int col = 0; col < gFireImg.cols; col++)
		{
			// If we erode more number of times than required
			dstPtr[col] = srcPtr[col] <= numberOfTimes ? backGround : foreGround;
		}
	}

	return 0;

}

/** This function makes a background pixel into a foreground pixel based on the connect method chosen.
* Assumes the foreground to be white (255), background to be black (0)
* @param srcImg address of the source binary image
* @param erodedimg address of the destination binary image
* @param numberOftimes times the dilation operation needs to be performed.
* @param connectValue[default=4] set value as 4 or 8 to mark 4-connected, 8-connected technique
* @param foreGround[default=255] value of the foreground pixel value.
* @param backGround[default=0] value of the background pixel value.
* @returns 0 if the dilusions is success.
* @note This function internally uses grassFireAlgorithm
*		AssertionError if connectValue not in (4,8)
*		AssertionError if foreGround or backGround values are not in range [0,255].
*/
int dilation(cv::Mat& srcImg, cv::Mat& dilatedImg, int numberOfTimes, int connectValue = 4, int foreGround = 255, int backGround = 0) {
	// Supports only 4-connected or 8-connected erosion. 
	assert(connectValue == 4 || connectValue == 8);

	// Only foreGround, backGround are not in range [0, 255]
	assert(foreGround >= 0 and foreGround <= 255);
	assert(backGround >= 0 and backGround <= 255);

	// Copy the image from the source to erodedImage
	srcImg.copyTo(dilatedImg);

	// Run the grassFire Algorithm to get the number of dilations required to dilate foreGround
	cv::Mat gFireImg;
	grassFireAlgorithm(srcImg, gFireImg, connectValue, 255 - foreGround);

	// Iterate overall the pixels to mark the pixels as foreGround
	for (int row = 0; row < srcImg.rows; row++)
	{
		uchar* srcPtr = gFireImg.ptr<uchar>(row);
		uchar* dstPtr = dilatedImg.ptr<uchar>(row);
		for (int col = 0; col < gFireImg.cols; col++)
		{
			// 
			dstPtr[col] = srcPtr[col] < numberOfTimes ? foreGround : backGround;
		}
	}

	return 0;

}