/**
* Written by: Samavedam Manikhanta Praphul
* This file provides the signatures of several functions required in the project.
*/

#include <opencv2/opencv.hpp>
#include "../include/tasks.h"
#include <stack> // Required for the stack operations in segmentation
#include <tuple> // Required for tupling the row and column in the stack
#include <map> // Required to map the color with the segmented region
#include <cstdlib> // Required for random number generation.

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
int grassFireAlgorithm(cv::Mat& srcImg, cv::Mat& dstimg, int connectValue, int foreGround, int backGround) {

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
int erosion(cv::Mat& srcImg, cv::Mat& erodedImg, int numberOfTimes, int connectValue, int foreGround, int backGround) {
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
int dilation(cv::Mat& srcImg, cv::Mat& dilatedImg, int numberOfTimes, int connectValue, int foreGround, int backGround) {
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

/** This function find the conencted foreground regions in a binary image using stack.
* Assumes the foreground to be white (255), background color as 255 - foreGround.
* @param srcImg address of the source binary image
* @param dstImg address of the destination binary image having the region labels
* @param connectValue[default=4] set value as 4 or 8 to mark 4-connected, 8-connected technique
* @param foreGround[default=255] value of the foreground pixel value.
* @param debug[default=false] set this for debug print.
* @returns 0 if the segmentation is successful.
* @note AssertionError if connectValue not in (4,8)
*		AssertionError if foreGround or backGround values are not in exactly 0 or 255.
*/
int regionGrowing(cv::Mat& srcImg, cv::Mat& dstImg, int connectValue, int foreGround, bool debug)
{
	// It can either be 4-connected or 8-connected approach.
	assert(connectValue == 4 or connectValue == 8);

	// Foreground color can only be 255 or 0.
	assert(foreGround == 255 or foreGround == 0);

	// Destination needs to binary image
	assert(dstImg.depth() == 1);

	// Destination and source images need to be of same size.
	assert(srcImg.size() == dstImg.size());

	int backGround = 255 - foreGround;
	int counter = 1;

	std::stack<std::tuple<int, int>> pixelStack;

	// Iterate over the pixels for the connected regions. 
	for (int row = 0; row < srcImg.rows; row++)
	{
		uchar* srcPtr = srcImg.ptr<uchar>(row);
		short* dstPtr = dstImg.ptr<short>(row);
		if (debug) { printf("Processing row: %d\n", row); }
		for (int col = 0; col < srcImg.cols; col++)
		{
			// Check if it is foreground pixel and is unlabelled
			if (srcPtr[col] == foreGround and dstPtr[col] == 0)
			{
				if (debug) { printf("Pixel (row,col):(%d,%d) is foreground\n", row, col); }
				dstPtr[col] = counter;
				pixelStack.push(std::make_tuple(row, col));

				while (!pixelStack.empty())
				{
					std::tuple<int, int> t = pixelStack.top();
					pixelStack.pop();
					int r = std::get<0>(t);
					int c = std::get<1>(t);

					// Neighbour pixel is foreground and unlabelled.
					if (r != 0) {
						// Not first row, so previous row exists
						if (srcImg.at<uchar>(r - 1, c) == foreGround and dstImg.at<short>(r - 1, c) == 0)
						{
							dstImg.at<short>(r - 1, c) = counter;
							pixelStack.push(std::make_tuple(r - 1, c));
						}
					}
					if (r != srcImg.rows - 1) {
						// Not the last row so next row exists
						if (srcImg.at<uchar>(r + 1, c) == foreGround and dstImg.at<short>(r + 1, c) == 0)
						{
							dstImg.at<short>(r + 1, c) = counter;
							pixelStack.push(std::make_tuple(r + 1, c));
						}
					}
					if (c != 0) { // Not first col, so previous col exists
						if (srcImg.at<uchar>(r, c - 1) == foreGround and dstImg.at<short>(r, c - 1) == 0)
						{
							dstImg.at<short>(r, c - 1) = counter;
							pixelStack.push(std::make_tuple(r, c - 1));
						}
					}
					if (c != srcImg.cols - 1)
					{ // Not the last col so next col exists
						if (srcImg.at<uchar>(r, c + 1) == foreGround and dstImg.at<short>(r, c + 1) == 0)
						{
							dstImg.at<short>(r, c + 1) = counter;
							pixelStack.push(std::make_tuple(r, c + 1));
						}
					}

					// Additional diagonal neighbours in case of 8- connected
					if (connectValue == 8) {
						if (r != 0)
						{ // Not the first row, so previous row is accessible
							if (c != 0) { // Not first coloumn, so previous col is accessible
								if (srcImg.at<uchar>(r - 1, c - 1) == foreGround and dstImg.at<short>(r - 1, c - 1) == 0)
								{
									dstImg.at<short>(r - 1, c - 1) = counter;
									pixelStack.push(std::make_tuple(r - 1, c - 1));
								}
							}
							if (c != srcImg.cols - 1)
							{
								if (srcImg.at<uchar>(r - 1, c + 1) == foreGround and dstImg.at<short>(r - 1, c + 1) == 0)
								{
									dstImg.at<short>(r - 1, c + 1) = counter;
									pixelStack.push(std::make_tuple(r - 1, c + 1));
								}
							}
						}
						if (r != srcImg.rows - 1)
						{ // Not the last row, so next row is accessible
							if (c != 0) { // Not first coloumn, so previous col is accessible
								if (srcImg.at<uchar>(r + 1, c - 1) == foreGround and dstImg.at<short>(r + 1, c - 1) == 0)
								{
									dstImg.at<short>(r + 1, c - 1) = counter;
									pixelStack.push(std::make_tuple(r + 1, c - 1));
								}
							}
							if (c != srcImg.cols - 1) { // No the last column, so next column is accessible
								if (srcImg.at<uchar>(r + 1, c + 1) == foreGround and dstImg.at<short>(r + 1, c + 1) == 0)
								{
									dstImg.at<short>(r + 1, c + 1) = counter;
									pixelStack.push(std::make_tuple(r + 1, c + 1));
								}
							}
						}
					}
				}

				if (debug) { printf("Region grown for %d\n", counter - 1); }
				counter += 1;
			}
			else {
				if (debug) { printf("Pixel (row,col):(%d,%d) is background\n", row, col); }
			}
		}
	}
	if (debug) { printf("Computed regions are %d\n", counter); }

	return 0;
}

class Compare {
public:
	bool operator()(std::tuple<int, int> first, std::tuple<int, int> second)
	{
		if (std::get<1>(first) < std::get<1>(second)) {
			return true;
		}
		else {
			return false;
		}


	}
};

int topNSegments(cv::Mat& regionMap, cv::Mat& dstImg, int NumberOfRegions, bool debug)
{
	// Binary image is required
	assert(regionMap.depth() == 1);

	assert(regionMap.size() == dstImg.size());

	std::priority_queue<std::tuple<int, int>, std::vector<std::tuple<int, int>>, Compare> pQueue;

	std::map<int, int> regionAreaMap;
	
	// Loop through the image for the number of pixels with that region ID.
	for (int row = 0; row < regionMap.rows; row++)
	{
		short* srcPtr = regionMap.ptr<short>(row);
		for (int col = 0; col < regionMap.cols; col++)
		{
			if (srcPtr[col] != 0) {
				if (regionAreaMap.find(int(srcPtr[col])) == regionAreaMap.end())
				{
					regionAreaMap[int(srcPtr[col])] = 1;
				}
				else {
					regionAreaMap[int(srcPtr[col])] += 1;
				}
			}
		}
	}

	
	int count = 1;
	for (std::map<int, int>::iterator it = regionAreaMap.begin(); it != regionAreaMap.end(); it++)
	{
		pQueue.push(std::make_tuple(it->first, it->second));
		count += 1;
		if (debug) { std::cout << "Region: " << it->first << "; Area: " << it->second << std::endl; }
	}

	int maxRegionsPossible = MIN(count, NumberOfRegions);
	// Map of the region ID with area and the new regionID
	std::map<int, std::tuple<uchar, int>> regionIDBinValueMap;
	int counter = 1;
	while (counter <= maxRegionsPossible) {
		regionIDBinValueMap[std::get<0>(pQueue.top())] = std::make_tuple(255, counter);
		if (debug){ std::cout << "Top Region: " << std::get<0>(pQueue.top()) << " Area: " << std::get<1>(pQueue.top()) << std::endl; }
		pQueue.pop();
		counter += 1;
	}
	
	for (int row = 0; row < regionMap.rows; row++)
	{
		short* srcPtr = regionMap.ptr<short>(row);
		uchar* dstPtr = dstImg.ptr<uchar>(row);
		for (int col = 0; col < regionMap.cols; col++)
		{
			dstPtr[col] = std::get<0>(regionIDBinValueMap[srcPtr[col]]);
			//printf("Pre - Region Image value: %d, ", srcPtr[col]);
			//printf("Destination Value: %d ", dstPtr[col]);
			srcPtr[col] = std::get<1>(regionIDBinValueMap[srcPtr[col]]);
			//printf("Post Region Image value: %d, \n", srcPtr[col]);
			
		}
	}

	if (debug) { printf("Segmented the image into %d regions", maxRegionsPossible); }
	return maxRegionsPossible;
}

/** This function colors the image based on the region Map provided. All the regions with same ID is colored with same random color.
* @param regionMap address of the regionMap image
* @paaram dstImage address of the destination image
* @note: AssertionError if the regionMap and dstImage have different 2D dimensions.
*		 AssertionError if the regionMap doesn't have depth of 1 color.
*		 AssertionError if the dstImage doesn't have depth of 3 colors/channels.
*/
int colorSegmentation(cv::Mat& regionMap, cv::Mat& dstImage) {
	srand(100); // Setting the seed as 100 for replicability of the code.

	assert(regionMap.size() == dstImage.size());

	assert(dstImage.depth() == 3);

	dstImage = cv::Mat::zeros(regionMap.size(), CV_8UC3);

	// Map for coloring the regions with same ID a single color.
	std::map<int, cv::Vec3s> regionColorMap;
	for (int row = 0; row < regionMap.rows; row++) {
		short* srcPtr = regionMap.ptr<short>(row);
		cv::Vec3b* dstPtr = dstImage.ptr<cv::Vec3b>(row);
		for (int col = 0; col < regionMap.cols; col++)
		{
			if (srcPtr[col] != 0) {
				if (regionColorMap.find(int(srcPtr[col])) == regionColorMap.end()) {
					int red = rand() % 255;
					int green = rand() % 255;
					int blue = rand() % 255;
					dstPtr[col] = cv::Vec3b(blue, green, red);
					regionColorMap[int(srcPtr[col])] = cv::Vec3b(blue, green, red);
				}
				else {
					dstPtr[col] = regionColorMap[int(srcPtr[col])];
				}
			}
		}
	}
	return 0;
}

int getFeatures(cv::Mat& regionMap, int regionID, std::vector<double>& featureVector)
{
	// Tmp Mat to store the binary image
	cv::Mat tmp = cv::Mat::zeros(regionMap.size(), CV_8UC1);

	// Only the specific region ID is marked as foreground, else everything is background
	for (int row = 0; row < regionMap.rows; row++)
	{
		uchar* srcPtr = regionMap.ptr<uchar>(row);
		uchar* dstPtr = tmp.ptr<uchar>(row);
		for (int col = 0; col < regionMap.cols; col++)
		{
			if (srcPtr[col] == regionID) {
				dstPtr[col] == 255;
			}
		}
	}
	//cv::imshow("Region specific Image",tmp);
	//cv::imshow("Segmented Image", tmp);
	cv::Moments Moments = cv::moments(tmp, true);
	
	double pixels = Moments.m00;

	// Obtain axis of least momentum
	long double mu_11 = Moments.mu11 * Moments.m00; // mu_11 = sigma(x - x_bar)(y - y_bar)/m00
	long double mu_20 = Moments.mu20 * Moments.m00; // mu_20 = sigma(x - x_bar)^2/m00
	long double mu_02 = Moments.mu02 * Moments.m00; // mu_02 = sigma(y - y_bar)^2/m00

	double alpha = 0.5 * atan((2*mu_11)/(mu_20 - mu_02)); // Alpha = tan-1(2*mu11/mu20-mu02)

	// std::cout << points << std::endl;

	return 0;
}


