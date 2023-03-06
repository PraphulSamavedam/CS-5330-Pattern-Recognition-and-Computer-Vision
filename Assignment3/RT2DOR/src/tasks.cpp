/**
* Written by: Samavedam Manikhanta Praphul
*                   Poorna Chandra Vemula
* This file provides the signatures of several functions required in the project.
*/

#define _CRT_SECURE_NO_WARNINGS // To supress spritnf_s
#include <opencv2/opencv.hpp>
#include "../include/tasks.h"
#include <stack> // Required for the stack operations in segmentation
#include <tuple> // Required for tupling the row and column in the stack
#include <map> // Required to map the color with the segmented region
#include <cstdlib> // Required for random number generation.
#include "../include/csv_util.h"
#include "../include/match_utils.h"

/*
  Disjoint set union datastructure for union find.
*/
class DSU{
    int V;
    int *parent,*rank;
    public:
        DSU(int V){
            this->V = V;
            parent = new int[V];
            rank = new int[V];
            for(int i=0;i<V;i++){
                parent[i] = -1;
                rank[i] = 1;
            }
        }
 
        int find(int i){
            if(parent[i]==-1)
                return i;
            return parent[i] = find(parent[i]);
        }
 
 
        void unite(int x,int y){
            int s1 = find(x);
            int s2 = find(y);
            if(s1!=s2){
                if(rank[s1] > rank[s2]){
                    parent[s2] = find(s1);
                    rank[s1] += rank[s2];
                }
                else{
                    parent[s1] = find(s2);
                    rank[s2] += rank[s1];
                }
            }
        }
};

/* This image is thresholded in the range of hueMin, hueMax; satMin, satMax;
and valMin , valMax; from the source image.
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

	// Uncomment to check the performance with OpenCV functions
	// cv::Mat grayImg;
	// cv::cvtColor(srcImg, grayImg, cv::COLOR_BGR2GRAY);

	// Checking if the image has value above threshold in the greyscale.
	for (int row = 0; row < srcImg.rows; row++)
	{
		cv::Vec3b* srcPtr = srcImg.ptr<cv::Vec3b>(row);
		// uchar* srtPtr = grayImg.ptr<uchar>(row); 
		uchar* dstPtr = thresholdedImg.ptr<uchar>(row);
		for (int col = 0; col < srcImg.cols; col++)
		{
			dstPtr[col] = ((srcPtr[col][0] * 0.206)  // 0.206 weight for blue channel
				+ (srcPtr[col][1] * 0.588) // 0.588 weight for green channel
				+ (srcPtr[col][2] * 0.206) // 0.206 weight for red channel
		> greyScaleThreshold) ? 0 : 255; // white color if below threshold else black
			// Uncomment to check the performance with OpenCV's function performance.
			// dstPtr[col] = srtPtr[col] > greyScaleThreshold ? 0 : 255; // white color if below threshold else black
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
					currRowPtr[col] = MIN(MIN(aboveRowPtr[col], aboveRowPtr[col + 1])
						, MIN(currRowPtr[col - 1], aboveRowPtr[col - 1])
					) + 1;
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
						MIN(MIN(belowRowPtr[col - 1], belowRowPtr[col])
							, MIN(belowRowPtr[col + 1], currRowPtr[col + 1]))
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


/** This function segments an image into different regions
* Assumes the foreground to be white (255), background to be black (0)
* @param srcImg address of the source binary image
* @param segmentedImg address of the destination binary image
* @param connectValue[default=4] set value as 4 or 8 to mark 4-connected, 8-connected technique
* @param foreGround[default=255] value of the foreground pixel value.
* @param backGround[default=0] value of the background pixel value.
* @returns 0 if the segmentation is success
* @note This function internally uses segmentation algorithm
*        AssertionError if connectValue not in (4,8)
*        AssertionError if foreGround or backGround values are not in range [0,255].
*/
int findRegionMap(cv::Mat &srcImg, cv::Mat &regionMap, int connectValue, int foreGround, int backGround){
    // Supports only 4-connected or 8-connected erosion.
    assert(connectValue == 4 || connectValue == 8);

    // Only foreGround, backGround are not in range [0, 255]
    assert(foreGround >= 0 and foreGround <= 255);
    assert(backGround >= 0 and backGround <= 255);
    
    
    regionMap.convertTo(regionMap, CV_32SC1);
    
    //initializing a DSU with 1000 nodes(for now assuming 1000 are the max regions, may need to take as an input)
    DSU s(1000000);
    
    // converting src to CV_8UC1
    srcImg.convertTo(srcImg, CV_8UC1);
    
    
    //Intializing tmp with zeroes(CV_8UC1)
    cv::Mat tmp = cv::Mat::zeros(srcImg.size(), CV_32SC1);


    int label = 0;

    
  
    // Implementing first for 4 connected
    if (connectValue == 4)
    {
        label = 1;
        //for each pixel look up and back
        for(int row = 0;row<srcImg.rows;row++){
            for(int col = 0; col<srcImg.cols;col++){

                if(int(srcImg.at<uchar>(row,col)) == foreGround){
                         
                    
                    int leftPixel;
                    int abovePixel;
                    
                    //check if row or col is zero and fill in zero accordingly
                    abovePixel = (row!=0) ? tmp.at<short>(row-1,col) : 0;
                    leftPixel =  (col!=0) ? tmp.at<short>(row,col-1) : 0;
                    
        

                    if(leftPixel==0 and abovePixel==0){
                        tmp.at<short>(row,col) = label;
                        label += 1;
                    }
                    
                    else if(leftPixel!=0 and abovePixel!=0){
                        tmp.at<short>(row,col) = (leftPixel > abovePixel) ? abovePixel : leftPixel;
                        if(leftPixel != abovePixel){
                            s.unite(leftPixel,abovePixel);
                        }
                        
                    }
                    
                    else{
                        tmp.at<short>(row,col) = (leftPixel < abovePixel) ? abovePixel : leftPixel;
                    }
                }
                
            }
        }
        
        
        // Second pass
        for (int row = 0; row < srcImg.rows; row++)
        {
            uchar* srcPtr = srcImg.ptr<uchar>(row);
            short* currRowPtr = tmp.ptr<short>(row);
            for (int col = 0; col < srcImg.cols; col++)
            {
                if(int(srcPtr[col]) == foreGround){
                    int newValue = s.find(int(currRowPtr[col]));
                    currRowPtr[col] = newValue;
                }
            }
        }
    }
    
    tmp.copyTo(regionMap);
    
    return 0;
}




/*
  Comparator for priority queue ot tuples
*/
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


<<<<<<< HEAD
/** This function provides the binary image wtih top N regions if they are present in the binary image.
* @param address of the regionMap which is segmented image with single channel with details of the region label.
* @param address of the destinationImage
* @param NumberOfRegions[default=5] number of the top regions (area-wise) which need to be present in the destination image.
* @param debug[default=false] set this to have print statements to debug
* @return 0 if we have processed the binary image for the top N regions.
*/
=======
>>>>>>> f7d60e98e4c1e65e573416660bddd586804e5e68
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


	int count = 0;
	for (std::map<int, int>::iterator it = regionAreaMap.begin(); it != regionAreaMap.end(); it++)
	{
			pQueue.push(std::make_tuple(it->first, it->second));
			count += 1;
			if(debug){ 
			std::cout << "Region: " << it->first << "; Area: " << it->second << std::endl;
			}
		
	}

	int maxRegionsPossible = MIN(count, NumberOfRegions);
	// Map of the region ID with area and the new regionID
	std::map<int, std::tuple<uchar, int>> regionIDBinValueMap;
	int counter = 1;
	while (counter <= maxRegionsPossible) {
		regionIDBinValueMap[std::get<0>(pQueue.top())] = std::make_tuple(255, counter);
		if (debug) { std::cout << "Top Region: " << std::get<0>(pQueue.top()) << " Area: " << std::get<1>(pQueue.top()) << std::endl; }
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

int topNSegments(bool minAreaRestriction, cv::Mat& regionMap, cv::Mat& dstImg, int NumberOfRegions , bool debug)
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


	int count = 0;
	for (std::map<int, int>::iterator it = regionAreaMap.begin(); it != regionAreaMap.end(); it++)
	{
		if (it->second >= ((regionMap.rows * regionMap.cols)/100))
		{
			pQueue.push(std::make_tuple(it->first, it->second));
			count += 1;
			std::cout << "Region: " << it->first << "; Area: " << it->second << std::endl;
		}

	}

	int maxRegionsPossible = MIN(count, NumberOfRegions);
	// Map of the region ID with area and the new regionID
	std::map<int, std::tuple<uchar, int>> regionIDBinValueMap;
	int counter = 1;
	while (counter <= maxRegionsPossible) {
		regionIDBinValueMap[std::get<0>(pQueue.top())] = std::make_tuple(255, counter);
		if (debug) { std::cout << "Top Region: " << std::get<0>(pQueue.top()) << " Area: " << std::get<1>(pQueue.top()) << std::endl; }
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


/**This function returns binary output image, moments, dimensions(width, height), region pixel count
* @param regionMap address of the mapped regions
* @param regionID  ID of the region whose binary image, moments, dimensions(width, height), region pixel count needs to be calculated.
* @param Moments to be updated
* @param Dimensions to be updated
* @param regionPixelCount to be updated
* @returns 0 if the feature is properly extracted.
*        non zero if the operation is failure.
*/
int binaryImageWithARegion(cv::Mat& regionMap, cv::Mat& binaryOutputImage, cv::Moments& Moments, std::pair<double, double>& Dimensions, int& regionPixelCount, int regionID) {

	//expects binaryOutputImage is 8UC1
	//expects regionMap is 32SC1

	assert(regionMap.depth() == 1);
	assert(binaryOutputImage.depth() == 1);

	assert(regionMap.size() == binaryOutputImage.size());

	//Initializing minRow, maxRow, minCol, maxCol
	int minRow = INT_MAX;
	int maxRow = -INT_MAX;
	int minCol = INT_MAX;
	int maxCol = -INT_MAX;
	regionPixelCount = 0;

	// Only the specific region ID is marked as foreground, else everything is background
	for (int row = 0; row < regionMap.rows; row++)
	{
		short* srcPtr = regionMap.ptr<short>(row);
		uchar* dstPtr = binaryOutputImage.ptr<uchar>(row);
		for (int col = 0; col < regionMap.cols; col++)
		{
			if (srcPtr[col] == regionID) {
				dstPtr[col] = 255;
				minRow = MIN(row, minRow);
				maxRow = MAX(row, maxRow);
				minCol = MIN(col, minCol);
				maxCol = MAX(col, maxCol);
				regionPixelCount += 1;
			}
		}
	}



	double width = (maxCol - minCol);
	double height = (maxRow - minRow);

	//first element in pair is width and sencond is height
	Dimensions.first = width;
	Dimensions.second = height;

	//compute moments
	Moments = cv::moments(binaryOutputImage, true);

	return 0;
}


/**This function populates the feature vectors in the featureVector for the specific region in the region map.
* @param regionMap address of the mapped regions
* @param regionID  ID of the region whose features needs to be calculated.
* @param featureVector address of the feature vecotr which needs to have the features of the selected region.
* @returns 0 if the feature is properly extracted.
*		non zero if the operation is failure.
*/
int getFeaturesForARegion(cv::Mat& regionMap, int regionID, std::vector<float>& featureVector) {

	// Tmp Mat to store the binary image
	cv::Mat tmp = cv::Mat::zeros(regionMap.size(), CV_8UC1);

	std::pair<double, double> dimensionsOfRegion;
	int regionPixelCount = 0;
	cv::Moments Moments;
	binaryImageWithARegion(regionMap, tmp, Moments, dimensionsOfRegion, regionPixelCount, regionID);


	//compute h/w ratio
	float width = dimensionsOfRegion.first;
	float height = dimensionsOfRegion.second;
	float hw_ratio = height / width;



	//compute percent fill ratio
	float area = height * width;
	float percentFill = regionPixelCount / area;



	//compute HuMoments
	double huMoments[7];
	cv::HuMoments(Moments, huMoments);


	//push all the featuers feature Vector
	featureVector.push_back(hw_ratio);
	featureVector.push_back(percentFill);
	for (double huMoment : huMoments) {
		float value = -1 * copysign(1.0, huMoment) * log10(abs(huMoment));
		//std::cout << "Hu Moment: " << huMoment << " Mod HuMomentValue: " << value << std::endl;
		featureVector.push_back(value);
	}


	return 0;
}


/** This function populates the feature vectors in the featureVector for the all the regions in the region map.
* @param regionMap address of the mapped regions
* @param featureVector address of the feature vecotr which needs to have the features
* @param numberOfRegions number of the regions to be identified in the regionMap.
* @returns 0 if the feature is properly extracted.
*		non zero if the operation is failure.
*/
int getFeatures(cv::Mat& regionMap, std::vector<float>& featureVector, int numberOfRegions)
{

	//for each region send regionMap and regionID, featureVector
	for (int i = 1; i <= numberOfRegions; i++) {
		getFeaturesForARegion(regionMap, i, featureVector);
	}


	return 0;
}



/**This function draws the bounding box for a region
* @param regionMap address of the mapped regions
* @param regionID  ID of the region.
* @param outputImg image to draw bounding box
* @param debug[default=false] set this to have print statements to debug
* @returns 0 if the feature is properly extracted.
*        non zero if the operation is failure.
*/
int drawBoundingBoxForARegion(cv::Mat& regionMap, cv::Mat& outputImg, int regionID, bool debug) {

	// Tmp Mat to store the binary image
	cv::Mat tmp = cv::Mat::zeros(regionMap.size(), CV_8UC1);

	int minRow = INT_MAX;
	int maxRow = -INT_MAX;
	int minCol = INT_MAX;
	int maxCol = -INT_MAX;

	// Only the specific region ID is marked as foreground, else everything is background
	for (int row = 0; row < regionMap.rows; row++)
	{
		short* srcPtr = regionMap.ptr<short>(row);
		uchar* dstPtr = tmp.ptr<uchar>(row);
		for (int col = 0; col < regionMap.cols; col++)
		{
			if (srcPtr[col] == regionID) {
				dstPtr[col] = 255;
				minRow = MIN(row, minRow);
				maxRow = MAX(row, maxRow);
				minCol = MIN(col, minCol);
				maxCol = MAX(col, maxCol);
			}
		}
	}

	cv::Moments Moments = cv::moments(tmp, true);

	// First calculate alpha - angle of least central moment
	// alpha = arctan(2*m11/m20-m02)
	float nu11 = Moments.nu11;
	float nu20 = Moments.nu20;
	float nu02 = Moments.nu02;
	float theta = atan(2 * nu11 / (nu20 - nu02)) / 2.0;

	if (debug) { printf("Angle Theta: %.04f\n", theta); }
	// compute xbar and ybar(center point)
	float m10 = Moments.m10;
	float m01 = Moments.m01;
	float m00 = Moments.m00;
	float xbar = m10 / m00;
	float ybar = m01 / m00;

	cv::Point2f centroid(xbar, ybar);
	if (debug) { printf("Centroid: %.02f, %0.2f\n", xbar, ybar); }

	// Compute the width and height of the bounding box
	float sin_theta = sin(theta);
	float cos_theta = cos(theta);
	float a = Moments.mu20 * cos_theta * cos_theta + 2.0 * Moments.mu11 * sin_theta * cos_theta + Moments.mu02 * sin_theta * sin_theta;
	float b = Moments.mu20 * sin_theta * sin_theta - 2.0 * Moments.mu11 * sin_theta * cos_theta + Moments.mu02 * cos_theta * cos_theta;
	/*float width = sqrt(a);
	float height = sqrt(b);*/
	float width = (maxCol - minCol);
	float height = (maxRow - minRow);
	if (debug) { printf("Height: %.02f, %0.2f\n", width, height); }

	// Create a rotated rectangle with the center point of the contour, width, height, and orientation angle
	cv::RotatedRect box(centroid, cv::Size2f(width, height), theta * 180.0 / CV_PI);

	cv::Point2f vertices[4];
	box.points(vertices);

	if (debug) { std::cout << "Vertices: " << vertices[0] << "," << vertices[1] << "," << vertices[2] << "," << vertices[3] << std::endl; }

	// Plot the bounding boxes
	for (int i = 0; i < 4; i++) {
		cv::line(outputImg, vertices[i], vertices[(i + 1) % 4], cv::Scalar(70, 18, 31), 2);
	}

	// Create major axis
	cv::Point2f majorAxis[2];
	// Mark the major axis points
	for (int i = 0; i < 2; i++)
	{
		int x = xbar + ((1 - (2 * i)) * (width * 0.5 * cos_theta));
		int y = ybar + ((1 - (2 * i)) * (width * 0.5 * sin_theta));
		majorAxis[i] = cv::Point2f(x, y);
	}

	// Plot major axis
	cv::line(outputImg, majorAxis[0], majorAxis[1], cv::Scalar(247, 223, 173), 5);
	// Create minor axis
	cv::Point2f minorAxis[2];
	// Mark the minor axis points
	sin_theta = sin(theta + (CV_PI / 2));
	cos_theta = cos(theta + (CV_PI / 2));
	for (int i = 0; i < 2; i++)
	{
		int x2 = xbar + ((1 - (2 * i)) * (height * 0.25 * cos_theta));
		int y2 = ybar + ((1 - (2 * i)) * (height * 0.25 * sin_theta));
		minorAxis[i] = cv::Point2f(x2, y2);
	}

	// Plot minor axis
	cv::line(outputImg, minorAxis[0], minorAxis[1], cv::Scalar(255, 255, 255), 3);

	// std::cout << "Vertices: " << vertices[0] << "," << vertices[1] << "," << vertices[2] << "," << vertices[3] << std::endl;

	return 0;
}



/**This function draws the bounding boxes for the all the regions in the region map.
* @param regionMap address of the mapped regions
* @param outputImg image to draw bounding box
* @param numberOfRegions number of the regions to draw the bounding box on
* @param debug[default=false] set this to have print statements to debug
* @returns 0 if the feature is properly extracted.
*        non zero if the operation is failure.
*/
int drawBoundingBoxes(cv::Mat& regionMap, cv::Mat& outputImg, int numberOfRegions, bool debug) {


	//for each region send regionMap,outputImg and region id
	for (int i = 1; i <= numberOfRegions; i++) {

		drawBoundingBoxForARegion(regionMap, outputImg, i);

	}


	return 0;
}

/**This function populates the confusion matrix in the confusion matrix file.
* @param featuresAndLabelsFile containing features and their labels
* @param confusionMatrixFile to write the confusion matrix
* @param labelnames contains true class labels
* @param predictedLabelNames contains predicted class labels
* @returns populates the confusion matrix csv file
*        non zero if the operation is failure.
*/
int confusionMatrixCSV(char* featuresAndLabelsFile, char* confusionMatrixFile,
						std::vector<char*> labelnames, std::vector<char*> predictedLabelNames) {

	std::set<std::string> labelsSet(labelnames.begin(), labelnames.end());

	std::unordered_map<std::string, std::unordered_map<std::string, float>> mp;
	std::unordered_map<std::string, float> current_label_map;

	for (std::string label : labelsSet) {
		current_label_map[label] = 0;
	}

	for (std::string label : labelsSet) {
		mp[label] = current_label_map;
	}

	for (int i = 0; i < labelnames.size(); i++) {
		mp[labelnames[i]][predictedLabelNames[i]] += 1;
	}

	//    std::vector<char*> ;
	//
	//    append_image_data_csv();

	for (auto labelRow : mp) {
		std::cout << "label : " << labelRow.first << std::endl;

		for (auto predictLabelCounts : labelRow.second) {
			std::cout << predictLabelCounts.first << predictLabelCounts.second << std::endl;
		}
	}

	// Write the confusion matrix to csv
	std::vector<char*> uniqueLabelNamesList;
	for (std::string label:labelsSet)
	{
		char* cstr = new char[label.length()+1];
		strcpy(cstr, label.c_str());
		uniqueLabelNamesList.push_back(cstr);
	}

	append_label_data_csv(confusionMatrixFile, uniqueLabelNamesList, true);

	for (char* label: uniqueLabelNamesList)
	{
		std::vector<int> confusionVector;
		for (char* predictedLabel: uniqueLabelNamesList)
		{
			confusionVector.push_back(mp[label][predictedLabel]);
		}
		append_confusion_data_csv(confusionMatrixFile, label, confusionVector, false);
	}
	printf("Successfully wrote confusion matrix to %s file.", confusionMatrixFile);
	return 0;
}


/*
   This class implements comparator for the priority queue.
   - priority queue is built using the second element in the pair
*/
class CompareSecondElement {
public:
	bool operator()(std::tuple<char*, char*, float> first, std::tuple<char*, char*, float> second)
	{
		if (std::get<2>(first) > std::get<2>(second)) {
			return true;
		}
		else {
			return false;
		}


	}
};

<<<<<<< HEAD

/**This function generates and populates the predicted labels.
* @param featuresAndLabelsFile containing features and their labels
* @param labelnames contains true class labels
* @param predictedLabelNames which will be updated with predicted class labels
* @returns populates the predictedLabels
*        non zero if the operation is failure.
*/
int generatePredictions(char* featuresAndLabelsFile, std::vector<char* > &predictedLabels, std::vector<char*>& labelnames, int N) {
=======
int generatePredictions(char* featuresAndLabelsFile, std::vector<char* > &predictedLabels, 
	std::vector<char*>& labelnames, char* &distanceMetric ,int N, bool debug) {
>>>>>>> f7d60e98e4c1e65e573416660bddd586804e5e68

	std::vector<std::vector<float>> data;
	std::vector<char*> filenames;
	
	int i = read_image_data_csv(featuresAndLabelsFile, filenames, labelnames, data, 0);

	if (i != 0) {
		std::cout << "file read unsuccessful" << std::endl;
		exit(-1);
	}

	for (int currFileIndex = 0; currFileIndex < filenames.size(); currFileIndex++)
	{
		std::vector<float> targetFeatureVector = data[currFileIndex];
		char labelPredicted[64];
		ComputingNearestLabelUsingKNN(targetFeatureVector,
			featuresAndLabelsFile, distanceMetric, labelPredicted, N);
		if (debug) {
			std::cout << "\nFor file " << filenames[currFileIndex] << " Ground truth:" << labelnames[currFileIndex];
			std::cout << " Predicted label:" <<  labelPredicted << "with K: " << N << std::endl;
		}
		predictedLabels.push_back(labelPredicted);
	}
	return 0;
}


/*
    Implemented Otsu algorithm for thresholding.
*/

int otsuThresholdImage(cv::Mat& srcImg, cv::Mat& thresholdedImg) {

 

    thresholdedImg = cv::Mat::zeros(srcImg.size(), CV_8UC1);
    
    cv::Mat grayImg;
    cv::cvtColor(srcImg, grayImg, cv::COLOR_BGR2GRAY);
    
    std::vector<float> Hist(256,0);
    
    float histCumSum = 0;
    
    for(int i=0;i<grayImg.rows;i++){
        for(int j=0;j<grayImg.cols;j++){
            int pixelValue = grayImg.at<uchar>(i,j);
            Hist[pixelValue] += 1;
            histCumSum += 1;
        }
    }
    
    //normalize
    for(int i=0;i<256;i++){
        Hist[i]/=histCumSum;
    }
    
    
    std::vector<float> bins(256);
    for(int i=0;i<256;i++){
        bins[i] = i;
    }
    
    float fn_min = INT_MAX;
    int thresh = -1;
    
    for(int i=0;i<256;i++){
        //get p1
        std::vector<float> p1;
        for(int j=0;j<i;j++){
            p1.push_back(Hist[j]);
        }
        
        std::vector<float> p2;
        for(int j=i+1;j<256;j++){
            p2.push_back(Hist[j]);
        }
        
        //fet q1,q2 -> csum
        float sum = 0.0;
        for(int j=0;j<i;j++){
            sum+=Hist[j];
        }
        float q1 = sum;
    
        
        sum = 0.0;
        for(int j=i+1;j<256;j++){
            sum+=Hist[j];
        }
        float q2 = sum;
        
        //write an if condition to continue
        if(q1 < 0.000001 or q2 < 0.000001){
            continue;
        }
        
        //get b1
        std::vector<float> b1;
        for(int j=0;j<i;j++){
            b1.push_back(bins[j]);
        }
        
        std::vector<float> b2;
        for(int j=i+1;j<256;j++){
            b2.push_back(bins[j]);
        }

        
        //find m1,m2
        sum = 0.0;
        for(int j=0;j<p1.size();j++){
            sum += p1[j]*b1[j];
        }
        float m1 =  sum/q1;
      
        
        sum = 0.0;
        for(int j=0;j<p2.size();j++){
            sum += p2[j]*b2[j];
        }
        float m2 = sum/q1;
        
        //find v1,v2
        sum = 0.0;
        for(int j=0;j<p1.size();j++){
            sum += ((b1[j]-m1)*(b1[j]-m1))*p1[j];
        }
        
        float v1 =  sum/q1;
     
        
        sum = 0.0;
        for(int j=0;j<p2.size();j++){
            sum += ((b2[j]-m1)*(b2[j]-m1))*p2[j];
        }
        float v2 = sum/q2;
 
        
      //calculates the minimization function
        float fn = v1*q1 + v2*q2;
   
        
        if(fn_min>fn){

            
            fn_min = fn;
            thresh = i;
        }
        
    }
    
    
    if(thresh != -1){
        
        for (int row = 0; row < thresholdedImg.rows; row++)
        {
            uchar* srcPtr = grayImg.ptr<uchar>(row);
            
            uchar* dstPtr = thresholdedImg.ptr<uchar>(row);
            for (int col = 0; col < thresholdedImg.cols; col++)
            {
                dstPtr[col] = (thresh-50 > srcPtr[col]) ? 255 : 0;
            }
        }
        
    }

    return 0;

}

