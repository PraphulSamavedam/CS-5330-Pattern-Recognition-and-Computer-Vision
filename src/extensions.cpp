#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

int laplacian_filter(cv::Mat& src, cv::Mat& dst) {
	/*This function convolves with laplacian filter
	*					 | 0 -1  0 |
	* Laplacian filter = |-1  4 -1 |
	*					 | 0 -1  0 |
	*/

	for (int row = 1; row < src.rows - 1; row++)
	{
		cv::Vec3b* srcRowP1Ptr = src.ptr<cv::Vec3b>(row + 1);
		cv::Vec3b* srcRowPtr = src.ptr<cv::Vec3b>(row);
		cv::Vec3b* srcRowM1Ptr = src.ptr<cv::Vec3b>(row - 1);
		cv::Vec3b* dstRowPtr = dst.ptr<cv::Vec3b>(row);
		for (int col = 1; col < src.cols - 1; col++)
		{
			dstRowPtr[col][0] = ((4 * srcRowPtr[col][0]) -srcRowM1Ptr[col][0] - srcRowPtr[col - 1][0] - srcRowPtr[col + 1][0] - srcRowP1Ptr[col][0]) / 6;
			dstRowPtr[col][1] = ((4 * srcRowPtr[col][1]) -srcRowM1Ptr[col][1] - srcRowPtr[col - 1][1] - srcRowPtr[col + 1][1] - srcRowP1Ptr[col][1]) / 6;
			dstRowPtr[col][2] = ((4 * srcRowPtr[col][2]) -srcRowM1Ptr[col][2] - srcRowPtr[col - 1][2] - srcRowPtr[col + 1][2] - srcRowP1Ptr[col][2]) / 6;
		}
	}

	return 0;
}

//int laplacian_filter2(cv::Mat& src, cv::Mat& dst) {
//	/*This function convolves with laplacian filter
//	*					 |  0 -1  0 |
//	* Laplacian filter = | -1  4 -1 |
//	*					 |  0 -1  0 |
//	*/
//
//	for (int row = 1; row < src.rows - 1; row++)
//	{
//		cv::Vec3b* srcRowP1Ptr = src.ptr<cv::Vec3b>(row + 1);
//		cv::Vec3b* srcRowPtr = src.ptr<cv::Vec3b>(row);
//		cv::Vec3b* srcRowM1Ptr = src.ptr<cv::Vec3b>(row - 1);
//		cv::Vec3b* dstRowPtr = dst.ptr<cv::Vec3b>(row);
//		for (int col = 1; col < src.cols - 1; col++)
//		{
//			dstRowPtr[col][0] = -(srcRowM1Ptr[col][0] + srcRowPtr[col - 1][0] - (4 * srcRowPtr[col][0]) + srcRowPtr[col + 1][0] + srcRowP1Ptr[col][0]) / 6;
//			dstRowPtr[col][1] = -(srcRowM1Ptr[col][1] + srcRowPtr[col - 1][1] - (4 * srcRowPtr[col][1]) + srcRowPtr[col + 1][1] + srcRowP1Ptr[col][1]) / 6;
//			dstRowPtr[col][2] = -(srcRowM1Ptr[col][2] + srcRowPtr[col - 1][2] - (4 * srcRowPtr[col][2]) + srcRowPtr[col + 1][2] + srcRowP1Ptr[col][2]) / 6;
//		}
//	}
//	return 0;
//}

int medianFilter3x3(cv::Mat& src, cv::Mat& dst) {
	/* This function removes salt pepper noise in the source image.*/
	for (int row = 1; row < src.rows-1; row++)
	{
		cv::Vec3b* srcRowM1Ptr = src.ptr<cv::Vec3b>(row - 1);
		cv::Vec3b* srcRowPtr = src.ptr<cv::Vec3b>(row);
		cv::Vec3b* srcRowP1Ptr = src.ptr<cv::Vec3b>(row + 1);
		cv::Vec3b* dstRowPtr = dst.ptr<cv::Vec3b>(row);
		for (int col = 1; col < src.cols - 1; col++)
		{
			for (int color = 0; color < 3; color++)
			{
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

int boxBlur3x3Brute(cv::Mat& src, cv::Mat& dst) {
	/*This function blurs the image appplying the box filter on the image.
			 | 1 1 1| 
	filter = | 1 1 1|
			 | 1 1 1|
	*/
	for (int row = 1; row < src.rows-1; row++)
	{
		cv::Vec3b* srcRowPtr = src.ptr<cv::Vec3b>(row);
		cv::Vec3b* srcRowM1Ptr = src.ptr<cv::Vec3b>(row-1);
		cv::Vec3b* srcRowP1Ptr = src.ptr<cv::Vec3b>(row+1);
		cv::Vec3b* dstRowPtr = dst.ptr<cv::Vec3b>(row);
		for (int col = 1; col < src.cols-1; col++)
		{
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

int boxBlur5x5Brute(cv::Mat& src, cv::Mat& dst) {
	/*This function blurs the image appplying the box filter on the image.
	* 			| 1 1 1 1 1|
	* 			| 1 1 1 1 1|
	* filter =  | 1 1 1 1 1|
	*			| 1 1 1 1 1|
	*			| 1 1 1 1 1|
	*/
	//Loop over all the rows
	for (int row = 2; row < src.rows - 2; row++)
	{
		cv::Vec3b* srcRowPtr = src.ptr<cv::Vec3b>(row);
		cv::Vec3b* srcRowM1Ptr = src.ptr<cv::Vec3b>(row - 1);
		cv::Vec3b* srcRowM2Ptr = src.ptr<cv::Vec3b>(row - 2);
		cv::Vec3b* srcRowP1Ptr = src.ptr<cv::Vec3b>(row + 1);
		cv::Vec3b* srcRowP2Ptr = src.ptr<cv::Vec3b>(row + 2);
		cv::Vec3b* dstRowPtr = dst.ptr<cv::Vec3b>(row);
		// Loop over all the cols
		for (int col = 2; col < src.cols - 2; col++)
		{
			// Loop over all colors
			for (int color = 0; color < 3; color++)
			{
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

int boxBlur5x5(cv::Mat& src, cv::Mat& dst) {
	/* This function applies box blur  of 5x5 dimenstion to the image using 2 separable filters approach*/
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

	for (int row = 2; row < src.rows-2; row++)
	{
		// Source is interim as it has updated values
		cv::Vec3b* srcRowM2Ptr = interim.ptr<cv::Vec3b>(row-2);
		cv::Vec3b* srcRowM1Ptr = interim.ptr<cv::Vec3b>(row-1);
		cv::Vec3b* srcRowPtr = interim.ptr<cv::Vec3b>(row);
		cv::Vec3b* srcRowP1Ptr = interim.ptr<cv::Vec3b>(row+1);
		cv::Vec3b* srcRowP2Ptr = interim.ptr<cv::Vec3b>(row+2);
		cv::Vec3b* dstRowPtr = dst.ptr<cv::Vec3b>(row);
		for (int col = 0; col < src.cols; col++)
		{
			dstRowPtr[col][0] = (srcRowM2Ptr[col][0] + srcRowM1Ptr[col][0] + srcRowPtr[col][0] + srcRowP1Ptr[col][0] + srcRowP2Ptr[col][0]) / 5;
			dstRowPtr[col] [1] = (srcRowM2Ptr[col] [1] + srcRowM1Ptr[col] [1] + srcRowPtr[col] [1] + srcRowP1Ptr[col] [1] + srcRowP2Ptr[col] [1] ) / 5;
			dstRowPtr[col] [2] = (srcRowM2Ptr[col] [2] + srcRowM1Ptr[col] [2] + srcRowPtr[col] [2] + srcRowP1Ptr[col] [2] + srcRowP2Ptr[col] [2] ) / 5;
		}
	}
	return 0;
}

int boxBlur3x3(cv::Mat& src, cv::Mat& dst) {
	/* This function applies box blur  of 5x5 dimenstion to the image using 2 separable filters approach*/
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

int ensembleEdgeDetector(cv::Mat& median, cv::Mat& dst) {
	
	return 0;
}