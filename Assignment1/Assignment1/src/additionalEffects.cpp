/* This file has all implementations of the additional effects implemented out of interest.
Written by: Samavedam Manikhanta Praphul.
*/
#include <opencv2/opencv.hpp>

int redOnlyImage(cv::Mat& src, cv::Mat& dst) {
/* This function removes the values of blue and green channels,
retaining the red channel values.
@param src -- address of input image(assumed to be in UChar).
@param dst -- address of modified image(assumed to be in UChar).
@return	   0 if transformation is successful
		-100 if source and destination sizes do not match.
@note Scaling is _not_ done.
*/
	if (src.size() != dst.size())
	{
		return -100; // As sizes do not match, cannot proceed. 
	}
	for (int row = 0; row < src.rows; row++)
	{
		cv::Vec3b* srcRowPtr = src.ptr<cv::Vec3b>(row);
		cv::Vec3b* dstRowPtr = dst.ptr<cv::Vec3b>(row);
		for (int col = 0; col < src.cols; col++)
		{
			// Color space uses BGR as channels
			dstRowPtr[col][0] = 0;
			dstRowPtr[col][1] = 0;
			dstRowPtr[col][2] = srcRowPtr[col][2];
		}
	}
	return 0;
}

int blueOnlyImage(cv::Mat& src, cv::Mat& dst) {
	/* This function removes the values of red and green channels,
	retaining the blue channel values.
	@param src -- address of input image(assumed to be in UChar).
	@param dst -- address of modified image(assumed to be in UChar).
	@return	   0 if transformation is successful
			-100 if source and destination sizes do not match.
	@note Scaling is _not_ done.
	*/
	if (src.size() != dst.size())
	{
		return -100; // As sizes do not match, cannot proceed. 
	}
	for (int row = 0; row < src.rows; row++)
	{
		cv::Vec3b* srcRowPtr = src.ptr<cv::Vec3b>(row);
		cv::Vec3b* dstRowPtr = dst.ptr<cv::Vec3b>(row);
		for (int col = 0; col < src.cols; col++)
		{
			// Color space uses BGR as channels
			dstRowPtr[col][0] = srcRowPtr[col][0];
			dstRowPtr[col][1] = 0;
			dstRowPtr[col][2] = 0;
		}
	}
	return 0;
}

int greenOnlyImage(cv::Mat& src, cv::Mat& dst) {
	/* This function removes the values of red and blue channels,
	retaining the green channel values.
	@param src -- address of input image(assumed to be in UChar).
	@param dst -- address of modified image(assumed to be in UChar).
	@return	   0 if transformation is successful
			-100 if source and destination sizes do not match.
	@note Scaling is _not_ done.
	*/
	if (src.size() != dst.size())
	{
		return -100; // As sizes do not match, cannot proceed. 
	}
	for (int row = 0; row < src.rows; row++)
	{
		cv::Vec3b* srcRowPtr = src.ptr<cv::Vec3b>(row);
		cv::Vec3b* dstRowPtr = dst.ptr<cv::Vec3b>(row);
		for (int col = 0; col < src.cols; col++)
		{
			// Color space uses BGR as channels
			dstRowPtr[col][0] = 0;
			dstRowPtr[col][1] = srcRowPtr[col][1];
			dstRowPtr[col][2] = 0;
		}
	}
	return 0;
}