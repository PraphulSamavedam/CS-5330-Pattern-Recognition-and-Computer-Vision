/** This file has quick utils required for the assignment.
Written By: Samavedam Manikhanta Praphul.
*/

#define _CRT_SECURE_NO_WARNINGS // Required to suppress warnings of localtime, sprintf_s

#include <ctime>
#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp>

int windowSize = cv::WINDOW_AUTOSIZE;

/** This function saves the image with the file name of the snapshot as
* format: IMG_<effect>_yyyy_mm_dd_HH_MM_SS.png
*/
int saveImage(cv::Mat& src, const char* effect) {
	if (!src.empty())
	{
		char fileName[50]; // buffer for the file name of the snapshot.
		std::time_t time_now = std::time(0);   // get time now
		std::tm* tm_now = std::localtime(&time_now); // To convert time into tm struct for easier computation.
		/* Reference : https://en.cppreference.com/w/cpp/chrono/c/tm */
		// Format file name as IMG_yyyy_mm_dd_HH_MM_SS.png
		sprintf(fileName, "output/IMG_%s_%04d_%02d_%02d_%02d_%02d_%02d.png", effect,
			(tm_now->tm_year + 1900), (tm_now->tm_mon + 1), tm_now->tm_mday,
			tm_now->tm_hour, tm_now->tm_min, tm_now->tm_sec);
		// Store the snapshot/frame to a file.
		cv::imwrite(fileName, src);
	}
	return (0);
}

/** This function displays the image with the windowName provided 
* and depending upon param scaleBeforeDisplay , the image is scaled before displaying it. 
* @param displayStatus if true then displayed else if any existing window is closed.
* @param image address of the image to displayed.
* @param windowName name of the window in which the image needs to be displayed. 
* @param scaleBeforeDisplay if true, the image passed by @param image is convert to 8UC3 type 
* with values in range of 0-255 before displaying the image. 
*/
int displayImage(bool displayStatus, cv::Mat& image, const char* windowName, bool scaleBeforeDisplay= false) {
	//Display the frame if it is set to display.
	if (displayStatus)
	{
		cv::namedWindow(windowName, windowSize);
		if (scaleBeforeDisplay) // Scale the image and display if scaling is required.
		{
			cv::Mat scaled_image;
			cv::convertScaleAbs(image, scaled_image);
			cv::imshow(windowName, image);
		}
		else { // Scaling of the image is not required.
			cv::imshow(windowName, image);
		}
	}
	//If it is disabled, then close the window.
	else if (cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) == 1.0)
	{
		cv::destroyWindow(windowName);
	}
	return (0);
}