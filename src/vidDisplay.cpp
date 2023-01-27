/* This program opens a video channel, 
create a window, and then loop, capturing 
a new frame and displaying it each time 
through the loop to make it appear as stream. 

returns -404 is video device is not found. 

if 'q' is pressed: program terminates.
if 's' is pressed: program saves the frame as an image 
					to a file with timestamp in the format IMG_yyyy_mm_dd_HH_MM_SS.png
*/

#define _CRT_SECURE_NO_WARNINGS // Required to suppress warnings of localtime, sprintf_s

#include <opencv2/opencv.hpp> // All the required openCV functions and definitions.
#include <cstdio>	// Primary C functions, definitions which interact with stdin, stdout.
#include <cstring> // All the required string functions, definitions.
#include <ctime> // Required for functions accessing the timestamp details for the file name.
#include "../include/imageFilters.h"

int greyscale_of_image(cv::Mat &src, cv::Mat &dst) {
	for (int row = 0; row < src.rows; row++) {
		cv::Vec3b *srcRPtr = src.ptr<cv::Vec3b>(row);
		cv::Vec3f *dstRPtr = dst.ptr<cv::Vec3f>(row);
		for (int col = 0; col < src.cols; col++)
		{
			float agg_val = (srcRPtr[col][1] + srcRPtr[col][0] + srcRPtr[col][2])/3;
			dstRPtr[col][0] = agg_val;
			dstRPtr[col][1] = agg_val;
			dstRPtr[col][2] = agg_val;
		}
	}
	return 0;
}

int main(int argc, char* argv[]) {
	
	cv::VideoCapture* capture = new cv::VideoCapture(0);
	// Check if any video capture device is present.
	if (!capture->isOpened())
	{
		printf("Unable to open the primary video device.\n");
		return(-404);
	}
	cv::Size refs((int)capture->get(cv::CAP_PROP_FRAME_WIDTH),
		capture->get(cv::CAP_PROP_FRAME_HEIGHT));
	printf("Expected size: %d x %d \n.", refs.width, refs.height);
	
	cv::namedWindow("Colour Video", cv::WINDOW_AUTOSIZE);
	cv::Mat frame;

	bool grey_enabled = false;
	bool my_grey_enabled = false;
	bool cartoon_bool = false;
	bool negative_bool = false;

	while (true)
	{
		*capture >> frame; 
		//get new frame from camera, treating as stream.

		if (frame.empty()){
			printf("Frame is empty"); 
			break;
		}
		// Display the current stream of images in the video stream.
		cv::imshow("Colour Video", frame);

		cv::Mat greyscale_img;
		greyscale_img = cv::Mat::zeros(frame.rows, frame.cols, CV_32F);

		cv::Mat my_greyscale_img;
		my_greyscale_img = cv::Mat::zeros(frame.rows, frame.cols, CV_32FC3);

		// Display the greyscale version of the video
		if (grey_enabled){
			cv::cvtColor(frame, greyscale_img, cv::COLOR_RGB2GRAY, 0);
			cv::namedWindow("Greyscale Video", cv::WINDOW_AUTOSIZE);
			cv::imshow("Greyscale Video", greyscale_img);
		}
		else if (cv::getWindowProperty("Greyscale Video", cv::WND_PROP_VISIBLE) == 1.0){
				cv::destroyWindow("Greyscale Video");
			}

		// Display the greyscale version of the video
		if (my_grey_enabled) {
			greyscale_of_image(frame, my_greyscale_img);
			// Scale the range of
			cv::Mat scaled_img;
			cv::convertScaleAbs(my_greyscale_img, scaled_img);
			cv::namedWindow("My Grey Scale", cv::WINDOW_AUTOSIZE);
			cv::imshow("My Grey Scale", scaled_img);
		}
		else if (cv::getWindowProperty("My Grey Scale", cv::WND_PROP_VISIBLE) == 1.0) {
			cv::destroyWindow("My Grey Scale");
		}

		if (cartoon_bool)
		{
			cv::Mat cartoon_image = cv::Mat(frame.size(), CV_16SC3);
			cartoon(frame, cartoon_image, 15, 20);
			cv::Mat scaled_cartoon_image;
			cv::convertScaleAbs(cartoon_image, scaled_cartoon_image);
			cv::namedWindow("Cartoon Image", cv::WINDOW_AUTOSIZE);
			cv::imshow("Cartoon Image", scaled_cartoon_image);
		}

		if (negative_bool) {
			cv::Mat negative_film = cv::Mat(frame.size(), frame.type());
			negativeImage(frame, negative_film);
			cv::namedWindow("Negative film", cv::WINDOW_AUTOSIZE);
			cv::imshow("Negative film", negative_film);
		}

		//Look for the keystroke entered.
		char key = cv::waitKey(10);
		if (key == 'q')
		{
			printf("Terminating the program as 'q' is pressed.");
			break;
		}
		else if (key == 's')
		{	
			char fileName[50]; // buffer for the file name of the snapshot.
			std::time_t time_now = std::time(0);   // get time now
			std::tm* tm_now = std::localtime(&time_now); // To convert time into tm struct for easier computation.
			/* Reference : https://en.cppreference.com/w/cpp/chrono/c/tm */
			// Format file name as IMG_yyyy_mm_dd_HH_MM_SS.png
			sprintf(fileName, "output/IMG_COL_%04d_%02d_%02d_%02d_%02d_%02d.png", 
				(tm_now->tm_year + 1900), (tm_now->tm_mon + 1), tm_now->tm_mday,
				tm_now->tm_hour, tm_now->tm_min, tm_now->tm_sec);
			// Store the snapshot/frame to a file.
			cv::imwrite(fileName, frame);
			if (grey_enabled)
			{
				sprintf(fileName, "output/IMG_GREY_%04d_%02d_%02d_%02d_%02d_%02d.png",
					(tm_now->tm_year + 1900), (tm_now->tm_mon + 1), tm_now->tm_mday,
					tm_now->tm_hour, tm_now->tm_min, tm_now->tm_sec);
				cv::imwrite(fileName, greyscale_img);
			}
			if (my_grey_enabled)
			{
				sprintf(fileName, "output/IMG_MYGREY_%04d_%02d_%02d_%02d_%02d_%02d.png",
					(tm_now->tm_year + 1900), (tm_now->tm_mon + 1), tm_now->tm_mday,
					tm_now->tm_hour, tm_now->tm_min, tm_now->tm_sec);
				cv::imwrite(fileName, my_greyscale_img);
			}
		}
		else if (key == 'g') {
			grey_enabled = true;
			continue;
		}
		else if (key == 'c') {
			grey_enabled = false;
			my_grey_enabled = false;
			continue;
		}
		else if (key == 'h')
		{
			my_grey_enabled = true;
			continue;
		}
		else if (key == 'o') {
			cartoon_bool = true;
			continue;
		}
		else if( key == 'n')
		{
			negative_bool = true;
			continue;
		}
	}

	delete capture;
	return 0;
}