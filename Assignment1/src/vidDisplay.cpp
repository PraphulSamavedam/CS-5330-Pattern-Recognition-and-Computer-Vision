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
	
	cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);
	cv::Mat frame;

	bool enable_gey = false;

	while (true)
	{
		*capture >> frame; 
		//get new frame from camera, treating as stream.

		if (frame.empty()){
			printf("Frame is empty"); 
			break;
		}
		// Display the current stream of images in the video stream.
		cv::imshow("Video", frame);

		//Look for the keystroke entered.
		char key = cv::waitKey(10);
		if (key == 'q')
		{
			printf("Terminating the program as 'q' is pressed.");
			break;
		}
		if (key == 's')
		{	
			char fileName[50]; // buffer for the file name of the snapshot.
			std::time_t time_now = std::time(0);   // get time now
			std::tm* tm_now = std::localtime(&time_now); // To convert time into tm struct for easier computation.
			/* Reference : https://en.cppreference.com/w/cpp/chrono/c/tm */
			// Format file name as IMG_yyyy_mm_dd_HH_MM_SS.png
			sprintf(fileName, "IMG_%04d_%02d_%02d_%02d_%02d_%02d.png", 
				(tm_now->tm_year + 1900), (tm_now->tm_mon + 1), tm_now->tm_mday,
				tm_now->tm_hour, tm_now->tm_min, tm_now->tm_sec);
			// Store the snapshot/frame to a file.
			cv::imwrite(fileName, frame);
		}
		if (key == 'g') {
			enable_gey = not(enable_gey);
			cv::namedWindow("Colour Image", cv::WINDOW_AUTOSIZE);
			cv::imshow("Colour Image", frame);
			cv::imwrite("Colour Image.png", frame);
			cv::Mat dst;
			dst = cv::Mat::zeros(frame.rows, frame.cols, CV_16U);
			cv::cvtColor(frame, dst, cv::COLOR_RGB2GRAY, 0);
			cv::namedWindow("GreyScale Image", cv::WINDOW_AUTOSIZE);
			cv::imshow("GreyScale Image", dst);
			cv::imwrite("GreyScale Image.png", dst);
		}
		if (enable_gey)
		{
			cv::Mat dst;
			dst = cv::Mat::zeros(frame.rows, frame.cols, CV_16U);
			cv::cvtColor(frame, dst, cv::COLOR_RGB2GRAY, 0);
			cv::namedWindow("Grey Scale", cv::WINDOW_AUTOSIZE);
			cv::imshow("Grey Scale", dst);
		}
		else
		{
			if (cv::getWindowProperty("Grey Scale", cv::WND_PROP_VISIBLE) == 1.0)
			{
				cv::destroyWindow("Grey Scale");
			}
		}

	}

	delete capture;
	return 0;
}