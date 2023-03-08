/**
* Weitte
This

*/

#include <opencv2/opencv.hpp>
#include <vector>
#include "../include/tasks.h"


int main(int argc, char argv[]) {

	cv::VideoCapture* capture = new cv::VideoCapture(0);
	// Check if any video capture device is present.
	if (!capture->isOpened())
	{
		printf("Unable to open the primary video device.\n");
		return(-404);
	}

	cv::Size refs((int)capture->get(cv::CAP_PROP_FRAME_WIDTH),
		capture->get(cv::CAP_PROP_FRAME_HEIGHT));
	printf("Camera Capture size: %d x %d \n.", refs.width, refs.height);
	
	cv::Mat frame;

	while (true) {
		*capture >> frame;
		//get new frame from camera, treating as stream.
		if (frame.empty()) {
			printf("Frame is empty");
			break;
		}

		std::vector<cv::Point2f> corners;
		int status = detectAndExtractChessBoardCorners(frame, corners);
		cv::namedWindow("Image", cv::WINDOW_GUI_EXPANDED);
		cv::imshow("Image", frame);
		char key = cv::waitKey(10);
		if (key == 'q')
		{
			cv::destroyAllWindows();
			break;
		}
		else if (key == 's') {
			printf("Waiting for the point set calculation.\n");
			if (status) {
			// Chessboard exists so update the 
			}
			else
			{
				// Need not store. 
			}
			exit(-100);

		}
	}
	return 0;
}