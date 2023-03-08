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
	std::vector<std::vector<cv::Point2f>> corners_list;
	std::vector<cv::Point2f> last_corners_set;
	cv::Mat last_image; 

	std::vector<cv::Vec3f> points_set;
	std::vector<cv::Vec3f> last_points_set;
	std::vector<std::vector<cv::Vec3f>> points_list;

	bool last_successful_capture = false;
	while (true) {
		*capture >> frame;
		//get new frame from camera, treating as stream.
		if (frame.empty()) {
			printf("Frame is empty");
			break;
		}
		std::vector<cv::Point2f> corners_set;
		bool status = detectAndExtractChessBoardCorners(frame, corners_set);
		printf("Status of chess board capture in video loop: %d\n", status);
		cv::namedWindow("Image", cv::WINDOW_GUI_EXPANDED);
		cv::imshow("Image", frame);
		char key = cv::waitKey(10);
		if (key == 'q')
		{
			cv::destroyAllWindows();
			for (auto corner : corners_list)
			{
				std::cout << corner << std::endl;
			}
			printf("\nPrinting points. \n");
			for (auto point_set : points_list)
			{
				for (auto point : point_set) {
					std::cout << point << " ";
				}
				std::cout << std::endl;
			}
			break;
		}
		else if (key == 's') {
			//printf("Waiting for the point set calculation.\n");
			if (status) {
				corners_list.push_back(corners_set);
				buildPointsSet(corners_set, points_set);
				points_list.push_back(points_set);
				
				// Mark that there was a successful capture
				last_successful_capture = true;
				last_corners_set = corners_set;
				buildPointsSet(last_corners_set, last_points_set);
				frame.copyTo(last_image); // Save the last image
			}
			else {
				if (last_successful_capture)
				{
					corners_list.push_back(last_corners_set);
					points_list.push_back(last_points_set);
				}
				else
				{
					printf("No previous successful capture found.");
					continue;
				}
			}
		}
	}
	return 0;
}