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
		// printf("Status of chess board capture in video loop: %d\n", status);
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
		else if (key == 'c')
		{

			// Image Size is refs
			cv::Mat cameraMatrix = cv::Mat::zeros(3, 3, CV_64FC1);
			cameraMatrix.at<float>(0,0) = 1;
			cameraMatrix.at<float>(1, 1) = 1;
			cameraMatrix.at<float>(2, 2) = 1;
			cameraMatrix.at<float>(0, 2) = frame.cols / 2;
			cameraMatrix.at<float>(1, 2) = frame.rows/2;
			std::vector<float> distortionCoefficients;
			for (int i = 0; i < 8; i++)
			{
				distortionCoefficients.push_back(0.0);
			}
			std::vector<cv::Mat> rVecs;
			std::vector<cv::Mat> tVecs;

			printf("Pre Camera Matrix\n");
			for (int i=0;i<3;i++ )
			{
				for (int j = 0; j < 3; j++)
				{
					printf("[%d,%d]: %.02f", i, j, cameraMatrix.at<double>(i, j));
				}
				printf("\n");
			}

			printf("Pre distortion coefficients are :\n");
			for (auto dist : distortionCoefficients)
			{
				std::cout << dist << std::endl;
			}
			printf("Using %zd points for calibration.\n", corners_list.size());
			cv::calibrateCamera(points_list, corners_list, refs, cameraMatrix, 
				distortionCoefficients, rVecs, tVecs, cv::CALIB_FIX_ASPECT_RATIO);
			printf("Calibrated the camera.\n");
			printf("Updated distortion coefficients are :\n");
			for (auto dist:distortionCoefficients) 
			{
				std::cout << dist << std::endl;
			}
			printf("Updated Camera Matrix\n");
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					printf("[%d,%d]: %.06f ", i, j, cameraMatrix.at<double>(i, j));
				}
				printf("\n");
			}

			//Calculating re-projection error ToDo Error Calculation failure
			float reprojection_error = 0.0;
			for (int i = 0; i < points_list.size(); i++)
			{
				std::vector<cv::Point3f> original_points_set;
				for (int j = 0; j < points_list[i].size(); j++)
				{
					original_points_set.push_back(cv::Point3f(points_list[i][j][0], points_list[i][j][1], points_list[i][j][2]));
				}
				std::vector<cv::Point2f> projected_corners_set;
				cv::projectPoints(original_points_set, rVecs, tVecs, cameraMatrix, distortionCoefficients, projected_corners_set);

				//debug
				printf("Projected Corners set size:%zd", projected_corners_set.size());
				for (auto vr:projected_corners_set)
				{
					std::cout << vr << std::endl;
				}
				printf("debug step pass");

				//Calculate the error for this projected points set
				for (int j = 0; j < corners_set.size(); j++)
				{
					double dx = (corners_list[i][j].x - projected_corners_set[j].x);
					double dy = (corners_list[i][j].y - projected_corners_set[j].y);
					reprojection_error += ((dx * dx) + (dy * dy));
				}

			}
			printf("Reprojection Error: %.06f ", reprojection_error);
			printf("Closing the function");
			break;
		}
	}
	return 0;
}