/** Written by: Samavedam Manikhanta Praphul
*				Poorna Chandra Vemula
* This functions works on a calibrated camera to start augmented reality. 
* 
*/

#include <opencv2/opencv.hpp> // Required for openCV functions.
#include "../include/csv_util.h" // Reading the csv file containing the camera intrinsic parameters
#include "../include/tasks.h" // For detectAndExtractChessBoardCorners function

int main(int argc, char* argv[]) {
	//char paramsFile[32];
	char paramsFile[32] = "../resources/cameraParams.csv";
	/*assert(argc > 1);
	strcpy(paramsFile, argv[1]);*/

	char metric_name_0[13] = "cameraMatrix";
	char metric_name_1[10] = "distCoeff";

	std::vector<char*> metricNames;
	std::vector<std::vector<float>> data;
	read_metric_data_csv(paramsFile, metricNames, data, false);

	assert(strcpy(metric_name_0, metricNames[0]) == 0);
	assert(strcpy(metric_name_1, metricNames[1]) == 0);

	int metric_values_length = data[0].size();
	cv::Mat cameraMatrix = cv::Mat::zeros(3, 3, CV_64FC1);
	for (int index = 0; index < metric_values_length; index++) {
		cameraMatrix.at<float>(index / 3, index % 3) = data[0][index];
	}

	std::vector<float> distortionCoefficients;
	metric_values_length = data[1].size();
	for (int index = 0; index < metric_values_length; index++) {
		distortionCoefficients.push_back(data[1][index]);
	}

	// Assuming we have succesfully read the parameters from the csv file, let us proceed for live video

	// Open the video capture to show live video. 
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
		std::vector<cv::Point2f> corners_set;
		bool status = detectAndExtractChessBoardCorners(frame, corners_set);
		if (status)
		{
			// ChessBoard exists in this frame.
			printf("Chess board exists in this frame\n");
			// Build the points set from the corner set
			std::vector<cv::Vec3f> points_set;
			buildPointsSet(corners_set, points_set);

			// Create placeholders for vectors of translation and rotation
			std::vector<cv::Mat> rVecs;
			std::vector<cv::Mat> tVecs;

			// Solve for the pose and position of the camera based on the capture. 
			cv::solvePnP(corners_set, points_set, cameraMatrix, distortionCoefficients, rVecs, tVecs);

			printf("Rotation vector is as follows: \n");
			for (auto rotation : rVecs)
			{
				std::cout << rotation << std::endl;
			}
			for (auto translation : tVecs)
			{
				std::cout << translation << std::endl;
			}
		}
		else {
			printf("Chessboard corners are not found.\n");
		}
	}
	return 0;
}