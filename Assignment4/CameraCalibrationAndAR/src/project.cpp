/** Written by: Samavedam Manikhanta Praphul
*				Poorna Chandra Vemula
* This functions works on a calibrated camera to start augmented reality.
*
*/

#define _CRT_SECURE_NO_WARNINGS // To supress strcpy warnings


#include <opencv2/opencv.hpp> // Required for openCV functions.
#include "../include/csv_util.h" // Reading the csv file containing the camera intrinsic parameters
#include "../include/tasks.h" // For detectAndExtractChessBoardCorners function

int main(int argc, char* argv[]) {
	//char paramsFile[32];
	char paramsFile[32] = "resources/cameraParams.csv";
	/*assert(argc > 1);
	strcpy(paramsFile, argv[1]);*/

	char metric_name_0[13] = "cameraMatrix";
	char metric_name_1[10] = "distCoeff";

	std::vector<char*> metricNames;
	std::vector<std::vector<float>> data;
	int status = read_metric_data_csv(paramsFile, metricNames, data, false);

	assert(status == 0);
	printf("Data is read to have length: %zd \n", data.size());

	assert(strcpy(metric_name_0, metricNames[0]) == 0);
	assert(strcpy(metric_name_1, metricNames[1]) == 0);

	int metric_values_length = data[0].size();
	cv::Mat cameraMatrix = cv::Mat::zeros(3, 3, CV_64FC1);
	for (int index = 0; index < metric_values_length; index++) {
		cameraMatrix.at<float>(index / 3, index % 3) = data[0][index];
	}
	printf("Camera Matrix is read as follows: \n");
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			std::cout << cameraMatrix.at<float>(i, j) << " ";
		}
		std::cout << std::endl;
	}

	std::vector<float> distortionCoefficients;
	//printf("Distortion coefficients are read as follows: \n");
	metric_values_length = data[1].size();
	for (int index = 0; index < metric_values_length; index++) {
		distortionCoefficients.push_back(data[1][index]);
		// std::cout << data[1][index] << std::endl;
	}

	printf("Distortion coefficients are read as follows: \n");
	for (int index = 0; index < metric_values_length; index++)
	{
		std::cout << distortionCoefficients[index] << " ";
	}
	std::cout << std::endl;

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

	// Create placeholders for vectors of translation and rotation
	cv::Mat rVector;
	cv::Mat tVector;

	cv::Mat frame;
	while (true) {
		*capture >> frame;
		//get new frame from camera, treating as stream.
		if (frame.empty()) {
			printf("Frame is empty");
			break;
		}
		cv::imshow("Live Video", frame);
		cv::waitKey(3);
		std::vector<cv::Point2f> corners_set;
		bool status = detectAndExtractChessBoardCorners(frame, corners_set);
		if (status)
		{
			// ChessBoard exists in this frame.
			printf("Chess board exists in this frame\n");

			// Build the points set from the corner set
			std::vector<cv::Vec3f> points_set;
			buildPointsSet(corners_set, points_set);
			printf("Solving for PnP\n");

			// Solve for the pose and position of the camera based on the capture. 
			cv::solvePnP(points_set, corners_set, cameraMatrix, distortionCoefficients, rVector, tVector);

			printf("Rotation vector is of shape (%d, %d) follows: \n", rVector.rows, rVector.cols);
			std::cout << rVector << std::endl;
			printf("Rotation vector is as follows: \n");
			for (int row = 0; row < rVector.rows; row++)
			{
				for (int col = 0; col < rVector.cols; col++)
				{
					std::cout << rVector.at<int>(row, col) << " ";
				}
				std::cout << std::endl;
			}

			printf("Translation vector is of shape (%d, %d) follows: \n", tVector.rows, tVector.cols);
			std::cout << tVector << std::endl;
			printf("Translation vector is as follows: \n");
			for (int row = 0; row < tVector.rows; row++)
			{
				for (int col = 0; col < tVector.cols; col++)
				{
					std::cout << tVector.at<int>(row, col) << " ";
				}
				std::cout << std::endl;
			}
			break;
			// cv::waitKey(0); // To Capture the details for report. 
			std::vector<cv::Vec2f> projectedObjectPoints;
			cv::projectPoints(points_set, rVector, tVector, cameraMatrix, distortionCoefficients, projectedObjectPoints);

			printf("Projected points are :\n");
			for (int index = 0; index < projectedObjectPoints.size(); index++)
			{
				std::cout << projectedObjectPoints[index] << std::endl;
			}
			// cv::imshow()
	
		}
		else {
			printf("Chessboard corners are not found.\n");
		}
	}
	return 0;
}