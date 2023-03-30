/** Written by: Samavedam Manikhanta Praphul
*                Poorna Chandra Vemula
* This functions works on a calibrated camera to project some virtual object either live 
* or on static iamge or an existing video based on the params
* Usage: projectStaticOrLive.exe <cam_intr_path> <mode> <static_file_path> <virtual_object>
* Defaults:
* cam_int_path = "resources/cameraParams.csv"
* mode = "l" options: "l", "i", "v" for live, staticImage, staticVideo
* static_file_path = NULL
* virtual_object = "a" options: "h", "a", "c", "t", "r"
*/

#define _CRT_SECURE_NO_WARNINGS // To supress strcpy warnings


#include <opencv2/opencv.hpp> // Required for openCV functions.
#include "../include/csv_util.h" // Reading the csv file containing the camera intrinsic parameters
#include <fstream> // Required for accessing static images/video
#include "../include/utils.h"  // Required for code simplicity

int main(int argc, char* argv[]) {

	// Configurable variables with default values
	char paramsFile[32] = "resources/cameraParams.csv";
	char mode[2] = "l";
	bool debug = false;
	char metric_name_0[13] = "cameraMatrix";
	char metric_name_1[10] = "distCoeff";
	char metric_name_2[18] = "reprojectionError";
	char detectedFrameName[32] = "Detected";
	char prVirObjFrameName[32] = "Augmented";
	char OriginalFrameName[32] = "Original";
	char path[256] = "resources/checkerboard.png";
	char virtual_object = 'h';
	int window_size = cv::WINDOW_GUI_NORMAL;
	cv::namedWindow(OriginalFrameName, window_size);
	cv::namedWindow(detectedFrameName, window_size);
	cv::namedWindow(prVirObjFrameName, window_size);

	// Override the defaults based on the params passed.
	if (argc > 1)
	{
		strcpy(paramsFile, argv[1]);
	}
	if (argc > 2)
	{
		strcpy(mode, argv[2]);
	}
	if (argc > 3)
	{	// Get the file path
		strcpy(path, argv[3]);
	}
	if (argc > 4)
	{
		virtual_object = argv[4][0];
	}
	std::cout << "Virtual object: " << virtual_object << std::endl;
	std::vector<char*> metricNames;
	std::vector<std::vector<float>> data;
	int status = read_metric_data_csv(paramsFile, metricNames, data, true);

	assert(status == 0);
	if (debug) { printf("Read the data of %zd rows from the parameters file\n", data.size()); }

	// Error check for the metric order.
	assert(strcmp(metric_name_0, metricNames[0]) == 0);
	assert(strcmp(metric_name_1, metricNames[1]) == 0);
	assert(strcmp(metric_name_2, metricNames[2]) == 0);

	// Load the camera matrix
	int metric_values_length = data[0].size();
	cv::Mat cameraMatrix = cv::Mat::zeros(3, 3, CV_64FC1);
	for (int index = 0; index < metric_values_length; index++) {
		cameraMatrix.at<double>(index / 3, index % 3) = data[0][index];
	}

	// Print the camera matrix read
	if (debug) {
		printf("Camera Matrix is read as follows: \n");
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				printf("%0.4f ", cameraMatrix.at<double>(i, j));
			}
			std::cout << std::endl;
		}
	}

	// Load the distortion coefficients
	std::vector<float> distortionCoefficients;
	metric_values_length = data[1].size();
	for (int index = 0; index < metric_values_length; index++) {
		distortionCoefficients.push_back(data[1][index]);
	}

	// Print the distortion coefficients read
	if (debug) {
		printf("Distortion coefficients are read as follows: \n");
		for (int index = 0; index < metric_values_length; index++)
		{
			printf("%.04f ", distortionCoefficients[index]);
		}
		std::cout << std::endl;
	}

	if (debug) { printf("Intrinsic params of the cameara are read successfully.\n"); }
	// Assuming we have succesfully read the parameters from the csv file, let us proceed for projection

	// Create placeholders for vectors of translation and rotation
	cv::Mat rVector;
	cv::Mat tVector;

	if (strcmp(mode, "l") == 0)
	{
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
		if (debug) {
			printf("Camera Capture size: %d x %d \n.", refs.width, refs.height);
		}

		//Aruco setup
		cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
		cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
		cv::aruco::ArucoDetector detector(dictionary, detectorParams);

		cv::Mat frame;
		while (true) {
			*capture >> frame;
			//get new frame from camera, treating as stream.
			if (frame.empty()) {
				printf("Frame is empty");
				break;
			}

			// Show the image now so that detected chessboard cornerPts are visible. 
			cv::imshow(OriginalFrameName, frame);
			char key = cv::waitKey(3);

			cv::Mat image;
			frame.copyTo(image);
			cv::Mat detect;
			frame.copyTo(detect);
			std::vector<cv::Point2f> cornerPts;
			bool status = detectAndExtractChessBoardCorners(detect, cornerPts);
			if (status)
			{
				getCameraPoseAndDrawVirtualObject(image, cornerPts, cameraMatrix, distortionCoefficients, virtual_object);
				cv::imshow(prVirObjFrameName, image);
				cv::imshow(detectedFrameName, detect);
			}
			else {
				printf("Chessboard corners are not found.\n");
				if (cv::getWindowProperty(prVirObjFrameName, cv::WND_PROP_VISIBLE) > 0)
				{
					cv::destroyWindow(prVirObjFrameName);
				}

			}

			//detecting Markers
			std::vector<int> ids;
			std::vector<std::vector<cv::Point2f>> corners;
			detector.detectMarkers(frame, corners, ids);

			// Checking with Arco Markers
			if (ids.size() > 0) {

				// Projected the exterior points.
				cv::Mat projection;
				frame.copyTo(projection);

				cv::aruco::drawDetectedMarkers(projection, corners, ids);
				int nMarkers = corners.size();
				std::vector<cv::Vec3d> rvecs(nMarkers), tvecs(nMarkers);

				float markerLength = 0.05;

				// Set coordinate system
				cv::Mat objPoints(4, 1, CV_32FC3);
				objPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-markerLength / 2.f, markerLength / 2.f, 0);
				objPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(markerLength / 2.f, markerLength / 2.f, 0);
				objPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(markerLength / 2.f, -markerLength / 2.f, 0);
				objPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-markerLength / 2.f, -markerLength / 2.f, 0);

				// Calculate pose for each marker
				for (int i = 0; i < nMarkers; i++) {
					cv::solvePnP(objPoints, corners.at(i), cameraMatrix, distortionCoefficients, rvecs.at(i), tvecs.at(i));
				}

				// Draw axis for each marker
				for (unsigned int i = 0; i < ids.size(); i++) {
					cv::drawFrameAxes(projection, cameraMatrix, distortionCoefficients, rvecs[i], tvecs[i], 0.1);
				}

				cv::imshow("projection markers", projection);

			}


			if (key == 'q')
			{
				break;
			}
			else if (key > 0)
			{
				virtual_object = key;
				std::cout << "Virtual object set as '" << virtual_object << "'" << std::endl;
			}
		}
	}
	else
	{	// Deal with the static file
		FILE* filePtr;
		filePtr = fopen(path, "r");
		if (filePtr == NULL)
		{
			printf("File at path %s provided does not exist.\n", path);
		}
		if (strcmp(mode, "i") == 0)
		{

			cv::Mat image = cv::imread(path);
			if (image.data == NULL)
			{
				printf("Error reading the file %s\n.", path);
				exit(-404);
			}
			std::vector<cv::Point2f> cornerPts;
			cv::Mat detect;
			image.copyTo(detect);
			cv::Mat projected;
			image.copyTo(projected);
			bool status = detectAndExtractChessBoardCorners(detect, cornerPts);
			if (status)
			{
				getCameraPoseAndDrawVirtualObject(projected, cornerPts, cameraMatrix, 
					distortionCoefficients, virtual_object);
			}
			cv::imshow(OriginalFrameName, image);
			cv::imshow(prVirObjFrameName, projected);
			cv::imshow(detectedFrameName, detect);
			cv::waitKey(0);
		}
		else if (strcmp(mode, "v") == 0)
		{
			//ToDo Read a video file and process for each frame. 
			cv::VideoCapture* cap = new cv::VideoCapture(path);
			cv::Mat image;

			float fps = cap->get(cv::CAP_PROP_FPS);
			printf("frames per second = % .04f\n", fps);

			int frame_width = cap->get(cv::CAP_PROP_FRAME_WIDTH);
			int frame_height = cap->get(cv::CAP_PROP_FRAME_HEIGHT);

			char outputFileName[100] = "outcpp.avi";
			cv::VideoWriter video(outputFileName,
				cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
				cv::Size(frame_width, frame_height));
			printf("\nProcessing video \n");
			while (cap->isOpened())
			{
				*cap >> image;
				printf(".");
				//get new frame from camera, treating as stream.
				if (image.empty()) {
					//printf("Image is empty\n");
					break;
				}
				std::vector<cv::Point2f> corners;
				bool status = detectAndExtractChessBoardCorners(image, corners);
				if (status)
				{ // Project on the image the virtual object
					getCameraPoseAndDrawVirtualObject(image, corners, cameraMatrix, distortionCoefficients, virtual_object);
				}
				// Else scenario is not required as the frame has to be the same. 
				video.write(image);
			}
			printf("\nSuccessfully processed static video.\n");
			
		}
		else
		{
			printf("Invalid option: %s", mode);
		}
	}

	return 0;
}
