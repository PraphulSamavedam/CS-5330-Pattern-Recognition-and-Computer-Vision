/** Written by: Samavedam Manikhanta Praphul
*                Poorna Chandra Vemula
* This functions works on a calibrated camera to start augmented reality.
*
*/

#define _CRT_SECURE_NO_WARNINGS // To supress strcpy warnings


#include <opencv2/opencv.hpp> // Required for openCV functions.
//#include <opencv2/aruco.hpp>  //  Required for aruco
#include "../include/csv_util.h" // Reading the csv file containing the camera intrinsic parameters
#include "../include/tasks.h" // For detectAndExtractChessBoardCorners function

int main(int argc, char* argv[]) {

	// Configurable variables
	char paramsFile[32];
	bool debug = false;
	char metric_name_0[13] = "cameraMatrix";
	char metric_name_1[10] = "distCoeff";
	char projectedFrameName[32] = "Projected Exterior points";
	char prVirObjFrameName[32] = "Projected Virtual Object";
	char virtual_object = 'a';

	/*assert(argc > 1);
	strcpy(paramsFile, argv[1]);*/
	strcpy(paramsFile, "resources/cameraParams.csv");

	std::vector<char*> metricNames;
	std::vector<std::vector<float>> data;
	int status = read_metric_data_csv(paramsFile, metricNames, data, true);

	assert(status == 0);
	printf("Data is read to have length: %zd \n", data.size());

	// Error check for the metric order.
	assert(strcmp(metric_name_0, metricNames[0]) == 0);
	assert(strcmp(metric_name_1, metricNames[1]) == 0);

	int metric_values_length = data[0].size();
	cv::Mat cameraMatrix = cv::Mat::zeros(3, 3, CV_64FC1);
	for (int index = 0; index < metric_values_length; index++) {
		cameraMatrix.at<double>(index / 3, index % 3) = data[0][index];
	}

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

	std::vector<float> distortionCoefficients;
	metric_values_length = data[1].size();
	for (int index = 0; index < metric_values_length; index++) {
		distortionCoefficients.push_back(data[1][index]);
	}

	if (debug) {
		printf("Distortion coefficients are read as follows: \n");
		for (int index = 0; index < metric_values_length; index++)
		{
			printf("%.04f ", distortionCoefficients[index]);
		}
		std::cout << std::endl;
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
	if (debug) {
		printf("Camera Capture size: %d x %d \n.", refs.width, refs.height);
	}
	printf("Camera Capture size: %d x %d \n.", refs.width, refs.height);
	// Create placeholders for vectors of translation and rotation
	cv::Mat rVector;
	cv::Mat tVector;

	//Aruco setup
	cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
	cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
	cv::aruco::ArucoDetector detector(dictionary, detectorParams);

	cv::Mat frame;
	int counter = 0;
	while (true) {
		*capture >> frame;
		//get new frame from camera, treating as stream.
		if (frame.empty()) {
			printf("Frame is empty");
			break;
		}
		cv::Mat detect;
		frame.copyTo(detect);
		std::vector<std::vector<cv::Point2f>> imagePointsForTargets;

		std::vector<cv::Point2f> imagePoints;
		bool status = detectAndExtractChessBoardCorners(detect, imagePoints);
		while (status)
		{
			imagePointsForTargets.push_back(imagePoints);
			float min_x = INFINITY;
			float min_y = INFINITY;
			float max_x = -INFINITY;
			float max_y = -INFINITY;
			for (int index = 0; index < imagePoints.size(); index++)
			{
				min_x = MIN(min_x, imagePoints[index].x);
				min_y = MIN(min_y, imagePoints[index].y);
				max_x = MAX(max_x, imagePoints[index].x);
				max_y = MAX(max_y, imagePoints[index].y);
				std::cout << imagePoints[index] << std::endl;
			}
			std::cout << "Min X: " << min_x << std::endl;
			std::cout << "Min Y: " << min_y << std::endl;
			std::cout << "Max X: " << max_x << std::endl;
			std::cout << "Max Y: " << max_y << std::endl;

			for (int i = min_y; i <= max_y; i++) {
				//std::cout << "Processing row: " << i << std::endl;
				for (int j = min_x; j <= max_x; j++) {
					detect.at<cv::Vec3b>(i, j)[0] = 0;
					detect.at<cv::Vec3b>(i, j)[1] = 0;
					detect.at<cv::Vec3b>(i, j)[2] = 0;
				}
			}
			status = detectAndExtractChessBoardCorners(detect, imagePoints);
			cv::imshow("Detected Chessboard", detect);
			std::cout << "Status " << status << std::endl;
		}


		// Show the image now so that detected chessboard corners are visible.
		cv::imshow("Live Video", frame);
		char key = cv::waitKey(3);

		//detecting Markers
		std::vector<int> ids;
		std::vector<std::vector<cv::Point2f>> corners;
		detector.detectMarkers(frame, corners, ids);


		int targetSize = imagePointsForTargets.size();

		if (targetSize > 0) {
			std::cout << "target size : " << targetSize << std::endl;
			//if (targetSize == 2) { exit(0); }
			cv::Mat projection;
			frame.copyTo(projection);
			cv::Mat VirualObjProjection;
			frame.copyTo(VirualObjProjection);
			for (int target = 0; target < targetSize; target++)
			{
				std::cout << "check loop:" << std::endl;
				// Build the points set from the corner set
				std::vector<cv::Vec3f> objectPoints;
				buildPointsSet(imagePointsForTargets[target], objectPoints);
				std::cout << "point set built:" << std::endl;
				if (debug) { printf("Solving for PnP\n"); }

				if (debug)
				{
					printf("Image Points: \n");
					for (int i = 0; i < imagePointsForTargets[target].size(); i++)
					{
						std::cout << imagePointsForTargets[target][i] << std::endl;
					}
				}

				if (debug)
				{
					printf("Object Points: \n");
					for (int i = 0; i < objectPoints.size(); i++)
					{
						std::cout << objectPoints[i] << std::endl;
					}
				}

				// Solve for the pose and position of the camera based on the capture.
				cv::solvePnP(objectPoints, imagePointsForTargets[target], cameraMatrix, distortionCoefficients, rVector, tVector);
				if (debug)
				{
					printf("Image Points: \n");
					for (int i = 0; i < imagePointsForTargets[target].size(); i++)
					{
						std::cout << imagePointsForTargets[target][i] << std::endl;
					}
				}

				if (debug)
				{
					printf("Object Points: \n");
					for (int i = 0; i < objectPoints.size(); i++)
					{
						std::cout << objectPoints[i] << std::endl;
					}
				}

				// Print the translation and rotation vectors
				printf("Rotation vector of shape (%d, %d) is as follows: \n", rVector.rows, rVector.cols);
				std::cout << rVector << std::endl;

				printf("Translation vector of shape (%d, %d) is as follows: \n", tVector.rows, tVector.cols);
				std::cout << tVector << std::endl;

				// Calculation for image points of exterior object points
				std::vector<cv::Vec2f> projectedImagePoints;
				std::vector<cv::Vec3f> exteriorObjectPoints;
				buildVirtualObjectPoints(exteriorObjectPoints, 'r');

				// Get the projected the exterior points.
				cv::projectPoints(exteriorObjectPoints, rVector, tVector, cameraMatrix, distortionCoefficients, projectedImagePoints);

				// Display the exterior rectangle
				drawVirtualObject(projection, projectedImagePoints, 'r');


				if (debug) {
					printf("Projected points size is %zd:\n", projectedImagePoints.size());
					for (int index = 0; index < projectedImagePoints.size(); index++)
					{
						std::cout << projectedImagePoints[index] << std::endl;
					}
				}

				// Calculations for virutal object projection
				std::vector<cv::Vec2f> projectedVirObjImgPts;
				std::vector<cv::Vec3f> VirObjObjectPts;
				bool validVirtualObject = buildVirtualObjectPoints(VirObjObjectPts, virtual_object);
				// If Valid object then project the virtual object.
				if (validVirtualObject)
				{
					cv::projectPoints(VirObjObjectPts, rVector, tVector, cameraMatrix, distortionCoefficients, projectedVirObjImgPts);

					if (debug) {
						printf("Vitual object's projected points of size %zd are:\n", projectedVirObjImgPts.size());
						for (int index = 0; index < projectedVirObjImgPts.size(); index++)
						{
							std::cout << projectedVirObjImgPts[index] << std::endl;
						}
					}

					// Projecting a virtual object
					drawVirtualObject(VirualObjProjection, projectedVirObjImgPts, virtual_object);

				}
				std::cout << "loop end" << std::endl;

			}

			std::cout << "p" << std::endl;
			cv::imshow(projectedFrameName, projection);
			cv::imshow(prVirObjFrameName, VirualObjProjection);


		}


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

		else {
			printf("Chessboard corners are not found.\n");
			//            if (cv::getWindowProperty(projectedFrameName, cv::WND_PROP_VISIBLE) > 0)
			//            {
			//                cv::destroyWindow(projectedFrameName);
			//            }
			//            if (cv::getWindowProperty(prVirObjFrameName, cv::WND_PROP_VISIBLE) > 0)
			//            {
			//                cv::destroyWindow(prVirObjFrameName);
			//            }

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
	return 0;
}

