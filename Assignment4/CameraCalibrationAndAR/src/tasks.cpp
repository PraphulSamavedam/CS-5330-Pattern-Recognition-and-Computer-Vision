/** 
* Written by: Poorna Chandra Vemula, Samavedam Manikhanta Praphul
* Version: 1.0
* This file has the utitity functions required for the main functionality.
*/

#include <opencv2/opencv.hpp>

/*This function checks for the presence of chessboard in the image. 
* If found, prints the first corner found along with the number of corners found.
* @param srcImage the address of source Image for which chess board corners are to be extracted. 
* @param corners output of the refined chessboard corners found in the image. 
* @return True if the chessboard is found and processing is complete.
*		  False if the operation is not successful.
* @Note the chess board image is supposed to have 9 internal points along row and 6 internal points along column.
*/
bool detectAndExtractChessBoardCorners(cv::Mat& srcImage, std::vector<cv::Point2f>& corners,
										int pointsPerRow, int pointsPerColumn){
	cv::Size patternSize = cv::Size(pointsPerRow, pointsPerColumn); // Width = 9, Height = 6

	bool status = cv::findChessboardCorners(srcImage, patternSize, corners);
	if (status)
	{
		printf("Successfully obtained the chessboard corners.\n");
		// Mark each corner caught with Magenta Circle.
		
		for (int i = 0; i < corners.size(); i++)
		{
		//	cv::circle(srcImage, corners[i], 9, cv::Scalar(255, 0, 255), -1);
			// printf("Corner %d: %.02f, %.02f\n", i, corners[i].x, corners[i].y);
		}
		
		// Print the details of capture
		printf("First corner: %.02f, %.02f\n", corners[0].x, corners[0].y);
		printf("Total number of corners found: %zd\n", corners.size());

		// Grayscale image  to have single channel image to focus on refining the corners
		cv::Mat grayscaleImage;
		cv::cvtColor(srcImage, grayscaleImage, cv::COLOR_BGR2GRAY);

		// Refining the corners detected
		cv::cornerSubPix(grayscaleImage, corners, cv::Size(5, 5), cv::Size(-1, 1),
			cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 0.001));

		// Draw the refined chess board corners
		cv::drawChessboardCorners(srcImage, cv::Size(pointsPerRow, pointsPerColumn), corners, status);
	}
	else {
		printf("Chessboard corners are not found.\n");
		return false;
	}
	return true;
}


/*This function populates the points_list for the corners list provided.
* @param corners_set set of points in the world euclidean. 
* @param points_set set of points corresponding to which the corners are found. 
* @return True if the chessboard is found and processing is complete.
*		  False if the operation is not successful.
* @Note the chess board image is supposed to have 9 internal points along row and 6 internal points along column.
*/
bool buildPointsSet(std::vector<cv::Point2f>& corners, std::vector<cv::Vec3f>& points,	int pointsPerRow, int pointsPerColumn) {
	// printf("Called Build Points Set\n");
	// Ensure that all corners are captured to proceed
	try
	{
		assert(corners.size() == (pointsPerRow * pointsPerColumn));
	}
	catch (const std::exception&)
	{
		printf("Invalid number of corners are passed");
		return -1;
	}
	
	// Populated the points based on the corners 
	points.clear(); // Width = 9, Height = 6 are default values
	for (int index = 0; index < corners.size(); index++)
	{
		points.push_back(cv::Vec3f(index % pointsPerRow, - index / pointsPerRow, 0));
	}
	return 0;
}
