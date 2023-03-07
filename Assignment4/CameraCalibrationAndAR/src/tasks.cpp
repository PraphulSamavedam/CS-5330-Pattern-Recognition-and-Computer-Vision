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
* @return 0 if the operation is successful.
*		non zero if the operation is not successful.
* @Note the chess board image is supposed to have 9 internal points along row and 6 internal points along column.
*/
int detectAndExtractChessBoardCorners(cv::Mat& srcImage, std::vector<cv::Point2f>& corners) {
	cv::Size patternSize = cv::Size(9, 6); // Width = 9, Height = 6

	bool status = cv::findChessboardCorners(srcImage, patternSize, corners);
	if (status)
	{
		printf("Successfully obtained the chessboard corners.\n");
		// Mark each corner caught with Magenta Circle.
		for (int i = 0; i < corners.size(); i++)
		{
			cv::circle(srcImage, corners[i], 9, cv::Scalar(255, 0, 255), -1);
		}
		printf("First corner: %.02f, %.02f\n", corners[0].x, corners[0].y);
		printf("Total number of corners found: %zd\n", corners.size());

		// Grayscale image  to have single channel image to focus on refining the corners
		cv::Mat grayscaleImage;
		cv::cvtColor(srcImage, grayscaleImage, cv::COLOR_BGR2GRAY);

		// Refining the corners detected
		cv::cornerSubPix(grayscaleImage, corners, cv::Size(5, 5), cv::Size(-1, 1),
			cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 0.001));

		// Draw the refined chess board corners
		cv::drawChessboardCorners(srcImage, cv::Size(9, 6), corners, status);
	}
	else {
		printf("Chessboard corners are not found.\n");
		return -1;
	}
	return 0;
}