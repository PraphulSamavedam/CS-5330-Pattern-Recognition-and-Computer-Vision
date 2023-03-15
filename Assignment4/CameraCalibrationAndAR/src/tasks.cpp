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
										int pointsPerRow, int pointsPerColumn, bool echo){
	cv::Size patternSize = cv::Size(pointsPerRow, pointsPerColumn); // Width = 9, Height = 6

	bool status = cv::findChessboardCorners(srcImage, patternSize, corners);
	if (status)
	{
		printf("Successfully obtained the chessboard corners.\n");
		// Mark each corner caught with Magenta Circle.
		
		if (echo) {
			for (int i = 0; i < corners.size(); i++)
			{
				//	cv::circle(srcImage, corners[i], 9, cv::Scalar(255, 0, 255), -1);
				printf("Corner %d: %.02f, %.02f\n", i, corners[i].x, corners[i].y);
			}
		}
		
		// Print the details of capture
		printf("First corner: %.02f, %.02f\n", corners[0].x, corners[0].y);
		printf("Total number of corners found: %zd\n", corners.size());

		if (echo)
		{
			printf("Obtained the chessboard corners, refining the corners....\n");
		}
		// Grayscale image  to have single channel image to focus on refining the corners
		cv::Mat grayscaleImage;
		cv::cvtColor(srcImage, grayscaleImage, cv::COLOR_BGR2GRAY);

		// Refining the corners detected
		cv::cornerSubPix(grayscaleImage, corners, cv::Size(5, 5), cv::Size(-1, 1),
			cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 0.001));

		if (echo) {
			printf("Draw refined corners.\n");
		}
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
* @param vir_obj_object_pts set of points corresponding to which the corners are found. 
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


/*This function populates the vir_obj_object_pts for the virtual object chosen in object option.
* @param vir_obj_object_pts set of points of the virtual object.
* @return True all points are provided.
*          False if invalid object option is passed.
*/
bool buildVirtualObjectPoints(std::vector<cv::Vec3f>& vir_obj_object_pts, char object) {
	vir_obj_object_pts.clear();
	if (object == 'h')
	{	// Base Floor
		vir_obj_object_pts.push_back(cv::Vec3f(0, 0, 0));
		vir_obj_object_pts.push_back(cv::Vec3f(4, 0, 0));
		vir_obj_object_pts.push_back(cv::Vec3f(4,-4, 0));
		vir_obj_object_pts.push_back(cv::Vec3f(0,-4, 0));

		// Roof
		vir_obj_object_pts.push_back(cv::Vec3f(0, 0, 4));
		vir_obj_object_pts.push_back(cv::Vec3f(4, 0, 4));
		vir_obj_object_pts.push_back(cv::Vec3f(4,-4, 4));
		vir_obj_object_pts.push_back(cv::Vec3f(0,-4, 4));

		// Center Line
		vir_obj_object_pts.push_back(cv::Vec3f(2, 0, 6));
		vir_obj_object_pts.push_back(cv::Vec3f(2,-4, 6));
		return true;
	}
	return false;
}

/** This function draws the virtual object in the image based on the virtual object chosen.
* @param image address of the image on which virtual object needs to be drawn
* @param vir_obj_img_pts image points of the virtual object.
* @param object virtual object which is passed to drawn.
* @return True if  virtual object is successfully drawn on image.
*         False if virtual object cannot be drawn.
*/
bool drawVirtualObject(cv::Mat& image, std::vector<cv::Vec2f>& vir_obj_img_pts, char object) {
	if (object == 'h')
	{
		// Draw the base floor
		cv::line(image, cv::Point2f(vir_obj_img_pts[0]), cv::Point2f(vir_obj_img_pts[1]), cv::Scalar(0, 255, 255), 2);
		cv::line(image, cv::Point2f(vir_obj_img_pts[1]), cv::Point2f(vir_obj_img_pts[2]), cv::Scalar(0, 255, 255), 2);
		cv::line(image, cv::Point2f(vir_obj_img_pts[2]), cv::Point2f(vir_obj_img_pts[3]), cv::Scalar(0, 255, 255), 2);
		cv::line(image, cv::Point2f(vir_obj_img_pts[3]), cv::Point2f(vir_obj_img_pts[0]), cv::Scalar(0, 255, 255), 2);

		// Draw the one wall
		cv::line(image, cv::Point2f(vir_obj_img_pts[1]), cv::Point2f(vir_obj_img_pts[5]), cv::Scalar(0, 255, 255), 2);
		cv::line(image, cv::Point2f(vir_obj_img_pts[5]), cv::Point2f(vir_obj_img_pts[6]), cv::Scalar(0, 255, 255), 2);
		cv::line(image, cv::Point2f(vir_obj_img_pts[6]), cv::Point2f(vir_obj_img_pts[2]), cv::Scalar(0, 255, 255), 2);

		//Draw the other wall
		cv::line(image, cv::Point2f(vir_obj_img_pts[0]), cv::Point2f(vir_obj_img_pts[4]), cv::Scalar(0, 255, 255), 2);
		cv::line(image, cv::Point2f(vir_obj_img_pts[4]), cv::Point2f(vir_obj_img_pts[7]), cv::Scalar(0, 255, 255), 2);
		cv::line(image, cv::Point2f(vir_obj_img_pts[7]), cv::Point2f(vir_obj_img_pts[3]), cv::Scalar(0, 255, 255), 2);

		// Draw the roof 
		cv::line(image, cv::Point2f(vir_obj_img_pts[6]), cv::Point2f(vir_obj_img_pts[9]), cv::Scalar(255, 0, 255), 2);
		cv::line(image, cv::Point2f(vir_obj_img_pts[7]), cv::Point2f(vir_obj_img_pts[9]), cv::Scalar(255, 0, 255), 2);
		cv::line(image, cv::Point2f(vir_obj_img_pts[5]), cv::Point2f(vir_obj_img_pts[8]), cv::Scalar(255, 0, 255), 2);
		cv::line(image, cv::Point2f(vir_obj_img_pts[4]), cv::Point2f(vir_obj_img_pts[8]), cv::Scalar(255, 0, 255), 2);

		cv::line(image, cv::Point2f(vir_obj_img_pts[8]), cv::Point2f(vir_obj_img_pts[9]), cv::Scalar(255, 0, 255), 2);
		return true;
	}
	return false;
}
