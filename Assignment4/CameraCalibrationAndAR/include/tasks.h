/**
* Written by: Poorna Chandra Vemula, Samavedam Manikhanta Praphul
* Version: 1.0
* This file has the signatures of the utitity functions.
*/

#include <opencv2/opencv.hpp>

/*This function checks for the presence of chessboard in the image.
* If found, prints the first corner found along with the number of corners found.
* @param srcImage the address of source Image for which chess board corners are to be extracted.
* @param corners output of the refined chessboard corners found in the image.
* @return 0 if the operation is successful.
*        non zero if the operation is not successful.
* @note the chess board image is supposed to have 9 internal points along row and 6 internal points along column.
*/
bool detectAndExtractChessBoardCorners(cv::Mat& srcImage, std::vector<cv::Point2f>& corners,
                                        int pointsPerRow = 9, int pointsPerColumn = 6);

/*This function populates the points_list for the corners list provided.
* @param corners_set set of points in the world euclidean.
* @param points_set set of points corresponding to which the corners are found.
* @return True if the chessboard is found and processing is complete.
*          False if the operation is not successful.
* @Note the chess board image is supposed to have 9 internal points along row and 6 internal points along column.
*/
bool buildPointsSet(std::vector<cv::Point2f>& corners, std::vector<cv::Vec3f>& points, int pointsPerRow = 9, int pointsPerColumn = 6);
