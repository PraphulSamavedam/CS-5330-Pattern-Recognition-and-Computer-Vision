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
*		non zero if the operation is not successful. 
* @note the chess board image is supposed to have 9 internal points along row and 6 internal points along column.
*/
int detectAndExtractChessBoardCorners(cv::Mat& srcImage, std::vector<cv::Point2f>& corners);