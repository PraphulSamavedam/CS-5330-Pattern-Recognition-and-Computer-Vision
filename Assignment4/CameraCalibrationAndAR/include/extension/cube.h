#ifndef CUBE_H
#define CUBE_H

#include <vector>

class cube
{
private:
	std::vector< cv::Point3f > cube_obj_pts;
	std::vector< cv::Point2f > cube_img_pts;

public:
	cube();
	cv::Mat drawcube(cv::Mat, cv::Mat, cv::Mat, cv::Mat, cv::Mat);
};

#endif