/** 
* Weitte
This 

*/

#include <opencv2/opencv.hpp>
#include <vector>
#include "../include/tasks.h"

int main(int argc, char argv[]) {
	printf_s("Hello world");
	char filePath[32] = "resources/checkerboard.png";
	cv::Mat image = cv::imread(filePath);
	if (image.data == NULL)
	{
		printf("Error reading the file\n"); 
		exit(-404);
	}
	std::vector<cv::Point2f> corners;
	detectAndExtractChessBoardCorners(image, corners);
	while (true) {
		cv::namedWindow("Image", cv::WINDOW_GUI_EXPANDED);
		cv::imshow("Image", image);
		char key = cv::waitKey(0);
		if (key == 'q')
		{
			cv::destroyAllWindows();
			break;
		}
	}
	return 0;
}