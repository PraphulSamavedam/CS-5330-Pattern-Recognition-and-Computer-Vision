#include <opencv2/opencv.hpp>
#include <vector>

int main(int argc, char argv[]) {
	printf_s("Hello world");
	char filePath[32] = "data/checkerboard.png";
	cv::Mat boardImg = cv::imread(filePath);
	if (boardImg.data == NULL)
	{
		printf("Error reading the file\n"); 
		exit(-100);
	}
	
	std::vector<cv::Point2f> corners;
	cv::Size patternSize;
	patternSize.height = 6;
	patternSize.width = 9;

	bool status = cv::findChessboardCorners(boardImg, patternSize, corners);
	if (status)
	{
		printf("Successfully obtained the chessboard corners.");
		for (int i = 0; i < corners.size(); i++)
		{
			printf("Corner: %d are %.02f, %.02f\n", i, corners[i].x, corners[i].y);
			cv::circle(boardImg, corners[i], 9, cv::Scalar(255, 0, 255),-1);
		}
	}
	else
	{
		printf("Missed to find the chessboard");
		exit(-400);
	}
	while (true) {
		cv::namedWindow("Image", cv::WINDOW_GUI_EXPANDED);
		cv::imshow("Image", boardImg);
		char key = cv::waitKey(0);
		if (key == 'q')
		{
			cv::destroyAllWindows();
			break;
		}
	}
	return 0;
}