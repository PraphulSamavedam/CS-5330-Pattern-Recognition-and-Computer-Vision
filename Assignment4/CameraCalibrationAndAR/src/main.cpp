#include <opencv2/opencv.hpp>
#include <vector>

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
	cv::Size patternSize = cv::Size(9, 6); // Width = 9, Height = 6

	bool status = cv::findChessboardCorners(image, patternSize, corners);
	if (status)
	{
		printf("Successfully obtained the chessboard corners.");
		for (int i = 0; i < corners.size(); i++)
		{
			printf("Corner: %d are %.02f, %.02f\n", i, corners[i].x, corners[i].y);
			cv::circle(image, corners[i], 9, cv::Scalar(255, 0, 255),-1);
		}
		cv::Mat grayscaleImage;
		cv::cvtColor(image, grayscaleImage, cv::COLOR_BGR2GRAY);
		// Grayscale image  to have single channel image to focus on refining the corners

		// Refining the corners detected
		cv::cornerSubPix(grayscaleImage, corners, cv::Size(5, 5), cv::Size(-1, 1),
			cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 0.001));

		cv::drawChessboardCorners(image, cv::Size(9, 6), corners, status);
	}
	else
	{
		printf("Missed to find the chessboard");
		exit(-400);
	}
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