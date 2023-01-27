#include <opencv2/opencv.hpp>
#include "../include/extensions.h"
//#include <iterator>
#include <map>

int main(int argc, char* argvs[]) {
	/*char filename[250] = "C:\\Users\\Samavedam\\Desktop\\lover.jpg";
	cv::Mat frame = cv::imread(filename);*/

	cv::VideoCapture* capture = new cv::VideoCapture(0);
	if (!capture->isOpened())
	{
		printf("Unable to open capturing device.\n");
		return(-404);
	}
	cv::Size refs((int)capture->get(cv::CAP_PROP_FRAME_WIDTH),
		capture->get(cv::CAP_PROP_FRAME_HEIGHT));
	printf("Video resolution: %d x %d \n.", refs.width, refs.height);
	int window_size = cv::WINDOW_GUI_NORMAL;
	cv::Mat frame;
	cv::namedWindow("Colour Video", window_size);

	cv::Mat laplacianImage;

	/*char filename[250] = "data\\standard\\snp_noise_image.png";
	cv::Mat snp_image = cv::imread(filename);
	cv::Mat medianImage3x3;
	snp_image.copyTo(medianImage3x3);
	medianFilter3x3(snp_image, medianImage3x3);
	cv::imshow("Salt and Pepper Noise image", snp_image);
	cv::imshow("Median Filtered", medianImage3x3);*/

	for (;;)
	{
		*capture >>frame;
		//get new frame from camera, treating as stream.

		if (frame.empty()) {
			printf("Frame is empty");
			break;
		}

		// Display the current stream of images in the video stream.
		cv::imshow("Colour Video", frame);

		laplacianImage = cv::Mat(frame.size(), frame.type());
		laplacian_filter(frame, laplacianImage);
		cv::imshow("Laplacian", laplacianImage);

		cv::Mat boxBlurImage3x3 = cv::Mat::zeros(frame.size(), frame.type());
		boxBlur3x3Brute(frame, boxBlurImage3x3);
		cv::imshow("Box Blurred 3x3", boxBlurImage3x3);

		cv::Mat boxBlurImage3x3Brute = cv::Mat::zeros(frame.size(), frame.type());
		boxBlur3x3Brute(frame, boxBlurImage3x3Brute);
		cv::imshow("Box Blurred 3x3 Brute", boxBlurImage3x3Brute);

		cv::Mat boxBlurImage5x5Brute = cv::Mat::zeros(frame.size(), frame.type());
		boxBlur5x5Brute(frame, boxBlurImage5x5Brute);
		cv::imshow("Box Blurred 5x5 Brute", boxBlurImage5x5Brute);

		cv::Mat boxBlurImage5x5S = cv::Mat::zeros(frame.size(), frame.type());
		boxBlur5x5(frame, boxBlurImage5x5S);
		cv::imshow("Box Blurred 5x5", boxBlurImage5x5S);

		char key = cv::waitKey(10);
		if (key == 'q')
		{
			cv::destroyAllWindows();
			break;
		}
		else if(key == 'e') // Laplacian to find edges
		{

		}
	}
	return 0;
}