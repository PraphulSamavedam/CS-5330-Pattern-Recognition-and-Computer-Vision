/** This program opens a video channel, loops through captures to appear as stream.
* Written by Samavedam Manikhanta Praphul.
* 
* @return  0 if program terminated with success.
		-404 if video device is not found.
		-100 if video device has no frame.
* @note:
	if 'q' is pressed: program terminates.
	if 's' is pressed: program saves the frame as an image 
						to a file with timestamp in the format IMG_<Filter>_yyyy_mm_dd_HH_MM_SS.png
	if 'g' is pressed: GreyScale video feed is on with openCV's cvtColor function. 
	if 'h' is pressed: Alternative Grayscale video feed.
	if 'x' is pressed: Vertical edges are detected using SobelX
	if 'y' is pressed: Horizontal edges are detected using SobelY
	if 'R' is pressed: Only Red Channel is shown.
	if 'G' is pressed: Only Green Channel is shown.
	if 'B' is pressed: Only Blue Channel is shown.
	if 'b' is pressed: Video feed is slightly blurred with gaussian blur.
	if 'm' is pressed: Horizontal and Vertical edges are detected using Gradient Magnitude based on sobelX and sobelY. 
	if 'n' is pressed: Negative video feed is toggled. 
	if 'p' is pressed: 
	*/

#define _CRT_SECURE_NO_WARNINGS // Required to suppress warnings of localtime, sprintf_s

#include <opencv2/opencv.hpp> // All the required openCV functions and definitions.
#include <cstdio>	// Primary C functions, definitions which interact with stdin, stdout.
#include <cstring> // All the required string functions, definitions.
#include <ctime> // Required for functions accessing the timestamp details for the file name.
#include "../include/filters.h" //Required filter functions 

#include "../include/utils.h" // For saving Image
#include "../include/additionalEffects.h" // Additional Special effects.
#include "../include/extensions.h" // Some extensions after special effects.



int main(int argc, char* argv[]) {
	
	cv::VideoCapture* capture = new cv::VideoCapture(0);
	// Check if any video capture device is present.
	if (!capture->isOpened())
	{
		printf("Unable to open the primary video device.\n");
		return(-404);
	}
	cv::Size refs((int)capture->get(cv::CAP_PROP_FRAME_WIDTH),
		capture->get(cv::CAP_PROP_FRAME_HEIGHT));
	printf("Camera Capture size: %d x %d \n.", refs.width, refs.height);
	
	cv::Mat frame;
	int windowSize = cv::WINDOW_AUTOSIZE;

	bool greyEnabled = false;
	cv::Mat greyscaleImg; // Will be initialized when effect is enabled.

	bool myGreyEnabled = false;
	cv::Mat myGreyscaleImg; // Will be initialized when effect is enabled.

	bool blurEnabled = false;
	cv::Mat blurredImg; // Will be initialized when effect is enabled.

	bool cartoonEnabled = false;
	cv::Mat cartoonImg; // Will be initialized when effect is enabled.

	bool negativeEnabled = false;
	cv::Mat negativeImg; // Will be initialized when effect is enabled.

	bool sobelXEnabled = false;
	cv::Mat sxImg; // Will be initialized when effect is enabled.
	cv::Mat scaled_sxImg;

	bool sobelYEnabled = false;
	cv::Mat syImg; // Will be initialized when effect is enabled.
	cv::Mat scaled_syImg;

	bool gradientEnabled = false;
	cv::Mat gradImg; // Will be initialized when effect is enabled.

	bool blurQuantEnabled = false;
	cv::Mat blurQuantImg; // Will be initialized when effect is enabled.

	bool redEnabled = false;
	cv::Mat redOnlyImg; // Will be initialized when effect is enabled.

	bool blueEnabled = false;
	cv::Mat blueOnlyImg; // Will be initialized when effect is enabled.

	bool greenEnabled = false;
	cv::Mat greenOnlyImg; // Will be initialized when effect is enabled.

	bool LoGEnabled = false;
	cv::Mat LoGImg; // Will be initialized when effect is enabled.

	while (true)
	{
		*capture >> frame; 
		//get new frame from camera, treating as stream.

		if (frame.empty()){
			printf("Frame is empty"); 
			break;
		}
		
		// Display the current stream of images in the video stream.
		displayImage(true, frame, "Color Video", false);
		if (greyEnabled){
			cv::cvtColor(frame, greyscaleImg, cv::COLOR_BGR2GRAY);
			cv::namedWindow("Greyscale Video", windowSize);
			cv::imshow("Greyscale Video", greyscaleImg);
		}
		else if (cv::getWindowProperty("Greyscale Video", cv::WND_PROP_VISIBLE) == 1.0) {
				cv::destroyWindow("Greyscale Video");
		}

		if (myGreyEnabled){
			greyscale(frame, myGreyscaleImg);
			cv::namedWindow("My Greyscale Video", windowSize);
			cv::imshow("My Greyscale Video", myGreyscaleImg);
		}
		else if (cv::getWindowProperty("My Greyscale Video", cv::WND_PROP_VISIBLE) == 1.0) {
				cv::destroyWindow("My Greyscale Video");
		}

		if (cartoonEnabled) {
			cartoon(frame, cartoonImg, 15, 15);
			cv::namedWindow("Cartoon Video", windowSize);
			cv::imshow("Cartoon Video", cartoonImg);
		}
		else if (cv::getWindowProperty("Cartoon Video", cv::WND_PROP_VISIBLE) == 1.0) {
			cv::destroyWindow("Cartoon Video");
		}

		if (blurEnabled) {
			blur5x5(frame, blurredImg);
			cv::namedWindow("Blurred Video", windowSize);
			cv::imshow("Blurred Video", blurredImg);
		}
		else if (cv::getWindowProperty("Blurred Video", cv::WND_PROP_VISIBLE) == 1.0) {
			cv::destroyWindow("Blurred Video");
		}

		if (sobelXEnabled) {
			sobelX3x3(frame, sxImg);
			cv::convertScaleAbs(sxImg, scaled_sxImg);
			cv::namedWindow("SobelX Video", windowSize);
			cv::imshow("SobelX Video", scaled_sxImg);
		}
		else if (cv::getWindowProperty("SobelX Video", cv::WND_PROP_VISIBLE) == 1.0) {
			cv::destroyWindow("SobelX Video");
		}

		if (sobelYEnabled) {
			sobelY3x3(frame, syImg);
			cv::convertScaleAbs(syImg, scaled_syImg);
			cv::namedWindow("SobelY Video", windowSize);
			cv::imshow("SobelY Video", scaled_syImg);
		}
		else if (cv::getWindowProperty("SobelY Video", cv::WND_PROP_VISIBLE) == 1.0) {
			cv::destroyWindow("SobelY Video");
		}

		if (gradientEnabled) {
			sobelX3x3(frame, sxImg);
			sobelY3x3(frame, syImg);
			magnitude(sxImg, syImg, gradImg);
			cv::namedWindow("Gradient Magnitude Video", windowSize);
			cv::imshow("Gradient Magnitude Video", gradImg);
		}
		else if (cv::getWindowProperty("Gradient Magnitude Video", cv::WND_PROP_VISIBLE) == 1.0) {
			cv::destroyWindow("Gradient Magnitude Video");
		}

		if (blurQuantEnabled) {
			blurQuantize(frame, blurQuantImg,15);
			cv::namedWindow("Blur Quantized Video", windowSize);
			cv::imshow("Blur Quantized Video", blurQuantImg);
		}
		else if (cv::getWindowProperty("Blur Quantized Video", cv::WND_PROP_VISIBLE) == 1.0) {
			cv::destroyWindow("Blur Quantized Video");
		}

		if (negativeEnabled) {
			negativeImage(frame, negativeImg);
			cv::namedWindow("Negative Video", windowSize);
			cv::imshow("Negative Video", negativeImg);
		}
		else if (cv::getWindowProperty("Negative Video", cv::WND_PROP_VISIBLE) == 1.0) {
			cv::destroyWindow("Negative Video");
		}

		if (redEnabled) {
			redOnlyImage(frame, redOnlyImg);
			cv::namedWindow("Red Channel Video", windowSize);
			cv::imshow("Red Channel Video", redOnlyImg);
		}
		else if (cv::getWindowProperty("Red Channel Video", cv::WND_PROP_VISIBLE) == 1.0) {
			cv::destroyWindow("Red Channel Video");
		}

		if (blueEnabled) {
			blueOnlyImage(frame, blueOnlyImg);
			cv::namedWindow("Blue Channel Video", windowSize);
			cv::imshow("Blue Channel Video", blueOnlyImg);
		}
		else if (cv::getWindowProperty("Blue Channel Video", cv::WND_PROP_VISIBLE) == 1.0) {
			cv::destroyWindow("Blue Channel Video");
		}

		if (greenEnabled) {
			greenOnlyImage(frame, greenOnlyImg);
			cv::namedWindow("Green Channel Video", windowSize);
			cv::imshow("Green Channel Video", greenOnlyImg);
		}
		else if (cv::getWindowProperty("Green Channel Video", cv::WND_PROP_VISIBLE) == 1.0) {
			cv::destroyWindow("Green Channel Video");
		}

		if (LoGEnabled) {
			gaussianLaplacian(frame, LoGImg);
			cv::namedWindow("LoG Filter Video", windowSize);
			cv::imshow("LoG Filter Video", LoGImg);
		}
		else if (cv::getWindowProperty("LoG Filter Video", cv::WND_PROP_VISIBLE) == 1.0) {
			cv::destroyWindow("LoG Filter Video");
		}

		//Look for the keystroke entered.
		char key = cv::waitKey(10);
		if (key == 'q' or key == 'Q')
		{
			printf("Terminating the program as 'q' is pressed.");
			break;
		}
		else if (key == 's')
		{	//Saves the image 
			saveImage(frame, "Color");
			saveImage(greyscaleImg, "Grey");
			saveImage(myGreyscaleImg, "MyGrey");
			saveImage(blurredImg, "Blur");
			saveImage(cartoonImg, "Cartoon");
			saveImage(negativeImg, "Negative");
			saveImage(scaled_sxImg, "SobelX");
			saveImage(scaled_syImg, "SobelY");
			saveImage(gradImg, "GradMagnitude");
			saveImage(blurQuantImg, "BlurQuantized");
			saveImage(redOnlyImg, "RedChannel");
			saveImage(blueOnlyImg, "BlueChannel");
			saveImage(greenOnlyImg, "GreenChannel");
			saveImage(LoGImg, "LoGFilter");
		}
		else if (key == 'g') { // Grey scale video feed is toggled.
			greyEnabled = not(greyEnabled);
			greyscaleImg = cv::Mat(frame.size(), frame.type()); // To ensure initialization.
			continue;
		}
		else if (key == 'b') { // Blur video feed is toggled.
			blurEnabled = not(blurEnabled);
			blurredImg = cv::Mat(frame.size(), frame.type()); // To ensure initialization.
		}
		else if (key == 'c') { //Cartoon video feed is toggled.
			cartoonEnabled = not(cartoonEnabled);
			cartoonImg = cv::Mat(frame.size(), frame.type()); // To ensure initialization.
			continue;
		}
		else if (key == 'h') { // My Grey scale video feed is toggled.
			myGreyEnabled = not(myGreyEnabled);
			myGreyscaleImg = cv::Mat(frame.size(), frame.type()); // To ensure initialization.
			continue;
		}
		else if( key == 'n') { // Negative video feed is toggled.
			negativeEnabled = not(negativeEnabled);
			negativeImg = cv::Mat(frame.size(), frame.type()); // To ensure initialization.
			continue;
		}
		else if (key == 'x') { // SobelX video Feed is toggled.
			sobelXEnabled = not(sobelXEnabled);
			sxImg = cv::Mat(frame.size(), CV_16SC3); // To ensure initialization.
			continue;
		}
		else if (key == 'y') { // SobelY video Feed is toggled.
			sobelYEnabled = not(sobelYEnabled);
			syImg = cv::Mat(frame.size(), CV_16SC3); // To ensure initialization.
			continue;
		}
		else if (key == 'm') { // Gradient Magnitude video Feed is toggled.
			gradientEnabled = not(gradientEnabled);
			sxImg = cv::Mat(frame.size(), CV_16SC3); // To ensure initialization.
			syImg = cv::Mat(frame.size(), CV_16SC3); // To ensure initialization.
			gradImg = cv::Mat(frame.size(), frame.type()); // To ensure initialization.
			continue;
		}
		else if (key == 'l') { // Blur Quantized video feed is toggled. 
			blurQuantEnabled = not(blurQuantEnabled);
			blurQuantImg = cv::Mat(frame.size(), frame.type()); // To ensure initialization.
			continue;
		}
		else if (key == 'R') { // Red Channel video feed is toggled.
			redEnabled = not(redEnabled);
			redOnlyImg = cv::Mat(frame.size(), frame.type()); // To ensure initialization.
			continue;
		}
		else if (key == 'G') { // Green Channel video feed is toggled.
			greenEnabled = not(greenEnabled);
			greenOnlyImg = cv::Mat(frame.size(), frame.type()); // To ensure initialization.
			continue;
		}
		else if (key == 'B') { // Blue Channel video feed is toggled.
			blueEnabled = not(blueEnabled);
			blueOnlyImg = cv::Mat(frame.size(), frame.type()); // To ensure initialization.
			continue;
		}
		else if (key == 'e') { // LoG Filter
			LoGEnabled = not(LoGEnabled);
			LoGImg = cv::Mat(frame.size(), frame.type()); // To ensure initialization.
			continue;
		}
	}

	delete capture;
	return 0;
}