/**
This file displays the first image passed as an argument to the application 
and ignores the other arguments. 
By pressing the keys, several effects are applied on the file read, 'q'/'Q' quits program.

Written by: Samavedam Manikhanta Praphul.
*/

#include <cstdio> // For printing messages/errors on screen.
#include <cstring> // For easy string manipulation
#include <opencv2/opencv.hpp> // For openCV functions
#include <fstream> // For io operations like checking if file exists.
#include "..\include\filters.h" // For image effects which are done as part of tasks in the filtering assignment.

#include "..\include\additionalEffects.h" // For additional effects.


/** This function simply saves the image with filename passed if the image is non-empty. 
@param src address of the Mat which needs to be saved.
@param fileName name of the file relative the current path.
@return 0 by default.
*/
int saveImage(cv::Mat& src,std::string fileName) {
	if (!src.empty())
	{
		cv::imwrite(fileName, src);
	}
	return 0;
}

/** This function simply displays the image in the window with name passed if the image is non-empty.
@param src address of the Mat which needs to be displayed.
@param windowname name of the window in which the image needs to be displayed.
@return 0 by default.
*/
int displayImage(cv::Mat& src, std::string windowName) {
	if (!src.empty())
	{
		cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
		cv::imshow(windowName, src);
	}
	return 0;
}

/** This function reads the first argument as the image file to be displayed and 
* based on the key pressed performs the actions.
* @return	0 for successful program completion
*		 -100 if no file has been provided as argument. 
*		 -404 if file doesn't exist.
*		 -500 if the file is in improper format/file is corrupted.
* @note Press the keys to view several effects and save them.
*	Press 'R' for Red Channel version of the image. 
*	Press 'G' for Green Channel version of the image. 
*   Press 'B' for Blue Channel version of the image.
*   Press 'b' for gaussian blur version of the image.
*   Press 'n' for negative of the image.
*   Press 'g' for greyscale of the image.
*   Press 's' to save the opened filters into the output folder.
*   Press 'q' to quit the program.
*/
int main(int argc, char const *argv[])
{
	/*End the program if no image file has been passed.*/
	if(argc <2){
		printf("Image argument is missing.\nUsage: imgDisplay.exe <image file>");
		return -100;
	}
	/*Store the file passed as a variable for quick load.*/
	char filepath[256];
	strcpy_s(filepath, argv[1]);

	/* Check if the file exist. */
	std::ifstream filestream;
	filestream.open(filepath);
	if (!filestream)	{
		printf("File does not exist");
		return -404;
	}

	/*Check if existing file is valid to load into memory.*/
	cv::Mat image = cv::imread(filepath);
	if (image.data == NULL)
	{	// Error to read the file, either file is corrupted or file is in unsupported format.
		printf("Error reading the exisitng file provided.\nKindly check that file \n%s\n", filepath);
		printf("if it is corrupted or is in unsupported format by openCV.\n");
		return -500;
	}

	cv::Mat greenOnlyImg;
	cv::Mat blueOnlyImg;
	cv::Mat redOnlyImg;
	cv::Mat negativeImg;
	cv::Mat blurredImg;
	cv::Mat greyImg;
	
	printf("Displaying image in file %s", filepath);
	bool quit = false;

	while (true)
	{	
		displayImage(image, "Original Image");
		displayImage(blurredImg, "Blurred Image");
		displayImage(blueOnlyImg, "Blue Channel");
		displayImage(greenOnlyImg, "Green Channel");
		displayImage(redOnlyImg, "Red Channel");
		displayImage(negativeImg, "Negative Image");
		displayImage(greyImg, "Greyscale Image");

		// Wait for the user to press some key
		char keyPressed = cv::waitKey(0);
		switch (keyPressed)
		{
		case 'q': // Quit the program
			printf("Terminating the program as 'q' is pressed!\n");
			cv::destroyAllWindows();
			quit = true;
			break;
		case 'Q': // Quit the program
			printf("Terminating the program as 'Q' is pressed!\n");
			cv::destroyAllWindows();
			quit = true;
			break;
		case 'b': // Blur the image
			image.copyTo(blurredImg);
			blur5x5(image, blurredImg);
			break;
		case 'B': // Blue channel
			image.copyTo(blueOnlyImg);
			blueOnlyImage(image, blueOnlyImg);
			break;
		case 'G': // Green Channel
			image.copyTo(greenOnlyImg);
			greenOnlyImage(image, greenOnlyImg);
			break;
		case 'R': // Red channel
			image.copyTo(redOnlyImg);
			redOnlyImage(image, redOnlyImg);
			break;
		case 'n': // Negative Image
			image.copyTo(negativeImg);
			negativeImage(image, negativeImg);
			break;
		case 'g': // Greyscale Image
			image.copyTo(greyImg);
			greyScaleImpl(image, greyImg);
			break;
		case 's': // Save images
			saveImage(image, "output/STILL_IMG.png");
			saveImage(blurredImg, "output/STILL_IMG_BLUR.png");
			saveImage(blueOnlyImg, "output/STILL_IMG_BLUE.png");
			saveImage(greenOnlyImg, "output/STILL_IMG_GREEN.png");
			saveImage(redOnlyImg, "output/STILL_IMG_RED.png");
			saveImage(negativeImg, "output/STILL_IMG_NEG.png");
			printf("\nSuccessfully saved the current frames\n***********************************\n");
			break;
		default:
			continue;
		}
		if (quit)
		{
			break;
		}
	}
	return 0;
}