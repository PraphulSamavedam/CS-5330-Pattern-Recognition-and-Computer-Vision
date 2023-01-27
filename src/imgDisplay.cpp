/*
This file displays the first image passed as an argument to the application and ignores the other arguments. 
This file is written by Samavedam Manikhanta Praphul.
*/

#include <cstdio> // For printing messages/errors on screen.
#include <cstring> // For easy string manipulation
#include <opencv2/opencv.hpp> // For openCV functions
#include <fstream> // For io operations like checking if file exists.
#include "..\include\imageFilters.h"; // For image effects

#include "..\include\additionalEffects.h" // For additional effects.

/* This function reads the first argument as the image file to be displayed and 
based on the key pressed performs the actions.
returns 0 for successful program completion
returns -100 if no file has been provided as argument. 
returns -404 if file doesn't exist
returns -500 if the file is in improper format/file is corrupted.
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
	cv::Mat greyScaleImg;
	cv::Mat blurredImg;
	
	printf("Displaying image in file %s", filepath);
	std::string windowName = "Display: Color Image";
	bool quit = false;
	while (true)
	{	
		if (!greenOnlyImg.empty())
		{
			cv::imshow("Green Channel", greenOnlyImg);
		}

		if (!blueOnlyImg.empty())
		{
			cv::imshow("Blue Channel", blueOnlyImg);
		}

		if (!redOnlyImg.empty())
		{
			cv::imshow("Red Channel", redOnlyImg);
		}

		if (!greyScaleImg.empty())
		{
			cv::imshow("GreyScale Channel", greyScaleImg);
		}

		if (!blurredImg.empty())
		{
			cv::imshow("Blurred Image", blurredImg);
		}

		cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
		cv::imshow(windowName, image);
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
		case 'B': // Blue channel
			image.copyTo(blueOnlyImg);
			blueOnlyImage(image, blueOnlyImg);
			break;
		case 'b': // Blur the image
			image.copyTo(blurredImg);
			blur5x5(image, blurredImg);
			break;
		case 'G': // Green Channel
			image.copyTo(greenOnlyImg);
			greenOnlyImage(image, greenOnlyImg);
			break;
		case 'g': // Image grey scale
			image.copyTo(blurredImg);
			grey(image, blurredImg);
			break;
		case 'R': // Red channel
			image.copyTo(redOnlyImg);
			redOnlyImage(image, redOnlyImg);
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