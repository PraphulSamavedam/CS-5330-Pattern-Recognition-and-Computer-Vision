/*
This file displays the first image passed as an argument to the application and ignores the other arguments. 

This file is written by Samavedam Manikhanta Praphul.
*/

#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

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
		printf("Image argument is missing.\nUsage: imgDisplay.exe	<image file>");
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
		printf("Error reading the exisitng file provided.\nKindly check that file \n%s\nexists.\n", filepath);
		return -500;
	}
	
	cv::namedWindow(filepath, cv::WINDOW_AUTOSIZE);
	cv::imshow(filepath, image);
	while (true)
	{	// Wait for the user to press some key
		int keyPressed = cv::waitKey(0);
		if (keyPressed == 113 or keyPressed == 81) {
			printf("Terminating the program as '%c' is pressed!\n", char(keyPressed));
			cv::destroyWindow(filepath);
			break;
		}
	}
	return 0;
}