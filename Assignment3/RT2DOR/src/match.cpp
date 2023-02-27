/*
	Poorna Chandra Vemula.
	CS 5330, Spring 2023
	RT2DOR, match.cpp
*/


#define _CRT_SECURE_NO_WARNINGS
/*
 including required headers
*/
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <dirent.h>
#include "../include/csv_util.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <queue>
#include "../include/match_utils.h"




/*
 This program takes in TargetImage, distanceMetric and Number of Matches and produces top 'N' Matches.
  Written by Poorna Chandra Vemula.

  It also implements a GUI with buttons using cvui

  Eg: "../data/test_images/beanie1.jpg" "euclidean" "5"

  @return  0 if program terminated with success.
		 -1 if invalid arguments or file not found.
*/
int main(int argc, char* argv[]) {

	//take target image, distance metric, feature set as arguments
	if (argc < 4) {
		std::cout << "pass valid arguments <./matchTarget <TargetImage> <distanceMetric> <TopNMatches> <csvFilePath[default='../data/db/features.csv' " << std::endl;
		exit(-1);
	}

	//pass arguments to variables
	char targetImage[256];
	strcpy(targetImage, argv[1]);
	char distanceMetric[256];
	strcpy(distanceMetric, argv[2]);
	int topNMatches = atoi(argv[3]);
	char featureVectorFile[256] = "../data/db/features.csv";

	std::vector<char*> nMatches;
	cv::Mat targetImg = cv::imread(targetImage);
	if (targetImg.data == NULL)
	{
		printf("Error reading the target image %s", targetImage);
		exit(-404);
	}

	char prediction[256];
	ComputingNearestLabelUsingKNN(targetImg, featureVectorFile, distanceMetric, prediction, topNMatches);
	std::cout << "Prediction:" << prediction << std::endl;

	//std::vector<char*> nLabels;
	//identifyMatches(targetImg, featureVectorFile, distanceMetric, N, nMatches,nLabels);

	placeLabel(targetImg, prediction);
	//showTopMatchedImages(nMatches);
	cv::namedWindow("Predicted Label", cv::WINDOW_GUI_NORMAL);
	cv::imshow("Predicted Label",targetImg);
	cv::imwrite("Image with Prediction.jpg", targetImg);
	cv::waitKey(0);
	return 0;
}

