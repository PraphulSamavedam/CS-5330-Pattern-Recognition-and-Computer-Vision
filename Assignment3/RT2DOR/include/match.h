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
#include "../include/utils.h"


void showTopMatchedImages(std::vector<char*>& nMatches);

int identifyMatches(cv::Mat& targetImage, char* featureVectorFile, char* distanceMetric, int N, std::vector<char*>& nMatches, std::vector<char*>& nLabels);

int ComputingNearestLabelUsingKNN(cv::Mat& targetImage, char* featureVectorFile, char* distanceMetric, char* Label, int K);