/**
* Written by: Samavedam Manikhanta Praphul
*                   Poorna Chandra Vemula
* This file provides the signatures of several functions required in the project.
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


/**This function shows top matched images
* @param nMatches vector of top 'n' matched filenames
*/
void showTopMatchedImages(std::vector<char*>& nMatches);

/**This function identfies the top 'n' matches for a target image
* @param targetImage
* @param featureVectorFile
* @param distanceMetric
* @param nMatches vector of matched filenames to be updated
* @param nLabels vector of matched labels to be updated
* @returns 0 if the feature is properly extracted.
*        non zero if the operation is failure.
*/
int identifyMatches(cv::Mat& targetImage, char* featureVectorFile, char* distanceMetric, int N, std::vector<char*>& nMatches, std::vector<char*>& nLabels);

/**This function predicts the label using KNN
* @param targetImage
* @param featureVectorFile
* @param distanceMetric
* @param Label char* to be updated
* @param K parameter for KNN
* @returns 0 if the feature is properly extracted.
*        non zero if the operation is failure.
*/
int ComputingNearestLabelUsingKNN(cv::Mat& targetImage, char* featureVectorFile, char* distanceMetric, char* Label, int K);
