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


int computeStandardDeviations(std::vector<std::vector<float>>& data, std::vector<float>& standardDeviations) {

	float stdDev = 0.0;
	float sum = 0.0;
	float mean = 0.0;
	float variance = 0.0;
	int n = data.size();

	for (int j = 0; j < data[0].size(); j++) {
		std::cout << "For feature vector : " << j << std::endl;
		sum = 0.0;
		mean = 0.0;
		for (int i = 0; i < data.size(); i++) {
			sum += data[i][j];
		}

		mean = sum / n;
		std::cout << "mean : " << mean << std::endl;

		sum = 0.0;
		for (int i = 0; i < data.size(); i++) {
			sum += ((data[i][j] - mean) * (data[i][j] - mean));
		}

		variance = sum / n;
		std::cout << "variance : " << variance << std::endl;
		stdDev = sqrt(variance);
		standardDeviations.push_back(stdDev);
	}

	return 0;
}



int eucledianDistance(std::vector<float>& x, std::vector<float>& y, std::vector<float>& standardDeviations, float& distance) {
	distance = 0;

	for (int i = 0; i < x.size(); i++) {
		std::cout << "std : " << standardDeviations[i] << std::endl;
		distance += abs((x[i] - y[i]) / standardDeviations[i]);
		
	}

	std::cout << "distance : " << distance << std::endl;

	return 0;
}



/*
   This class implements comparator for the priority queue.
   - priority queue is built using the second element in the pair
*/
class Compare {
public:
	bool operator()(std::tuple<char*, char*, float> first, std::tuple<char*, char*, float> second)
	{
		if (std::get<2>(first) < std::get<2>(second)) {
			return true;
		}
		else {
			return false;
		}


	}
};

/*
  This function just takes in nMatches(vector<char*>) as argument
  and displays the images in this vector.
*/
void showTopMatchedImages(std::vector<char*>& nMatches) {

	for (auto fileName : nMatches) {
		cv::Mat showImage = cv::imread(fileName);
		cv::imshow(fileName, showImage);
	}

	int key = cv::waitKey(0);

	cv::destroyAllWindows();
}


int identifyMatches(char* targetImage, char* featureVectorFile, char* distanceMetric, int N, std::vector<char*>& nMatches) {



	std::priority_queue<std::tuple<char*, char*, float>, std::vector<std::tuple<char*, char*, float>>, Compare> pq;


	//conditional feature computing based on various feature sets
	std::vector<float> targetFeatureVector;

	getFeaturesForImage(targetImage, targetFeatureVector);

	std::vector<char*> filenames;
	std::vector<char*> labels;
	std::vector<std::vector<float>> data;

	int i = read_image_data_csv(featureVectorFile, filenames, labels, data, 0);


	if (i != 0) {
		std::cout << "file read unsuccessful" << std::endl;
		exit(-1);
	}


	float minDistance = INT_MAX;
	char topMatch[100] = "filename";

	std::vector<float> stdDeviations;
	computeStandardDeviations(data, stdDeviations);

	//calculating distances
	for (int datapoint = 0; datapoint < data.size(); datapoint++) {
		//change distance based on the distance metric being used
		float distance = 0.0;


		
		eucledianDistance(data[datapoint], targetFeatureVector, stdDeviations, distance);
		

		if (distance < minDistance && strcmp(filenames[datapoint], targetImage) != 0) {
			minDistance = distance;
			strcpy(topMatch, filenames[datapoint]);
		}
		printf("Filename: %s ",filenames[datapoint]);
		printf("Label: %s ", labels[datapoint]);
		printf("Distance: %.04f \n", distance);
		pq.push(std::make_tuple(filenames[datapoint], labels[datapoint], distance ));
	}

	while (N-- && !pq.empty()) {

		nMatches.push_back(std::get<0>(pq.top()));
		std::cout << "filename: " << std::get<0>(pq.top()) << " labels: " << std::get<1>(pq.top()) << ", distance from target: " << std::get<2>(pq.top()) << std::endl;

		pq.pop();
	}




	return(0);

}


/*
 This program takes in TargetImage, distanceMetric and Number of Matches and produces top 'N' Matches.
  Written by Poorna Chandra Vemula.

  It also implements a GUI with buttons using cvui

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
	int N = atoi(argv[3]);
	char featureVectorFile[256] = "../data/db/features.csv";
	/*if (argc >=4)
	{
		std::cout << argv[4] << std::endl;
		strcpy(featureVectorFile, argv[4]);
	}*/
	std::vector<char*> nMatches;

	identifyMatches(targetImage, featureVectorFile, distanceMetric, N, nMatches);

	showTopMatchedImages(nMatches);


	return 0;
}

