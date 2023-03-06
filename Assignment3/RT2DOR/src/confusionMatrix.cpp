/*
* Written by : Samavedam Manikhanta Praphul
*              Poorna Chandra Vemula
* This file generates the confusion matrix for the model. 
*/

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include "../include/tasks.h" // For generating the predictions.


using namespace std;

/** This function requires 2 inputs 
* 1. featuresFile having feature vectors for the all images.
* 2. Confusionmatrix file to which the output needs to be written.
* 
* Eg. ConfusionMatrixGenerator.exe "../data/db/features.csv" "../data/db/confusionMatrix_allData.csv"
* 
* @note: This function assumes features vector is pre-populated. 
*/
int main(int argc, char* argv[])
{
	if (argc <2)
	{
		printf("Invalid arguments\nUsage: %s <featuresFileName> <confusionMatrixFile>", argv[0]);
		exit(-100);
	}

	char featuresAndLabelsFile[256];
	strcpy(featuresAndLabelsFile, argv[1]);

	char confusionMatrixFile[256];
	strcpy(confusionMatrixFile, argv[2]);

	vector<char*> predictedLabels;
	std::vector<char*> labelnames;
	char distanceMetric[32] = "euclidean";
	generatePredictions(featuresAndLabelsFile, predictedLabels, labelnames, 1);
	
	confusionMatrixCSV(featuresAndLabelsFile, confusionMatrixFile, labelnames, predictedLabels);
}
