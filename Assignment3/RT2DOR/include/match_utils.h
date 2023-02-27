#include <opencv2/opencv.hpp>


int computeStandardDeviations(std::vector<std::vector<float>>& data, std::vector<float>& standardDeviations);

int sumSquaredError(std::vector<float>& x, std::vector<float>& y, float& distance);

int eucledianDistance(std::vector<float>& x, std::vector<float>& y, std::vector<float>& standardDeviations, float& distance);

/*
  This function just takes in nMatches(vector<char*>) as argument
  and displays the images in this vector.
*/
void showTopMatchedImages(std::vector<char*>& nMatches);

int identifyMatches(cv::Mat& targetImage, char* featureVectorFile, char* distanceMetric, int N, std::vector<char*>& nMatches, std::vector<char*>& nLabels);

int ComputingNearestLabelUsingKNN(cv::Mat& targetImage, char* featureVectorFile, char* distanceMetric, char* Label, int K);

void placeLabel(cv::Mat& image, char* label, int fontSize = 8, int fontWeight = 3);