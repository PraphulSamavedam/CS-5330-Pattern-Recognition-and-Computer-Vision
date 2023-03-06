/*
* Written by :Poorna Chandra Vemula
            Samavedam Manikhanta Praphul
* This file has the utility function required to match target iamge with the images in database.
*/
#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/opencv.hpp>
#include "../include/utils.h"


int computeStandardDeviations(std::vector<std::vector<float>>& data, std::vector<float>& standardDeviations) {

	float stdDev = 0.0;
	float sum = 0.0;
	float mean = 0.0;
	float variance = 0.0;
	int n = data.size();
	bool debug = false;

	for (int j = 0; j < data[0].size(); j++) {
		if (debug) { std::cout << "For feature vector : " << j << std::endl; }
		sum = 0.0;
		mean = 0.0;
		if (debug) {
			std::cout << "Moments: " << std::endl;
		}
		for (int i = 0; i < data.size(); i++) {
			sum += data[i][j];
			if (debug)
			{
				std::cout << " " << data[i][j];
			}
		}
		if (debug) {
			std::cout << std::endl;
		}

		mean = sum / n;
		if (debug) {
			std::cout << "mean : " << mean << std::endl;
		}

		sum = 0.0;
		for (int i = 0; i < data.size(); i++) {
			sum += ((data[i][j] - mean) * (data[i][j] - mean));
		}

		variance = sum / n;
		if (debug) {
			std::cout << "variance : " << variance << std::endl;
		}
		stdDev = sqrt(variance);
		standardDeviations.push_back(stdDev);
	}

	return 0;
}

int sumSquaredError(std::vector<float>& x, std::vector<float>& y, float& distance) {
	distance = 0;

	for (int i = 0; i < x.size(); i++) {
		distance += ((x[i] - y[i]) * (x[i] - y[i]));
		std::cout << "x_i: " << x[i] << "y_i: " << y[i] << std::endl;
	}
	std::cout << "distance : " << distance << std::endl;

	return 0;
}

int eucledianDistance(std::vector<float>& x, std::vector<float>& y, std::vector<float>& standardDeviations, float& distance) {
	distance = 0;

	for (int i = 0; i < x.size(); i++) {
		distance += abs((x[i] - y[i]) / standardDeviations[i]);

	}

	//std::cout << "distance : " << distance << std::endl;
	//sumSquaredError(x, y, distance);

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
		if (std::get<2>(first) > std::get<2>(second)) {
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


int identifyMatches(cv::Mat& targetImage, char* featureVectorFile, char* distanceMetric, int N, std::vector<char*>& nMatches, std::vector<char*>& nLabels) {

	bool debug = false;

	std::priority_queue<std::tuple<char*, char*, float>, std::vector<std::tuple<char*, char*, float>>, Compare> pq;


	//conditional feature computing based on various feature sets
	std::vector<float> targetFeatureVector;

	getFeaturesForImage(targetImage, targetFeatureVector);

	if (debug) {
		std::cout << "Target feature vector is:\n";
		for (float featureValue : targetFeatureVector)
		{
			std::cout << featureValue << " ";
		}
		std::cout << std::endl;
	}

	std::vector<char*> filenames;
	std::vector<char*> labels;
	std::vector<std::vector<float>> data;

	int i = read_image_data_csv(featureVectorFile, filenames, labels, data, false);


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


		if (distance < minDistance) {
			minDistance = distance;
			strcpy(topMatch, filenames[datapoint]);
		}
		if (debug) {
			printf("Filename: %s ", filenames[datapoint]);
			printf("Label: %s ", labels[datapoint]);
			printf("Distance: %.04f \n", distance);
		}
		pq.push(std::make_tuple(filenames[datapoint], labels[datapoint], distance));
	}
	std::cout << "Top N images are:" << std::endl;
	while (N-- && !pq.empty()) {

		nMatches.push_back(std::get<0>(pq.top()));
		nLabels.push_back(std::get<1>(pq.top()));
		if (debug) { std::cout << "filename: " << std::get<0>(pq.top()) << " labels: " << std::get<1>(pq.top()) << ", distance from target: " << std::get<2>(pq.top()) << std::endl; }

		pq.pop();
	}




	return(0);

}


int identifyMatches(std::vector<float>& targetFeatureVector, char* featureVectorFile, char* distanceMetric, int N, std::vector<char*>& nMatches, std::vector<char*>& nLabels) {

	bool debug = false;

	std::priority_queue<std::tuple<char*, char*, float>, std::vector<std::tuple<char*, char*, float>>, Compare> pq;

	std::vector<char*> filenames;
	std::vector<char*> labels;
	std::vector<std::vector<float>> data;

	int i = read_image_data_csv(featureVectorFile, filenames, labels, data, false);


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


		if (distance < minDistance) {
			minDistance = distance;
			strcpy(topMatch, filenames[datapoint]);
		}
		if (debug) {
			printf("Filename: %s ", filenames[datapoint]);
			printf("Label: %s ", labels[datapoint]);
			printf("Distance: %.04f \n", distance);
		}
		pq.push(std::make_tuple(filenames[datapoint], labels[datapoint], distance));
	}
	std::cout << "Top N images are:" << std::endl;
	while (N-- && !pq.empty()) {

		nMatches.push_back(std::get<0>(pq.top()));
		nLabels.push_back(std::get<1>(pq.top()));
		if (debug) { std::cout << "filename: " << std::get<0>(pq.top()) << " labels: " << std::get<1>(pq.top()) << ", distance from target: " << std::get<2>(pq.top()) << std::endl; }

		pq.pop();
	}

	return(0);

}


int identifyMatches(cv::Mat& targetImage, std::vector<std::vector<float>> data,  std::vector<char*> filenames, std::vector<char*> labels, char* distanceMetric, int N, std::vector<char*>& nMatches, std::vector<char*>& nLabels) {

    bool debug = false;

    std::priority_queue<std::tuple<char*, char*, float>, std::vector<std::tuple<char*, char*, float>>, Compare> pq;


    //conditional feature computing based on various feature sets
    std::vector<float> targetFeatureVector;

    getFeaturesForImage(targetImage, targetFeatureVector, 124, 4, 4, 8, 1, false, true);

    if (debug) {
        std::cout << "Target feature vector is:\n";
        for (float featureValue : targetFeatureVector)
        {
            std::cout << featureValue << " ";
        }
        std::cout << std::endl;
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


        if (distance < minDistance) {
            minDistance = distance;
            strcpy(topMatch, filenames[datapoint]);
        }
        if (debug) {
            printf("Filename: %s ", filenames[datapoint]);
            printf("Label: %s ", labels[datapoint]);
            printf("Distance: %.04f \n", distance);
        }
        pq.push(std::make_tuple(filenames[datapoint], labels[datapoint], distance));
    }
	//    std::cout << "Top N images are:" << std::endl;
    while (N-- && !pq.empty()) {

        nMatches.push_back(std::get<0>(pq.top()));
        nLabels.push_back(std::get<1>(pq.top()));
        if (debug) { std::cout << "filename: " << std::get<0>(pq.top()) << " labels: " << std::get<1>(pq.top()) << ", distance from target: " << std::get<2>(pq.top()) << std::endl; }

        pq.pop();
    }




    return(0);
}


int ComputingNearestLabelUsingKNN(cv::Mat& targetImage, char* featureVectorFile, char* distanceMetric, char* Label, int K) {

	std::vector<char*> kMatches;
	std::vector<char*> kLabels;

	std::unordered_map<std::string, int> mp;
	bool debug = false;

	identifyMatches(targetImage, featureVectorFile, distanceMetric, K, kMatches, kLabels);


	/*showTopMatchedImages(kMatches);*/

	for (char* label : kLabels) {
		std::cout << label << std::endl;
		if (mp.find(label) != mp.end()) {
			mp[label] += 1;
		}
		else {
			mp[label] = 1;
		}
	}

	int maxLabelCount = -INT_MAX;
	char maxLabel[256] = "";

	for (auto kv_pair : mp) {
		if (debug) { std::cout << "Label: " << kv_pair.first << " Count: " << kv_pair.second << std::endl; }
		if (maxLabelCount <= kv_pair.second) {
			maxLabelCount = kv_pair.second;
			strcpy(maxLabel, (char*)kv_pair.first.c_str());
		}
	}

	strcpy(Label, maxLabel);

	return 0;
}



int ComputingNearestLabelUsingKNN(cv::Mat& targetImage, std::vector<std::vector<float>> data,  std::vector<char*> filenames, std::vector<char*> labels, char* featureVectorFile, char* distanceMetric, char* Label, int K) {

    std::vector<char*> kMatches;
    std::vector<char*> kLabels;

    std::unordered_map<std::string, int> mp;
    bool debug = false;

    identifyMatches(targetImage,  data,   filenames, labels,  distanceMetric, K, kMatches, kLabels);


    /*showTopMatchedImages(kMatches);*/

    for (char* label : kLabels) {
//        std::cout << label << std::endl;
        if (mp.find(label) != mp.end()) {
            mp[label] += 1;
        }
        else {
            mp[label] = 1;
        }
    }

    int maxLabelCount = -INT_MAX;
    char maxLabel[256] = "";

    for (auto kv_pair : mp) {
        if (debug) { std::cout << "Label: " << kv_pair.first << " Count: " << kv_pair.second << std::endl; }
        if (maxLabelCount <= kv_pair.second) {
            maxLabelCount = kv_pair.second;
            strcpy(maxLabel, (char*)kv_pair.first.c_str());
        }
    }

    strcpy(Label, maxLabel);

    return 0;
}



int ComputingNearestLabelUsingKNN(std::vector<float>& targetFeatureVector, char* featureVectorFile, char* distanceMetric, char* Label, int K) {

	std::vector<char*> kMatches;
	std::vector<char*> kLabels;

	std::unordered_map<std::string, int> mp;
	bool debug = false;

	identifyMatches(targetFeatureVector, featureVectorFile, distanceMetric, K, kMatches, kLabels);

	/*showTopMatchedImages(kMatches);*/

	for (char* label : kLabels) {
		std::cout << label << std::endl;
		if (mp.find(label) != mp.end()) {
			mp[label] += 1;
		}
		else {
			mp[label] = 1;
		}
	}

	int maxLabelCount = -INT_MAX;
	char maxLabel[256] = "";

	for (auto kv_pair : mp) {
		if (debug) { std::cout << "Label: " << kv_pair.first << " Count: " << kv_pair.second << std::endl; }
		if (maxLabelCount <= kv_pair.second) {
			maxLabelCount = kv_pair.second;
			strcpy(maxLabel, (char*)kv_pair.first.c_str());
		}
	}

	strcpy(Label, maxLabel);

	return 0;
}


void placeLabel(cv::Mat& image, char* label, int font_size, int font_weight) {
	cv::Point text_position(image.cols / 2, image.rows / 2);// Declaring the text position at the centre//
	cv::Scalar font_Color(0, 255, 255);// Declaring the color of the font//
	putText(image, label, text_position, cv::FONT_HERSHEY_COMPLEX, font_size, font_Color, font_weight);
}
