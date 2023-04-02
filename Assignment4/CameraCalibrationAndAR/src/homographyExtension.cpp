#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace aruco;

int main(int argc, char** argv)
{

    // Load the captured image and define the marker size
    Mat capturedImage = imread("capturedImage.jpg");
    float markerSize = 0.1; // in meters

    int window_size = cv::WINDOW_GUI_NORMAL;

    if (capturedImage.data == NULL)
    {
        printf("Error reading the %s file", "catpuredImage.jpg");
        exit(-100);
    }

    // Define the dictionary and parameters for Aruco marker detection
    //aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_100);

    aruco::DetectorParameters parameters;

    // Set the detection parameters
    parameters.adaptiveThreshConstant = 10;
    parameters.adaptiveThreshWinSizeMax = 23;
    parameters.adaptiveThreshWinSizeMin = 3;
    parameters.minCornerDistanceRate = 0.05;
    parameters.minDistanceToBorder = 3;
    parameters.minMarkerDistanceRate = 0.05;
    parameters.polygonalApproxAccuracyRate = 0.05;
    parameters.maxErroneousBitsInBorderRate = 0.04;
    parameters.maxMarkerPerimeterRate = 4.0;
    parameters.minOtsuStdDev = 10.0;
    parameters.errorCorrectionRate = 0.6;

    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    // Detect the Aruco markers in the captured image
    vector<int> markerIds;
    vector<vector<Point2f>> markerCorners, rejectedCorners;
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);
    detector.detectMarkers(capturedImage, markerCorners, markerIds, rejectedCorners);

    // If at least two markers are detected, continue
    if (markerIds.size() >= 2)
    {
        // Draw the detected markers on the captured image
        aruco::drawDetectedMarkers(capturedImage, markerCorners, markerIds);

        // Select the first two markers and their corners
        int marker1 = 0;
        int marker2 = 1;
        vector<Point2f> corners1 = markerCorners[marker1];
        vector<Point2f> corners2 = markerCorners[marker2];

        // Find the homography matrix to warp the new scene image
        Mat homography = findHomography(corners1, corners2);

        // Load the new scene image and resize it to the same size as the captured image
        Mat newSceneImage = imread("newScene.jpg");
        resize(newSceneImage, newSceneImage, capturedImage.size());

        // Warp the new scene image using the homography matrix
        Mat warpedImage;
        warpPerspective(newSceneImage, warpedImage, homography, capturedImage.size());

        // Show the captured image with the detected markers and the warped new scene image
        Mat resultImage = capturedImage.clone();
        warpedImage.copyTo(resultImage(Rect(0, 0, warpedImage.cols, warpedImage.rows)));
        cv::namedWindow("Result Image", window_size);
        imshow("Result Image", resultImage);
        cv::namedWindow("Original Image", window_size);
        imshow("Original Image", capturedImage);
        cv::namedWindow("Project Image", window_size);
        imshow("Project Image", newSceneImage);
        waitKey(0);
        return 0;
    }
    else
    {
        cout << "Error: At least two Aruco markers are required in the captured image" << endl;
        return -1;
    }
}