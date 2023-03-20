/**
* Written by:   Poorna Chandra Vemula
*               Samavedam Manikhanta Praphul
* Version : 1.0
* This file starts a video stream and is used to calibrate the
* camera and store the camera intrinsic params into a csv file.
*/

#define _CRT_SECURE_NO_WARNINGS

#include <opencv2/opencv.hpp>
#include <vector>
#include "../include/tasks.h"
#include "../include/csv_util.h"


int main(int argc, char *argv[]) {

    char cameraParametersFile[32] = "./resources/cameraParams.csv";

    cv::VideoCapture* capture = new cv::VideoCapture(0);
    // Check if any video capture device is present.
    if (!capture->isOpened())
    {
        printf("Unable to open the primary video device.\n");
        return(-404);
    }

    cv::Size refs((int)capture->get(cv::CAP_PROP_FRAME_WIDTH),
        capture->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Camera Capture size: %d x %d \n.", refs.width, refs.height);
    
    cv::Mat frame;
    std::vector<std::vector<cv::Point2f>> corners_list;
    std::vector<cv::Point2f> last_corners_set;
    cv::Mat last_image;

    std::vector<cv::Vec3f> points_set;
    std::vector<cv::Vec3f> last_points_set;
    std::vector<std::vector<cv::Vec3f>> points_list;
    

    bool last_successful_capture = false;
    while (true) {
        *capture >> frame;
        //get new frame from camera, treating as stream.
        if (frame.empty()) {
            printf("Frame is empty");
            break;
        }
        std::vector<cv::Point2f> corners_set;
        bool status = detectAndExtractChessBoardCorners(frame, corners_set);
        // printf("Status of chess board capture in video loop: %d\n", status);
        cv::namedWindow("Image", cv::WINDOW_GUI_EXPANDED);
        cv::imshow("Image", frame);
        char key = cv::waitKey(10);
        if (key == 'q')
        {
            cv::destroyAllWindows();
            for (auto corner : corners_list)
            {
                std::cout << corner << std::endl;
            }
            printf("\nPrinting points. \n");
            for (auto point_set : points_list)
            {
                for (auto point : point_set) {
                    std::cout << point << " ";
                }
                std::cout << std::endl;
            }
            break;
        }
        else if (key == 's') {
            //printf("Waiting for the point set calculation.\n");
            if (status) {
                corners_list.push_back(corners_set);
                buildPointsSet(corners_set, points_set);
                points_list.push_back(points_set);
                
                // Mark that there was a successful capture
                last_successful_capture = true;
                last_corners_set = corners_set;
                buildPointsSet(last_corners_set, last_points_set);
                frame.copyTo(last_image); // Save the last image
            }
            else {
                if (last_successful_capture)
                {
                    corners_list.push_back(last_corners_set);
                    points_list.push_back(last_points_set);
                }
                else
                {
                    printf("No previous successful capture found.");
                    continue;
                }
            }
        }
        else if (key == 'c')
        {

            // Image Size is refs
            cv::Mat cameraMatrix = cv::Mat::zeros(3, 3, CV_64FC1);
            cameraMatrix.at<float>(0,0) = 1;
            cameraMatrix.at<float>(1, 1) = 1;
            cameraMatrix.at<float>(2, 2) = 1;
            cameraMatrix.at<float>(0, 2) = frame.cols / 2;
            cameraMatrix.at<float>(1, 2) = frame.rows/2;
            std::vector<float> distortionCoefficients;
            for (int i = 0; i < 8; i++)
            {
                distortionCoefficients.push_back(0.0);
            }
            std::vector<cv::Mat> rVecs;
            std::vector<cv::Mat> tVecs;

            printf("Pre Camera Matrix\n");
            for (int i=0;i<3;i++ )
            {
                for (int j = 0; j < 3; j++)
                {
                    printf("[%d,%d]: %.02f", i, j, cameraMatrix.at<double>(i, j));
                }
                printf("\n");
            }

            printf("Pre distortion coefficients are :\n");
            for (auto dist : distortionCoefficients)
            {
                std::cout << dist << std::endl;
            }
            
            printf("Using %zd points for calibration.\n", corners_list.size());
            double reprojection_error = cv::calibrateCamera(points_list, corners_list, refs, cameraMatrix,
                distortionCoefficients, rVecs, tVecs, cv::CALIB_FIX_ASPECT_RATIO);
            printf("Calibrated the camera.\n");
            printf("Updated distortion coefficients are :\n");
    
            //9 element camera matrix vector
            std::vector<float> cameraMatrixVector;
            
            printf("Updated Camera Matrix\n");
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    printf("[%d,%d]: %.06f ", i, j, cameraMatrix.at<double>(i, j));
                    cameraMatrixVector.push_back(cameraMatrix.at<double>(i, j));
                }
                printf("\n");
            }
            
            //append three zeroes for distortion coeff for consistency
           /* for(int i=0;i<4;i++){
                distortionCoefficients.push_back(0);
            }*/
            char metricName[32] = "cameraMatrix";
            
            //append camera matrix
            append_metric_data_csv(cameraParametersFile, metricName, cameraMatrixVector, true);
            
            strcpy(metricName, "distCoeff");
            //append dist coefficients
            append_metric_data_csv(cameraParametersFile, metricName, distortionCoefficients, false);

            //append the reprojection error
            strcpy(metricName, "reprojectionError");
            std::vector<float> reprojection;
            reprojection.push_back(reprojection_error);
            append_metric_data_csv(cameraParametersFile, metricName, reprojection, false);
            printf("Reprojection Error: %.06f\n", reprojection_error);
        }
    }
    return 0;
}
