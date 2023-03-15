/** Written by: Samavedam Manikhanta Praphul
*                Poorna Chandra Vemula
* This functions works on a calibrated camera to start augmented reality.
*
*/

#define _CRT_SECURE_NO_WARNINGS // To supress strcpy warnings


#include <opencv2/opencv.hpp> // Required for openCV functions.
#include "../include/csv_util.h" // Reading the csv file containing the camera intrinsic parameters
#include "../include/tasks.h" // For detectAndExtractChessBoardCorners function

int main(int argc, char* argv[]) {

    // Configurable variables
    char paramsFile[32];
    bool debug = false;
    char metric_name_0[13] = "cameraMatrix";
    char metric_name_1[10] = "distCoeff";

    /*assert(argc > 1);
    strcpy(paramsFile, argv[1]);*/
    strcpy(paramsFile, "resources/cameraParams.csv");

    std::vector<char*> metricNames;
    std::vector<std::vector<float>> data;
    int status = read_metric_data_csv(paramsFile, metricNames, data, true);

    assert(status == 0);
    printf("Data is read to have length: %zd \n", data.size());

    // Error check for the metric order.
    assert(strcmp(metric_name_0, metricNames[0]) == 0);
    assert(strcmp(metric_name_1, metricNames[1]) == 0);

    int metric_values_length = data[0].size();
    cv::Mat cameraMatrix = cv::Mat::zeros(3, 3, CV_64FC1);
    for (int index = 0; index < metric_values_length; index++) {
        cameraMatrix.at<double>(index / 3, index % 3) = data[0][index];
    }

    if (debug) {
        printf("Camera Matrix is read as follows: \n");
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                printf("%0.4f ", cameraMatrix.at<double>(i, j));
            }
            std::cout << std::endl;
        }
    }

    std::vector<float> distortionCoefficients;
    metric_values_length = data[1].size();
    for (int index = 0; index < metric_values_length; index++) {
        distortionCoefficients.push_back(data[1][index]);
    }

    if (debug) {
        printf("Distortion coefficients are read as follows: \n");
        for (int index = 0; index < metric_values_length; index++)
        {
            printf("%.04f ", distortionCoefficients[index]);
        }
        std::cout << std::endl;
    }

    // Assuming we have succesfully read the parameters from the csv file, let us proceed for live video

    // Open the video capture to show live video.
    cv::VideoCapture* capture = new cv::VideoCapture(0);
    // Check if any video capture device is present.
    if (!capture->isOpened())
    {
        printf("Unable to open the primary video device.\n");
        return(-404);
    }

    cv::Size refs((int)capture->get(cv::CAP_PROP_FRAME_WIDTH),
        capture->get(cv::CAP_PROP_FRAME_HEIGHT));
    if (debug) {
        printf("Camera Capture size: %d x %d \n.", refs.width, refs.height);
    }

    // Create placeholders for vectors of translation and rotation
    cv::Mat rVector;
    cv::Mat tVector;
    //std::vector<float> rVector;
    //std::vector<float> tVector;

    cv::Mat frame;
    while (true) {
        *capture >> frame;
        //get new frame from camera, treating as stream.
        if (frame.empty()) {
            printf("Frame is empty");
            break;
        }
        
        //Check if chessboard exists in the frame.
        std::vector<cv::Point2f> imagePoints;
        bool status = detectAndExtractChessBoardCorners(frame, imagePoints);
        
        // Show the image now so that detected chessboard corners are visible.
        cv::imshow("Live Video", frame);
        char key = cv::waitKey(3);
        if (status)
        {
            debug = true;
            // Build the points set from the corner set
            std::vector<cv::Vec3f> objectPoints;
            buildPointsSet(imagePoints, objectPoints);
            if (debug) { printf("Solving for PnP\n"); }

            if (debug)
            {
                printf("Image Points: \n");
                for (int i = 0; i < imagePoints.size(); i++)
                {
                    std::cout << imagePoints[i] << std::endl;
                }
            }

            if (debug)
            {
                printf("Object Points: \n");
                for (int i = 0; i < objectPoints.size(); i++)
                {
                    std::cout << objectPoints[i] << std::endl;
                }
            }

            // Solve for the pose and position of the camera based on the capture.
            cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distortionCoefficients, rVector, tVector);

            // Code to print the matrices
            printf("Rotation vector is of shape (%d, %d) follows: \n", rVector.rows, rVector.cols);
            std::cout << rVector << std::endl;
            printf("Rotation vector is as follows: \n");
            for (int row = 0; row < rVector.rows; row++)
            {
                for (int col = 0; col < rVector.cols; col++)
                {
                    std::cout << rVector.at<int>(row, col) << " ";
                }
                std::cout << std::endl;
            }

            printf("Translation vector is of shape (%d, %d) follows: \n", tVector.rows, tVector.cols);
            std::cout << tVector << std::endl;
            printf("Translation vector is as follows: \n");
            for (int row = 0; row < tVector.rows; row++)
            {
                for (int col = 0; col < tVector.cols; col++)
                {
                    std::cout << tVector.at<int>(row, col) << " ";
                }
                std::cout << std::endl;
            }

            // cv::waitKey(0); // To Capture the details for report.
            std::vector<cv::Vec2f> projectedObjectPoints;
            cv::projectPoints(objectPoints, rVector, tVector, cameraMatrix, distortionCoefficients, projectedObjectPoints);

            printf("Projected points are :\n");
            for (int index = 0; index < projectedObjectPoints.size(); index++)
            {
                std::cout << projectedObjectPoints[index] << std::endl;
            }
            

        }
        else {
            printf("Chessboard corners are not found.\n");
        }

        if (key == 'q')
        {
            break;
        }
    }
    return 0;
}
