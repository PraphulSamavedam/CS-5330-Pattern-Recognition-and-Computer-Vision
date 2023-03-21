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


    while (true) {
        *capture >> frame;
        //get new frame from camera, treating as stream.
        if (frame.empty()) {
            printf("Frame is empty");
            break;
        }
        
        // printf("Status of chess board capture in video loop: %d\n", status);
        cv::namedWindow("Image", cv::WINDOW_GUI_EXPANDED);
        cv::imshow("Image", frame);
        char key = cv::waitKey(10);

        cv::Mat cornersImage;
        frame.copyTo(cornersImage);
        cornerHarris(cornersImage);
//
       
//        cv::waitKey(10);
        
        for( int i = 0; i < cornersImage.rows ; i++ )
           {
               for( int j = 0; j < cornersImage.cols; j++ )
               {
                   if( (int) cornersImage.at<uchar>(i,j) > 150 )
                   {
                       cv::circle( cornersImage, cv::Point(j,i), 5,  cv::Scalar(0), 2, 8, 0 );
                   }
               }
        }
        
        
       
        cv::namedWindow("corners Image", cv::WINDOW_GUI_EXPANDED);
        cv::imshow("corners Image", cornersImage);
    }
    return 0;
}

