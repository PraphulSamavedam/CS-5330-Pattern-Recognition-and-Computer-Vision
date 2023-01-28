/** This program reads a test image from the data/images folder and displays it for 5 seconds.
 * Created by Samavedam Manikhanta Praphul
*/

#include <cstdio> // Lots of standard C/C++ functions including printf, scanf */
#include <iostream>
#include <cstring> // C/C++ functions for working with strings (including char arrays)
#include <opencv2/opencv.hpp> // main OpenCV include file which has all the headers
#include <opencv2/highgui.hpp> // Required for Keyboard functionalities
// using namespace cv;
#include "include/helper.h"

// define main function
// argc is the number of command line arguments
// argv is the array of character arrays of command line arguments
// argv[0] is always the name of the executable file/function
int main(int argc, char* argv[]) {
    cv::Mat image_src; // defines a Mat data type with only header and no data.
    char fileName[265]; // char arary only header and no data

    if (argc < 2)
    {
        printf(" Fatal error, image argument is missing\nUsage is %s <image fileName> \n", argv[0]);
        return -1;
    }
    strcpy_s(fileName, argv[1]);
    printf("Trying to access %s \n", fileName);

    image_src = cv::imread(fileName); // reads the image from the filepath and allocates space to image_src variable 
    // By default, converted to 8-bit color channel BGR format

    if (image_src.data == NULL) { // Check if image read is successful
        printf("Unable to read the image %s\n", argv[1]);
        return(-1);
    }
    
    printf("Successfully read %s.\n", argv[1]);

    /*OpenCV allocates image data when
    imread, create, copyTo, as a result of operations on images.
    many other OpenCV functions return a new image allocated.
    OpenCV deallocates the data when it is out of scope/ all references to the data are gone.
    */

    // Aliasing situation
    cv::Mat src = image_src; // No new memory is allocated. src also refers image_src
    // For Copying image_src to src 
    image_src.copyTo(src); // Allocates new data to src and copies from image_src to src

    // Creating a new empty image --> cv::Mat.create(row, column, type)
    /* Open CV constants for types:
    CV_8U -- 8 bit Unsigned grayscale image
    CV_8UC3 -- 8 bit Unsigned 3-color image
    CV_26S -- 16 bit Short greyscale image
    CV_16SC3 -- 16bit Short 3 color image
    CV_32F -- 32 bit Float greyscale image
    CV_32FC3 -- 32 bit Float 3-color image
    */
    src.create(100, 200, CV_16SC3); // 100 rowx 200 coloumn 3-color image

    cv::Mat src2 = cv::Mat::zeros(100, 200, CV_32FC3);

    // Creating an image of same size and same type
    cv::Mat src3;
    src3.create(src.size(), src.type());

    // Size of the image --> rows, cols
    printf("Size of the image is %d x %d.\n", image_src.rows, image_src.cols);

    // Start to split into functions
    cv::Mat image_data = cv::imread(fileName); // reads the image from the filepath and allocates space to image_src variable 
    // By default, converted to 8-bit color channel BGR format

    if (image_data.data == NULL) { // Check if image read is successful
        printf("Unable to read the image %s\n", argv[1]);
        return(-1);
    }

    cv::namedWindow("Original image read",cv::WINDOW_KEEPRATIO);
    cv::imshow("Original image read", image_data);

    cv::Mat negative_image;
    negative_image = negate_image(image_data);
    printf("Size of Negative image is %d & %d\n", negative_image.rows, negative_image.cols);
    cv::namedWindow("Negative image read", cv::WINDOW_FREERATIO);
    cv::imshow("Negative image read", negative_image);
    while (true)
    {
        std::int32_t  keyPressed = cv::waitKey(0);
        if (keyPressed == 113 or keyPressed == 81) {
            printf("Terminating the program as '%c' is pressed!\n", char(keyPressed));
            break;
        }
    }
    cv::destroyAllWindows();
    /*cv::destroyWindow("Original image read");
    cv::destroyWindow("Negative image read");*/
    return 0;
}

