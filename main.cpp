/** This program reads a test image from the data/images folder and displays it for 5 seconds.
 * Created by Samavedam Manikhanta Praphul
*/

#include <cstdio> // Lots of standard C/C++ functions including printf, scanf */
#include <iostream>
#include <cstring> // C/C++ functions for working with strings (including char arrays)
#include <opencv2/opencv.hpp> // main OpenCV include file which has all the headers
// using namespace cv;

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
    
    if (image_src.data == NULL){ // Check if image read is successful
        printf("Unable to read the image %s\n", argv[1]);
        return(-1);
    }

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

   cv::Mat src2 = cv::Mat::zeros(100, 200, CV_32FC3 );

   // Creating an image of same size and same type
   cv::Mat src3;
   src3.create(src.size(), src.type());

   // Size of the image --> rows, cols
   printf("Size of the image is %d %d ", src.rows, src.cols);

    // Start to split into functions
    cv::Mat image_data = cv::imread(fileName); // reads the image from the filepath and allocates space to image_src variable 
    // By default, converted to 8-bit color channel BGR format
    
    if (image_data.data == NULL){ // Check if image read is successful
        printf("Unable to read the image %s\n", argv[1]);
        return(-1);
    }
    
    cv::namedWindow("Original image read",1);
    cv::imshow("Original image read",image_data);
    cv::waitKey(0);
    if (cv::pollKey() == 'q')
    {
        cv::destroyWindow("Original image read");
    }

    return 0;
}

cv::Mat negate_image(cv::Mat given_image){
    /** Creating a negative of an existing image
     * Have a copy of the image and negative pixels of the copied image.
        Pseudo code
        copy image
        loop over rows
            loop over columns 
                update the value as 255 - current value
    */
    cv::Mat negative_copy;
    given_image.copyTo(negative_copy);
    for (int row=0; row< negative_copy.rows;row++){
        cv::Vec3b *rowptr = negative_copy.ptr<cv::Vec3b>(row);
        // uchar *charptr = negative_copy.ptr<uchar>(row);
        for (int column=0;column < negative_copy.cols;column++){
            /* Inefficient way 
            src.at<uchar>(i,j) = 255 - src.at<uchar>(i,j);
            */
            /* Quicker method*/
            rowptr[column][0] = 255 - rowptr[column][0];
            rowptr[column][1] = 255 - rowptr[column][1];
            rowptr[column][2] = 255 - rowptr[column][2];
            /* Alternative way for quick method
            charptr[3*j + 0] = 255 - charptr[3*j + 0];
            charptr[3*j + 1] = 255 - charptr[3*j + 1];
            charptr[3*j + 2] = 255 - charptr[3*j + 2];
            */
        }
    }
    return negative_copy;
}
