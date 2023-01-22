#include <opencv2/opencv.hpp>


/*This function takes a Mat image nad provides a */
cv::Mat negate_image(cv::Mat &given_image) {
    /** Creating a negative of an existing image
     * Have a copy of the image and negative pixels of the copied image.
        Pseudo code
        copy image
        loop over rows
            loop over columns
                update the value as 255 - current value
    */
    cv::Mat negative_copy;
    negative_copy = cv::Mat::zeros(given_image.size(), CV_16SC3);
    printf("Size of original image is %d & %d\n", given_image.rows, given_image.cols);
    printf("Size of copied image is %d & %d\n", negative_copy.rows, negative_copy.cols);
    for (int row = 0; row < given_image.rows; row++) {
        cv::Vec3b *rowptr = given_image.ptr<cv::Vec3b>(row);
        // uchar *charptr = negative_copy.ptr<uchar>(row);
        for (int column = 0; column < given_image.cols; column++) {
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
