/*
  Author: Bruce A. Maxwell
  Borrowed to have the csv functions.

  Utility functions for reading and writing CSV files with a specific format

  Each line of the csv file is a filename in the first column, followed by numeric data for the remaining columns
  Each line of the csv file has to have the same number of columns
 */
#include <vector>

#ifndef CVS_UTIL_H
#define CVS_UTIL_H

 /*
   Given a filename, and image filename, and the image features, by
   default the function will append a line of data to the CSV format
   file.  If reset_file is true, then it will open the file in 'write'
   mode and clear the existing contents.

   The image filename is written to the first position in the row of
   data. The values in image_data are all written to the file as
   floats.

   The function returns a non-zero value in case of an error.
  */
int append_image_data_csv(char* filename, char* image_filename, std::vector<float>& image_data, int reset_file = 0);

/*ToDo Write Label data to csv*/
int append_label_data_csv(char* filename, std::vector<char*>& image_data, int reset_file);

/*ToDo Write confusion data to csv*/
int append_confusion_data_csv(char* filename, char* className, std::vector<int>& confusion_data, int reset_file);

/*
  Given a filename, and image filename, image label, and the image features, by
  default the function will append a line of data to the CSV format
  file.  If reset_file is true, then it will open the file in 'write'
  mode and clear the existing contents.

  The image filename is written to the first position in the row of data.
  The image label is written to the second position in the row of data.
  The values in image_data are all written to the file as floats.

  The function returns a non-zero value in case of an error.

  @note: @Overloaded to have label as the second column from the original function.
 */
int append_image_data_csv(char* filename, char* image_filename, char* image_label, std::vector<float>& image_data, int reset_file = 0);

/*
  Given a file with the format of a string as the first column and
  floating point numbers as the remaining columns, this function
  returns the filenames as a std::vector of character arrays, and the
  remaining data as a 2D std::vector<float>.

  filenames will contain all of the image file names.
  data will contain the features calculated from each image.

  If echo_file is true, it prints out the contents of the file as read
  into memory.

  The function returns a non-zero value if something goes wrong.
 */
int read_image_data_csv(char* filename, std::vector<char*>& filenames, std::vector<std::vector<float>>& data, bool echo_file = false);

/*
  Given a file with the format of a string as the first column, label as the second column, and
  floating point numbers as the remaining columns, this function
  returns the filenames as a std::vector of character arrays, and the
  remaining data as a 2D std::vector<float>.

  filenames will contain all of the image file names.
  data will contain the features calculated from each image.

  If echo_file is true, it prints out the contents of the file as read
  into memory.

  The function returns a non-zero value if something goes wrong.
 */
int read_image_data_csv(char* filename, std::vector<char*>& filenames, std::vector<char*>& labelnames, std::vector<std::vector<float>>& data, bool echo_file);

/*
  This function returns the fileNames and the labels / ground truths of the images.
 */
int read_file_labels_only(char* featuresFile, std::vector<char*>& filenames, std::vector<char*>& labelnames, bool echo_file);

#endif
