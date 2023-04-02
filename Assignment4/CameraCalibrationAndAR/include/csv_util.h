/** Written by: Samavedam Manikhanta Praphul
*                Poorna Chandra Vemula
* Borrowed from: Bruce Maxwell
* Utility functions for reading and writing CSV files with a specific format
* Each line of the csv file is a metric name in the first column, followed by numeric data for the remaining columns.
*/
#include <vector>

#ifndef CVS_UTIL_H
#define CVS_UTIL_H

/*
  Given a filename, and metric name, and the metric features, by
  default the function will append a line of data to the CSV format
  file.  If reset_file is true, then it will open the file in 'write'
  mode and clear the existing contents.

  The metric name is written to the first position in the row of data.
  The values in metric are all written to the file as floats.

  The function returns a non-zero value in case of an error.
 */
int append_metric_data_csv( char *filename, char *metric_name, std::vector<float> &metric_values_data, int reset_file = 0 );


/*
  Given a file with the format of metric name as a string the first column and
  floating point numbers as the remaining columns, this function
  returns the filenames as a std::vector of character arrays, and the
  remaining data as a 2D std::vector<float>.

  data will contain the metric values calculated for each metric name.

  If echo_file is true, it prints out the contents of the file as read
  into memory.

  The function returns a non-zero value if something goes wrong.
 */
int read_metric_data_csv( char *fileName, std::vector<char*> &metricNames, std::vector<std::vector<float>> &data, bool echo_file = false );

#endif
