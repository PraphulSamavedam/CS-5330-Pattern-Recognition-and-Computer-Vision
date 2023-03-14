/** Written by: Samavedam Manikhanta Praphul
* Borrowed from Bruce A. Maxwell
* CS 5330 Computer Vision
* Spring 2023

CPP functions for reading CSV files with a specific format
- first column is a string containing a metric name
- every other column is a number (float)

The function returns a std::vector of char* for the metricnames and a 2D std::vector of floats for the data
*/

#define _CRT_SECURE_NO_WARNINGS // To supress strcpy warnings

#include <cstdio>
#include <cstring>
#include <vector>
#include "opencv2/opencv.hpp"

/*
  reads a string from a CSV file. the 0-terminated string is returned in the char array os.

  The function returns false if it is successfully read. It returns true if it reaches the end of the line or the file.
 */
int getstring(FILE* fp, char os[]) {
    int p = 0;
    int eol = 0;

    for (;;) {
        char ch = fgetc(fp);
        if (ch == ',') {
            break;
        }
        else if (ch == '\n' || ch == EOF) {
            eol = 1;
            break;
        }
        // printf("%c", ch ); // uncomment for debugging
        os[p] = ch;
        p++;
    }
    // printf("\n"); // uncomment for debugging
    os[p] = '\0';

    return(eol); // return true if eol
}

int getint(FILE* fp, int* v) {
    char s[256];
    int p = 0;
    int eol = 0;

    for (;;) {
        char ch = fgetc(fp);
        if (ch == ',') {
            break;
        }
        else if (ch == '\n' || ch == EOF) {
            eol = 1;
            break;
        }

        s[p] = ch;
        p++;
    }
    s[p] = '\0'; // terminator
    *v = atoi(s);

    return(eol); // return true if eol
}

/*
  Utility function for reading one float value from a CSV file

  The value is stored in the v parameter

  The function returns true if it reaches the end of a line or the file
 */
int getfloat(FILE* fp, float* v) {
    char s[256];
    int p = 0;
    int eol = 0;

    for (;;) {
        char ch = fgetc(fp);
        if (ch == ',') {
            break;
        }
        else if (ch == '\n' || ch == EOF) {
            eol = 1;
            break;
        }

        s[p] = ch;
        p++;
    }
    s[p] = '\0'; // terminator
    *v = atof(s);

    return(eol); // return true if eol
}

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
int append_metric_data_csv(char* filename, char* metric_name, std::vector<float>& metric_values_data, int reset_file = 0) {
    char buffer[256];
    char mode[8];
    FILE* fp;

    strcpy(mode, "a");

    if (reset_file) {
        strcpy(mode, "w");
    }

    fp = fopen(filename, mode);
    if (!fp) {
        printf("Unable to open output file %s\n", filename);
        exit(-1);
    }

    // write the filename and the feature vector to the CSV file
    strcpy(buffer, metric_name);
    std::fwrite(buffer, sizeof(char), strlen(buffer), fp);
    for (int i = 0; i < metric_values_data.size(); i++) {
        char tmp[256];
        sprintf(tmp, ",%.4f", metric_values_data[i]);
        std::fwrite(tmp, sizeof(char), strlen(tmp), fp);
    }

    std::fwrite("\n", sizeof(char), 1, fp); // EOL

    fclose(fp);

    return(0);
}

/*
  Given a file with the format of a string as the first column and
  floating point numbers as the remaining columns, this function
  returns the metricNames as a std::vector of character arrays, and the
  remaining data as a 2D std::vector<float>.

  metricNames will contain all of the image file names.
  data will contain the features calculated from each image.

  If echo_file is true, it prints out the contents of the file as read
  into memory.

  The function returns a non-zero value if something goes wrong.
 */
int read_metric_data_csv(char* filename, std::vector<char*>& metricNames, std::vector<std::vector<float>>& data, bool echo_file) {
    FILE* fp;
    float fval;
    char img_file[256];

    fp = fopen(filename, "r");
    if (!fp) {
        printf("Unable to open feature file\n");
        return(-1);
    }

    printf("Reading %s\n", filename);
    for (;;) {
        std::vector<float> dvec;


        // read the filename
        if (getstring(fp, img_file)) {
            break;
        }
        // printf("Evaluting %s\n", filename);

        // read the whole feature file into memory
        for (;;) {
            // get next feature
            float eol = getfloat(fp, &fval);
            dvec.push_back(fval);
            if (eol) break;
        }
        // printf("read %lu features\n", dvec.size() );

        data.push_back(dvec);

        char* fname = new char[strlen(img_file) + 1];
        strcpy(fname, img_file);
        metricNames.push_back(fname);
    }
    fclose(fp);
    printf("Finished reading CSV file\n");

    if (echo_file) {
        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < data[i].size(); j++) {
                printf("%.4f  ", data[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    return(0);
}