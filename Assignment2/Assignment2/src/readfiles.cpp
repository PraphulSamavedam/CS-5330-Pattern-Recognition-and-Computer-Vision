/*  Written by: Samavedam Manikhanta Praphul
  @note Inspired from sample code shared by Bruce A.Maxwell to identify image fils in a directory
  Tweaked to get the list of the files from the directory read instead of printing the path on screen. 
*/
#define _CRT_SECURE_NO_WARNINGS //To Supress strcpy, strcpy warnings
#include <cstdio> // Standard IO lib
#include <cstring> // String Standard lib
#include <cstdlib> // C Standard lib 
#include <dirent.h> // Required to iterate over the files in the directory
#include <vector> //To store the files to process in the given directory 

/*
  Given a directory on the command line, scans through the directory for image files.

  Prints out the full path name for each file.  This can be used as an argument to fopen or to cv::imread.
 */
int getFilesFromDirectory(char* directoryPath, std::vector<char*>& filesList, int echoStatus) {
  char dirname[256];
  char buffer[256];
  FILE *fp;
  DIR *dirp;
  struct dirent *dp;
  int i;

  // get the directory path
  strcpy(dirname, directoryPath);
  printf("Processing directory %s\n", dirname );

  // open the directory
  dirp = opendir( dirname );
  if( dirp == NULL) {
    printf("Cannot open directory %s\n", dirname);
    exit(-1);
  }

  // loop over all the files in the image file listing
  while( (dp = readdir(dirp)) != NULL ) {

    // check if the file is an image
    if( strstr(dp->d_name, ".jpg") ||
	strstr(dp->d_name, ".png") ||
	strstr(dp->d_name, ".ppm") ||
	strstr(dp->d_name, ".tif") ) {
        if (echoStatus) { printf("processing image file: %s\n", dp->d_name); }
        // build the overall filename
        strcpy(buffer, dirname);
        strcat(buffer, "/");
        strcat(buffer, dp->d_name);
        if (echoStatus) { printf("full path name: %s\n", buffer); }
        char* fname = new char[strlen(buffer) + 1];
        strcpy(fname, buffer);
        filesList.push_back(fname);
    }
  }

  return(0);
}


