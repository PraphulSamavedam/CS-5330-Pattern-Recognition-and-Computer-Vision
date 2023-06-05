/**
* Written by: Samavedam Manikhanta Praphul
*                   Poorna Chandra Vemula
* This file provides the signatures of several functions required in the project.
*/

/**This function draws the bounding box for a region
* @param directoryPath path of the dir to read files from
* @param filesList list of files
* @param echostatus[default=false] set this to have print statements about status
* @returns 0 if the feature is properly extracted.
*        non zero if the operation is failure.
*/
#include <vector>

int getFilesFromDirectory(char* directoryPath, std::vector<char* >& filesList, int echoStatus);
