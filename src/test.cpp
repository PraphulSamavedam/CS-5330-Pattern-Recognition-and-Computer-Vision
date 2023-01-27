#include <opencv2/opencv.hpp>
#include "../include/imageFilters.h"
#include "../include/extensions.h"



int main(int argc, char* argvs[]) {
	char filename[250] = "C:\\Users\\Samavedam\\Desktop\\alpha.jpg";
	cv::Mat original_image = cv::imread(filename);
	if (original_image.data == NULL)
	{
		printf("Error reading the file %s", filename);
		return -1;
	}

	int window_size = cv::WINDOW_AUTOSIZE;

	/*Original Image*/
	cv::namedWindow("Original Image", window_size);
	cv::imshow("Original Image", original_image);

	/*Blurred Image*/
	cv::Mat blurred_image = cv::Mat(original_image.size(), original_image.type());
	blur5x5(original_image, blurred_image);
	cv::imshow("Original blur image", blurred_image);

	/* // Not required as the image is already in the desired type.
	cv::Mat scaled_image;
	cv::convertScaleAbs(blurred_image, scaled_image);
	cv::namedWindow("Scaled Blur Image", window_size);
	cv::imshow("Scaled Blur Image", scaled_image);*/

	/*SobelX Image*/
	cv::Mat sobelX_image = cv::Mat(original_image.size(), CV_16SC3);
	sobelX3x3(original_image, sobelX_image);
	cv::Mat scaled_sobelXimage;
	cv::convertScaleAbs(sobelX_image, scaled_sobelXimage);
	cv::namedWindow("SobelX Image", window_size);
	cv::imshow("SobelX Image", scaled_sobelXimage);

	/*SobelX Brute Image to understand the separable implementation benefits.
	cv::Mat sobelX_Brute_image = cv::Mat(original_image.size(), CV_16SC3);
	sobelX3x3_brute(original_image, sobelX_Brute_image);
	cv::Mat scaled_sobelX_brute_image;
	cv::convertScaleAbs(sobelX_image, scaled_sobelX_brute_image);
	//absolute_image(sobelX_image, scaled_sobelXimage);
	cv::namedWindow("SobelX Brute Image", window_size);
	cv::imshow("SobelX Brute Image", scaled_sobelX_brute_image);
	*/

	/* SobelX using 2D filter to cross check implementation
	cv::Mat sx_image = cv::Mat(original_image.size(), CV_16SC3);
	cv::Mat_<float> kernel(3,3);
	kernel << -1, 0, 1, -2, 0, 2, -1, 0, 1;
	cv::filter2D(original_image, sx_image, 0, kernel);
	cv::Mat scaled_sx_image;
	cv::convertScaleAbs(sx_image, scaled_sx_image);
	cv::namedWindow("SobelX Filter2D Image", window_size);
	cv::imshow("SobelX Filter2D Image", scaled_sx_image);
	*/

	//SobelY Image
	cv::Mat sobelY_image = cv::Mat(original_image.size(), CV_16SC3);
	sobelY3x3(original_image, sobelY_image);
	cv::Mat scaled_sobelYimage;
	cv::convertScaleAbs(sobelY_image, scaled_sobelYimage);
	cv::namedWindow("SobelY Image", window_size);
	cv::imshow("SobelY Image", scaled_sobelYimage);

	/* SobelY using 2D filter to cross check implementation
	cv::Mat sy_image = cv::Mat(original_image.size(), CV_16SC3);
	cv::Mat_<float> kernel(3, 3);
	kernel << 1, 2, 1, 0, 0, 0, -1, -2, 1;
	cv::filter2D(original_image, sy_image, 0, kernel);
	cv::Mat scaled_sy_image;
	cv::convertScaleAbs(sy_image, scaled_sy_image);
	cv::namedWindow("SobelY Filter2D Image", window_size);
	cv::imshow("SobelY Filter2D Image", scaled_sy_image);
	*/

	//Magnitude Image
	cv::Mat magnitude_image = cv::Mat(original_image.size(), CV_16SC3); 
	magnitude(sobelX_image, sobelY_image, magnitude_image);
	cv::namedWindow("Magnitude Image", window_size); 
	cv::imshow("Magnitude Image", magnitude_image);

	//Blur and Quantize
	cv::Mat blur_quantized_image = cv::Mat(original_image.size(), original_image.type());
	blurQuantize(original_image, blur_quantized_image, 10);
	cv::namedWindow("Blur and Quantize Image", window_size);
	cv::imshow("Blur and Quantize Image", blur_quantized_image);

	////Quantize
	//cv::Mat quantized_image = cv::Mat(original_image.size(), original_image.type());
	//blurQuantize(original_image, quantized_image, 10);
	//cv::namedWindow("Quantized Image", window_size);
	//cv::imshow("Quantized Image", quantized_image);

	// Animation/Cartoon
	cv::Mat cartoon_image = cv::Mat(original_image.size(), CV_16SC3);
	cartoon(original_image, cartoon_image, 15, 15);
	cv::Mat scaled_cartoon_image;
	cv::convertScaleAbs(cartoon_image, scaled_cartoon_image);
	cv::namedWindow("Cartoon Image", window_size);
	cv::imshow("Cartoon Image", scaled_cartoon_image);

	cv::Mat edges = cv::Mat(original_image.size(), original_image.type());
	laplacian_filter(original_image, edges);
	cv::imshow("Edges detected", edges);
	
	while (true) {
		char key = cv::waitKey(0);
		if (key == 'q') {
			printf("Terminating the program\n\n");
			cv::destroyAllWindows();
			break;
		}
		/*else if (key == 's') {
			if (!blurred_image.empty())
			{
				cv::imwrite("Original Blur image", blurred_image)
			}
		}*/
		else
		{
			printf("Waiting to press q to exit\n");
		}
	}
	return 0;
}
