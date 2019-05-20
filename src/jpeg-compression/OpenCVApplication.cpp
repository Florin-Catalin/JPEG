// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <queue>

using namespace std;
using namespace cv;

Mat get_image(char* received_image, int opening_type) {
	Mat image;
	image = imread(received_image, opening_type);
	return image;
}
void display_image(Mat image) {
	namedWindow("Display Image", WINDOW_AUTOSIZE);
	imshow("Display Image", image);
	waitKey(0);
}
Mat convert_to_YCrCb(Mat image) {
	Mat converted_image = image.clone();
	cvtColor(image, converted_image, CV_BGR2YCrCb);
	return converted_image;
}

double cosines(int y, int i, int N)
{
	double arg;
	arg = (((2 * y) + 1)*i*PI) / (2 * N);
	return cos(arg);
}

void FDCT(Mat * img, Mat *result, int q[][8])
{
	int DCT[256][256];
	double sum;
	for (int m = 0; m < img->rows; m += 8)
	{
		for (int n = 0; n < img->cols; n += 8)
		{
			for (int u = 0; u < 8; u++)
			{
				for (int v = 0; v < 8; v++)
				{
					sum = 0.0;
					for (int i = 0; i < 8; i++)
					{
						for (int j = 0; j < 8; j++)
						{
							CvScalar itensity = cvGet2D(img, i + m, j + n);
							sum += cosines(i, u, 8) * cosines(j, v, 8) * (double)itensity.val[0];
						}
					}
				}
			}
		}
	}
}


int main()
{
	char fname[MAX_PATH];
	if (openFileDlg(fname)) {
		
		display_image(convert_to_YCrCb(get_image(fname, CV_LOAD_IMAGE_COLOR)));
	}

	waitKey(0);
	
	// quantization matrix 
	int quan[8][8] = { 16, 11, 10, 16, 24, 40, 51, 61,
					12, 12, 14, 19, 26, 58, 60, 55,
					14, 13, 16, 24, 40, 57, 69, 56,
					14, 17, 22, 29, 51, 87, 80, 62,
					18, 22, 37, 56, 68, 109, 103, 77,
					24, 35, 55, 64, 81, 104, 113, 92,
					48, 64, 78, 87, 113, 121, 120, 101,
					72, 92, 95, 98, 112, 100, 103, 99 };
	return 0;
}