// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <queue>

//#define MAX_PATH 1024


void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}



void histograma()
{
	float M = 0;
	float p[256];
	float miu = 0;
	float sigma = 0;
	int h[256];
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		M = src.rows*src.cols;

		for (int i = 0; i < 256; ++i)
			h[i] = 0;

		for (int i = 0; i < src.rows; ++i)
			for (int j = 0; j < src.cols; ++j)
				h[src.at<uchar>(i, j)]++;


		//the standard deviation of the intensity leves is given by
		for (int g = 0; g < 256; g++)
		{
			p[g] = h[g] / M;
			miu += g * p[g];
		}

		for (int g = 0; g < 256; g++)
			sigma += pow(g - miu, 2)*p[g];
			sigma = sqrt(sigma);


		showHistogram("histogram", h, 256, 500);
		printf("miu: %f\n", miu);
		printf("sigma: %f\n", sigma);
		imshow("Original", src);
		waitKey();
	}
}

void histogram(Mat img)
{
	int h[256];

	for (int i = 0; i < 256; i++)
		h[i] = 0;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			h[img.at<uchar>(i, j)]++;

	showHistogram("Histogram", h, 256, 500);
	waitKey();
}

void media() {
	char fname[MAX_PATH];
	openFileDlg(fname);

	Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Initial image", img);


	float average = 0;
	int M = img.rows * img.cols;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			average += img.at<uchar>(i, j);

	average /= M;

	std::cout << "The medium itensity: " << average << std::endl;
	histogram(img);


}
void standardDeviation() {

	// media 
	char fname[MAX_PATH];
	openFileDlg(fname);

	Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Imaginea initiala", img);

	int M = img.rows * img.cols;

	float average = 0;
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			average += img.at<uchar>(i, j);

	average /= M;



	
	float intensity = average;
	float deviation = 0, x;


	//from 0 to L = 255 highest intesity level 
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			x = pow(img.at<uchar>(i, j) - intensity, 2);
			deviation += x;
		}

	deviation /= M;
	deviation = sqrt(deviation);

	std::cout << "The standard deviation: " << deviation;
	histogram(img);
}


//global threshold  algorithm
void globalBinarization()
{
	char fname[MAX_PATH];
	openFileDlg(fname);

	Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Imagine originala", img);


	//compute the histogram
	int histogramValues[256];

	for (int i = 0; i < 256; i++)
		histogramValues[i] = 0;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			histogramValues[img.at<uchar>(i, j)]++;

	showHistogram("Histograma", histogramValues, 255, 200);


	//initialization of the maximum and minimum intensity level 
	int iMin = 256, iMax = 0;

	//find the minimum intensity 
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			if (img.at<uchar>(i, j) < iMin)
				iMin = img.at<uchar>(i, j);
	

	//find the maximum intensity 
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			if (img.at<uchar>(i, j) > iMax)
				iMax = img.at<uchar>(i, j);

	// takean initial value for the threshold  like the average of the maxim and min intensity level 
	int T = (iMin + iMax) / 2;

	// segment the image after T by dividing the image pixels in 2 groups 
	int lastT = 0;
	float error = 0.1f;//assing an error value (must be positive ) 

	//while  Tk - Tk-1 < error ( |error| < 1 , error > 0 )  Tk = T ( current threshold ) Tk-1 = lastT ( last threshold
	while ((T - lastT) >= error)
	{
		int NG1 = 0, NG2 = 0;
		float G1 = 0, G2 = 0;

		for (int f = iMin; f <= T; f++)
			NG1 += histogramValues[f];

		for (int f = T + 1; f <= iMax; f++)
			NG2 += histogramValues[f];

		for (int f = iMin; f <= T; f++)
			G1 += (f * histogramValues[f]);

		for (int f = T + 1; f <= iMax; f++)
			G2 += (f * histogramValues[f]);

		//calculam mediile G1 si G2 folosind histograma initiala
		G1 /= (float)NG1;
		G2 /= (float)NG2;

		lastT = T;
		//actualizarea pragului
		T = (int)(G1 + G2) / 2;
	}

	Mat rez(img.rows, img.cols, CV_8UC1);

	//binarizare
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			if (img.at<uchar>(i, j) < T)
				rez.at<uchar>(i, j) = 0;
			else
				rez.at<uchar>(i, j) = 255;
		}
	std ::cout << T ;
	imshow("Rezult", rez);
	waitKey(0);
}


void gammaEqualizer()
{
	char fname[MAX_PATH];
	openFileDlg(fname);

	Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	

	int histogramValues[256];

	for (int i = 0; i < 256; i++)
		histogramValues[i] = 0;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			histogramValues[img.at<uchar>(i, j)]++;

	Mat rez(img.rows, img.cols, CV_8UC1);

	float gamma;
	std::cout << "Introduce the correction factor";
	std::cin >> gamma;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			rez.at<uchar>(i, j) = (uchar)(255 * pow(img.at<uchar>(i, j) / (float)255, gamma));

			if (rez.at<uchar>(i, j) < 0)
				rez.at<uchar>(i, j) = 0;

			if (rez.at<uchar>(i, j) > 255)
				rez.at<uchar>(i, j) = 255;
		}

	int newHistogramValues[256];

	for (int i = 0; i < 256; i++)
		newHistogramValues[i] = 0;

	for (int i = 0; i < rez.rows; i++)
		for (int j = 0; j < rez.cols; j++)
			newHistogramValues[rez.at<uchar>(i, j)]++;
	imshow("Initial image", img);

	showHistogram("The histogram of the initial image ", histogramValues, 255, 200);

	showHistogram("The histogram after applying the correction", newHistogramValues, 255, 200);
	imshow("rez", rez);
	waitKey(0);
}


void histogramEqualizer()
{
	char fname[MAX_PATH];
	openFileDlg(fname);

	Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	imshow("image", img);

	int histogramValues[256];
	float FDP[256];

	for (int i = 0; i < 256; i++)
		histogramValues[i] = 0;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			histogramValues[img.at<uchar>(i, j)]++;



	int M = img.rows * img.cols;

	for (int i = 0; i < 256; i++)
		FDP[i] = histogramValues[i] / (float)M;

	Mat rez(img.rows, img.cols, CV_8UC1);

	float s = 0;
	float PC[256];
	int rk;

	for (int k = 0; k <= 255; k++)
	{
		rk = (int)(k / (float)255);

		for (int g = 0; g <= k; g++)
			s += (FDP[g] / (float)M);

		PC[rk] = s;
	}

	int tab[256];

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			tab[img.at<uchar>(i, j)] = (int)(255 * PC[img.at<uchar>(i, j)]);

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			rez.at<uchar>(i, j) = tab[img.at<uchar>(i, j)];

	int newHistogramValues[256];

	for (int i = 0; i < 256; i++)
		newHistogramValues[i] = 0;

	for (int i = 0; i < rez.rows; i++)
		for (int j = 0; j < rez.cols; j++)
			newHistogramValues[rez.at<uchar>(i, j)]++;
	showHistogram("Initial histogram", histogramValues, 255, 200);
	showHistogram("Resulted histogram", newHistogramValues, 255, 200);
	waitKey(0);
}




void filtruLaplace()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = src.clone();
		int dim = 3;
		int mat[3][3] = { { 0, -1, 0 }, { -1, 4, -1 }, { 0, -1, 0 } };

		for (int i = 1; i < src.rows - 1; i++) {
			for (int j = 1; j < src.cols - 1; j++) {
				int suma = 0;
				for (int k = -dim / 2; k <= dim / 2; k++) {
					for (int m = -dim / 2; m <= dim / 2; m++)
					{
						suma += src.at<uchar>(i + k, j + m)*mat[k + dim / 2][m + dim / 2];
					}

				}
				if (suma >= 255)
					dst.at<uchar>(i, j) = 255;
				else {
					if (suma < 0)
						dst.at<uchar>(i, j) = 0;
					else
						dst.at<uchar>(i, j) = suma;
				}
			}
		}

		imshow("Source", src);
		imshow("Dest", dst);
		waitKey();
	}
}




void generalConvolution(int w  )
{
	char fname[MAX_PATH];
	openFileDlg(fname);

	Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Initial image", img);


	
		Mat dst = img.clone();

		int H[3][3] = { {0,-1,0 },
						{ -1,5,-1},
						{0,-1,0} };

		//the sum of positive filter coefficients magnitude
		int S_pos = 0;


		//the sum of positive filter coefficients magnitude
		int S_neg = 0;

		int k = w / 2;
		double fs = 0;

		for (int j = -k; j <= k; j++) {
			for (int i = -k; i <= k; i++) {
				if (H[j + k][i + k] > 0) {
					S_pos += H[j + k][i + k];
				}
				else if (H[j + k][i + k] < 0) {
					S_neg -= H[j + k][i + k];
				}
			}
		}


		fs = 1.0 / (2.0*max(S_pos, S_neg));

		for (int y = k; y < img.rows - k; y++) {
			for (int x = k; x < img.cols - k; x++) {
				int aux = 0;
				for (int i = -k; i <= k; i++) {
					for (int j = -k; j <= k; j++) {
						aux += img.at<uchar>(x + j, y + i)*H[j + k][i + k];
					}
					dst.at<uchar>(x, y) = aux * fs + 127;
				}
			}
		}

		imshow("result high_filter", dst);
		waitKey(0);



}


void filtruGauss()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = src.clone();
		int dim = 3;
		int mat[3][3] = { { 1, 2, 1 }, { 2, 4, 2 }, { 1, 2, 1 } };

		for (int i = 1; i < src.rows - 1; i++) {
			for (int j = 1; j < src.cols - 1; j++) {
				int suma = 0;
				for (int k = -dim / 2; k <= dim / 2; k++) {
					for (int m = -dim / 2; m <= dim / 2; m++)
					{
						suma += src.at<uchar>(i + k, j + m)*mat[k + dim / 2][m + dim / 2];
					}

				}
				dst.at<uchar>(i, j) = suma / 16;
			}
		}

		imshow("Source", src);
		imshow("Dest", dst);
		waitKey();
	}
}

void filtruMedian()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = src.clone();

		int w;
		printf(" Introduce a value for w: ex(4) ");
		scanf("%d", &w);
		vector<uchar> v;

		for (int i = w / 2; i < src.rows - w / 2; i++)
			for (int j = w / 2; j < src.cols - w / 2; j++)
			{
				v.clear();
				for (int k = -w / 2; k <= w / 2; k++)
					for (int m = -w / 2; m <= w / 2; m++)
					{
						v.push_back(src.at<uchar>(k + i, m + j));
					}
				sort(v.begin(), v.end());
				dst.at<uchar>(i, j) = v[w*w / 2];
			}
		imshow("Sursa", src);
		imshow("Destinatie", dst);
		waitKey();
	}
}



void centering_transform(Mat img) {
	//expects floating point image
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
		}
	}
}
Mat generic_frequency_domain_filter(Mat src) {
	//convert input image to float image
	Mat srcf;
	src.convertTo(srcf, CV_32FC1);
	//centering transformation
	centering_transform(srcf);
	//perform forward transform with complex image output
	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);
	//split into real and imaginary channels
	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	split(fourier, channels); // channels[0] = Re(DFT(I)), channels[1] = Im(DFT(I))
	//calculate magnitude and phase in floating point images mag and phi
	Mat mag, phi;
	magnitude(channels[0], channels[1], mag);
	phase(channels[0], channels[1], phi);
	//display the phase and magnitude images here
	// ......



	//high pass 
	int h = fourier.rows;
	int w = fourier.cols;
	int R = 10;
	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
			if (((h / 2 - i)*(h / 2 - i) + (w / 2 - j)*(w / 2 - j)) <= R * R)
			{
				channels[0].at<float>(i, j) = 0;
				channels[1].at<float>(i, j) = 0;
			}

	//insert filtering operations on Fourier coefficients here
	// ......
	//store in real part in channels[0] and imaginary part in channels[1]
	// ......
	//perform inverse transform and put results in dstf
	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
	//inverse centering transformation
	centering_transform(dstf);
	//normalize the result and put in the destination image
	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	return dst;
}

Mat lowPassGauss(Mat src) {
	Mat srcf;
	src.convertTo(srcf, CV_32FC1);
  centering_transform(srcf);

	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	split(fourier, channels);

	int h = fourier.rows;
	int w = fourier.cols;
	float A = 20;
	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
		{
			channels[0].at<float>(i, j) = channels[0].at<float>(i, j)*exp(-(((h / 2 - i)*(h / 2 - i) + (w / 2 - j)*(w / 2 - j)) / (A*A)));
			channels[1].at<float>(i, j) = channels[1].at<float>(i, j)*exp(-(((h / 2 - i)*(h / 2 - i) + (w / 2 - j)*(w / 2 - j)) / (A*A)));
		}


	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

	centering_transform(dstf);
	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	return dst;
}




Mat convolution(Mat src, Mat kernel, float m) {

	float S = 0;
	float k = 0;

	Mat dst = Mat(src.size(), CV_8UC1);

	k = kernel.cols / 2;
	if (k < 2) k = kernel.rows / 2;
	dst.convertTo(dst, CV_32FC1);

	for (int i = k; i < src.rows - k; i++) {
		for (int j = k; j < src.cols - k; j++) {

			S = 0.0;
			for (int ki = 0; ki < kernel.rows; ki++)
				for (int kj = 0; kj < kernel.cols; kj++) {

					S += (1 / m) * src.at<byte>(i + ki - k, j + kj - k) * kernel.at<float>(ki, kj);
				}
			dst.at<float>(i, j) = S;
		}
	}
	dst.convertTo(dst, CV_8UC1);
	return dst;

}

void gaussian_filter_1x2() {

	Mat src, dst;
	char fname[MAX_PATH];
	float data[100];

	//float sigma;
	//printf("Introduce the sigma value : \n");
	//scanf("%f", &sigma);

	//int n;
	//n = 6 * sigma;
	//if (n % 2 == 0) n += 1;

	//printf("The size of the kernel is: %d ", n);

	int n;
	printf("Introduce w : (odd number ) \n");
	scanf("%i", &n); 
	if (n % 2 == 0) n++;

	float sigma;
	sigma = (float)n / 6; 

	Mat kernel = Mat(n, n, CV_32F);
	float sum = 0;
	while (openFileDlg(fname)) {

		double t = (double)getTickCount();

		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		for (int ki = 0; ki < n; ki++) {
			for (int kj = 0; kj < n; kj++) {
				float intermed = -((ki - n / 2) * (ki - n / 2) + (kj - n / 2)*(kj - n / 2)) / (2 * sigma*sigma);
				kernel.at<float>(ki, kj) = (1 / (2 * PI*sigma*sigma)) * exp(intermed);
				sum += kernel.at<float>(ki, kj);
			}
		}

		dst = convolution(src, kernel, sum);

		t = ((double)getTickCount() - t) / getTickFrequency();

		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("src", src);
		imshow("dst", dst);

		waitKey();
	}

}

Mat apply_convolution(Mat *src, Mat *k, float sum) {
	Mat dst = Mat(src->rows, src->cols, CV_8UC1);
	int dim1 = k->rows;
	int dim2 = k->cols;
	for (int i = dim1 / 2; i <= src->rows - dim1 / 2 - 1; i++) {
		for (int j = dim2 / 2; j <= src->cols - dim2 / 2 - 1; j++) {
			int val = 0;
			for (int ki = -dim1 / 2; ki <= dim1 / 2; ki++) {
				for (int m = -dim2 / 2; m <= dim2 / 2; m++) {
					val += (src->at<uchar>(i + ki, j + m) * k->at<float>(ki + dim1 / 2, m + dim2 / 2));
				}
			}

			dst.at<uchar>(i, j) = (float)val / sum;
		}
	}
	return dst;
}

Mat apply_convolution_float(Mat *src, Mat *k, float sum) {
	Mat dst = Mat(src->rows, src->cols, CV_32FC1);
	int dim1 = k->rows;
	int dim2 = k->cols;
	for (int i = dim1 / 2; i <= src->rows - dim1 / 2 - 1; i++) {
		for (int j = dim2 / 2; j <= src->cols - dim2 / 2 - 1; j++) {
			int val = 0;
			for (int ki = -dim1 / 2; ki <= dim1 / 2; ki++) {
				for (int m = -dim2 / 2; m <= dim2 / 2; m++) {
					val += (src->at<uchar>(i + ki, j + m) * k->at<float>(ki + dim1 / 2, m + dim2 / 2));
				}
			}

			dst.at<float>(i, j) = (float)val / sum;
		}
	}
	return dst;
}


void gaussian_filter_2x1() {
	char fname[MAX_PATH];
	Mat src, dst, intermediate;

	/*
	float sigma;
	printf("Introduce the sigma value : \n");
	scanf("%f", &sigma);

	int n = sigma * 6;
	if (n % 2 == 0) n += 1;
	*/

	int n;
	printf("Introduce w : (odd number ) \n");
	scanf("%i", &n);
	if (n % 2 == 0) n++;

	float sigma;
	sigma = (float)n / 6;
	Mat kernel1 = Mat(n, 1, CV_32FC1);
	Mat kernel2 = Mat(1, n, CV_32FC1);

	int x0 = n / 2;
	int y0 = n / 2;

	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		dst = Mat(src.rows, src.cols, CV_8UC1, Scalar(255));

		double t = (double)getTickCount();

		float sumx = 0;
		float sumy = 0;

		for (int i = 0; i < kernel1.rows; i++)
		{
			float exponent = ((i - x0)*(i - x0)) / (2 * sigma*sigma);
			kernel1.at<float>(i, 0) = 1 / (2 * PI*sigma*sigma)*exp(-exponent);
			sumx += kernel1.at<float>(i, 0);
		}

		for (int i = 0; i < kernel2.cols; i++)
		{
			float exponent = ((i - y0)*(i - y0)) / (2 * sigma*sigma);
			kernel2.at<float>(0, i) = 1 / (2 * PI*sigma*sigma)*exp(-exponent);
			sumy += kernel2.at<float>(0, i);
		}

		dst = apply_convolution(&apply_convolution(&src, &kernel1, sumx), &kernel2, sumy);



		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("Time = %.3f [ms]\n", t * 1000);
		
		imshow("src", src);
		imshow("dst", dst);
		

		waitKey();
	}
}


void median_filter(int kernel_dim)
{
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1, Scalar(255));

		double t = (double)getTickCount();

		int dx[9] = { -1,-1,-1,0,0,0,1,1,1 };
		int dy[9] = { -1,0,1,-1,0,1,-1,0,1 };
		//int kernel_dim = 7;
		int height = src.rows;
		int width = src.cols;
		int sum;

		for (int i = 1; i < src.rows - 1; i++)
		{
			for (int j = 1; j < src.cols - 1; j++)
			{
				// current pixel is (i,j)

				std::vector<int> val;
				//for (int i = 0; i <= kernel_dim/2 ; i ++ )
			   // for (int j = 0; j <= kernel_dim/2; j ++ )
				for (int k = 0; k <= 8; k++)
				{ 
					val.push_back(src.at<uchar>(i + dy[k], j + dx[k]));
				}
				std::sort(val.begin(), val.end());

				dst.at<uchar>(i, j) = val[4];

			}
		}

		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("init image", src);
		imshow("final image", dst);
	}
}


Mat apply_convolution_float(Mat *src, Mat *p) {
	Mat dst = Mat(src->rows, src->cols, CV_32FC1);
	for (int i = 1; i < src->rows - 1; i++) {
		for (int j = 1; j < src->cols - 1; j++) {
			float val = 0;
			for (int ki = -1; ki <= 1; ki++) {
				for (int m = -1; m <= 1; m++) {
					val += (src->at<uchar>(i + ki, j + m) * p->at<float>(ki + 1, m + 1));
				}
			}
			dst.at<float>(i, j) = val;
		}
	}
	return dst;
}


void canny_gradient()
{
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		Mat Gx, Gy;

		//Sobel
		Mat Sx = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
		Mat Sy = (Mat_<float>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);


		Gx = apply_convolution_float(&src, &Sx);
		Gy = apply_convolution_float(&src, &Sy);


		Mat G = Mat(src.rows, src.cols, CV_32FC1);
		Mat alfa = Mat(src.rows, src.cols, CV_32FC1);

		Mat Gx_show;
		Mat Gy_show;
		

		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++) {
				G.at<float>(i, j) = sqrt(pow(Gx.at<float>(i, j), 2) + pow(Gy.at<float>(i, j), 2));
				alfa.at<float>(i, j) = atan2(Gy.at<float>(i, j), Gx.at<float>(i, j));
				
				if (alfa.at<float>(i, j) < 0)
					alfa.at<float>(i, j) += 2 * PI;


			}

		Gx.convertTo(Gx_show, CV_8UC1);
		Gy.convertTo(Gy_show, CV_8UC1);
		imshow("gx", Gx_show);
		imshow("gy", Gy_show);

		Mat G_show;
		G.convertTo(G_show, CV_8UC1);
		imshow("gradient matrix ", G_show);
		imshow("src", src);

	

		Mat newG = G.clone();

		for (int i = 1; i < newG.rows-1; i++)
			for (int j = 1; j < newG.cols - 1; j++)
			{
				float angle = alfa.at<float>(i, j);
				int beta = (int)round(alfa.at<float>(i, j) * 8 / (2 * PI)) % 8;
				float pixel = abs(newG.at<float>(i, j));

				switch (beta % 4 )
				{
				case 0:
					if (pixel >= abs((newG.at<float>(i, j - 1)) && pixel >= abs((newG.at<float>(i, j + 1)))))
						newG.at<float>(i, j) = pixel;
					else
						newG.at<float>(i, j) = 0;
					break;
				case 1:
					if (pixel >= abs((newG.at<float>(i - 1, j + 1)) && pixel >= abs((newG.at<float>(i + 1, j - 1)))))
						newG.at<float>(i, j) = pixel;
					else
						newG.at<float>(i, j) = 0;
					break;
				case 2:
					if (pixel >= abs(newG.at<float>(i + 1, j)) && pixel >= abs(newG.at<float>(i - 1, j)))
						newG.at<float>(i, j) = pixel;
					else
						newG.at<float>(i, j) = 0;
					break;
				case 3:
					if (pixel >= abs(newG.at<float>(i - 1, j - 1)) && pixel >= abs(newG.at<float>(i + 1, j + 1)))
						newG.at<float>(i, j) = pixel;
					else
						newG.at<float>(i, j) = 0;
					break;
				}
			}

		Mat newG_show;

		newG.convertTo(newG_show, CV_8UC1);

		imshow("subtiata", newG_show);

		int v[256];

		for (int k = 0; k < 256; k++)
			v[k] = 0;

		for (int i = 1; i < src.rows - 1; i++) {
			for (int j = 1; j < src.cols - 1; j++) {
				v[newG_show.at<uchar>(i, j)] ++;
			}
		}

		//showHistogram("histogram", v, 256, 256);


		int noEdgePixels;
		int noNonZeroGradientPixels;
		float procent = 0.1f;

		noNonZeroGradientPixels = (newG_show.rows - 2) * (newG_show.cols - 2) - v[0];
		noEdgePixels = procent * noNonZeroGradientPixels;

		int ok = 0;
		int computed_sum = 0;
		int treshold_high;
		for (int i = 255; i >= 0 && ok == 0; i--) {
			if (computed_sum < noEdgePixels)
				computed_sum += v[i];
			else {
				ok = 1;
				treshold_high = i;
			}
		}

		int treshold_low;
		float k_procent = 0.4f;
		treshold_low = k_procent * treshold_high;

		printf("treshold high %d", treshold_high);


		for (int i = 1; i < newG_show.rows - 1; i++) {
			for (int j = 1; j < newG_show.cols - 1; j++)
			{
				if (newG_show.at<uchar>(i, j) >= treshold_high) {
					newG_show.at<uchar>(i, j) = 255;
				}
				else if (newG_show.at<uchar>(i, j) < treshold_high && newG_show.at<uchar>(i, j) > treshold_low) {
					newG_show.at<uchar>(i, j) = 128;
				}
				else if (newG_show.at<uchar>(i, j) = 255 <= treshold_low) {
					newG_show.at<uchar>(i, j) = 0;
				}
			}
		}

		int i_index[8] = { -1, 0, 1, 0, -1, -1, 1, 1 };
		int j_index[8] = { 0, -1, 0, 1, -1, 1, -1, 1 };
		uchar neighbor[8];


		//bfs
		for (int i = 1; i < newG_show.rows - 1; i++) {
			for (int j = 1; j < newG_show.cols - 1; j++) {
				if (newG_show.at<uchar>(i, j) == 255) {
					std::queue<Point2i> Q;
					Q.push(Point2i(i, j));
					while (!Q.empty())
					{
						Point2i q = Q.front();
						Q.pop();

						// check neighbours
						for (int k = 0; k < 8; k++)
							neighbor[k] = newG_show.at<uchar>(q.x + i_index[k], q.y + j_index[k]);

						for (int k = 0; k < 8; k++) {
							if (neighbor[k] == 128)
							{
								newG_show.at<uchar>(q.x + i_index[k], q.y + j_index[k]) = 255;
								Q.push(Point2i(q.x + i_index[k], q.y + j_index[k]));

							}
						}
					}
				}

			}
		}

		// change weak points non-edge points
		for (int i = 1; i < newG_show.rows - 1; i++) {
			for (int j = 1; j < newG_show.cols - 1; j++) {
				if (newG_show.at<uchar>(i, j) == 128) {
					newG_show.at<uchar>(i, j) = 0;
				}
			}
		}

		imshow("final Canny", newG_show);
		waitKey();

	}
}
	
	


int main()
{
	
	canny_gradient();

	
	return 0;
}