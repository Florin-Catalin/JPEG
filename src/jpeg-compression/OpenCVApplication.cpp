// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include "common.h"
#include <queue>

using namespace std;
using namespace cv;

Mat get_image(char* received_image, int opening_type) {
	Mat image;
	image = imread(received_image, opening_type);
	return image;
}

Mat_<Vec3b> YCbCr_to_RGB(const Mat_<Vec3i>& img_ycbcr)
{
	Mat_<Vec3b> img_rgb(img_ycbcr.rows, img_ycbcr.cols);

	Vec3i old_val;
	Vec3b new_val;
	for (int i = 0; i < img_ycbcr.rows; ++i)
	{
		for (int j = 0; j < img_ycbcr.cols; ++j)
		{
			old_val = img_ycbcr(i, j);

			new_val[2] = (uchar)(MIN(255, MAX(0, old_val[0] + 1.402 * (old_val[2] - 128))));
			new_val[1] = (uchar)(MIN(255, MAX(0, old_val[0] - 0.344136 * (old_val[1] - 128) - 0.714136 * (old_val[2] - 128))));
			new_val[0] = (uchar)(MIN(255, MAX(0, old_val[0] + 1.772 * (old_val[1] - 128))));

			img_rgb(i, j) = new_val;
		}
	}

	return img_rgb;
}

void display_image(Mat image) {
	namedWindow("Display Image", WINDOW_AUTOSIZE);
	imshow("Display Image", image);
	waitKey(0);
}
Mat convert_to_YCrCb(Mat image) {
	Mat converted_image = image.clone();
	cvtColor(image, converted_image, CV_RGB2YCrCb);
	return converted_image;
}

Mat convert_to_RGB(Mat image) {
	Mat converted_image = image.clone();
	cvtColor(image, converted_image, CV_YCrCb2RGB);
	return converted_image;
}


 const int ql[8][8]{
		{ 16, 11, 10, 16, 24, 40, 51, 61 },
		{ 12, 12, 14, 19, 26, 58, 60, 55 },
		{ 14, 13, 16, 24, 40, 57, 69, 56 },
		{ 14, 17, 22, 29, 51, 87, 80, 62 },
		{ 18, 22, 37, 56, 68, 109, 103, 77 },
		{ 24, 35, 55, 64, 81, 104, 113, 92 },
		{ 49, 64, 78, 87, 103, 121, 120, 101 },
		{ 72, 92, 95, 98, 112, 100, 103, 99 }
};


 /* 500 KB for parrots
 const int qc[8][8]{
   { 128, 128, 128, 128, 99, 99, 99, 99 },
   { 128, 128, 128,128, 99, 99, 99, 99 },
   { 128, 128, 128, 128, 99, 99, 99, 99 },
   { 128, 128, 128, 128, 99, 99, 99, 99 },
   { 128, 128, 128, 99, 99, 99, 99, 99 },
   { 128, 128, 128, 99, 99, 99, 99, 99 },
   {128, 128, 128, 99, 99, 99, 99, 99 },
   { 128, 128, 128, 99, 99, 99, 99, 99 }
};*/
 const int qc[8][8]{
	{ 17, 18, 24, 47, 99, 99, 99, 99 },
	{ 18, 21, 26, 66, 99, 99, 99, 99 },
	{ 24, 26, 56, 99, 99, 99, 99, 99 },
	{ 47, 66, 99, 99, 99, 99, 99, 99 },
	{ 99, 99, 99, 99, 99, 99, 99, 99 },
	{ 99, 99, 99, 99, 99, 99, 99, 99 },
	{ 99, 99, 99, 99, 99, 99, 99, 99 },
	{ 99, 99, 99, 99, 99, 99, 99, 99 }
};

double cosines(int y, int i, int N)
{
	double arg;
	arg = (((2 * y) + 1)*i*PI) / (2 * N);
	return cos(arg);
}
static Mat_<Vec3i> FCDT(const Mat_<Vec3i>& src)
{
	Mat_<Vec3i> dst(src.rows, src.cols);

	double ny, ncb, ncr, ci, cj;
	for (int br = 0; br < src.rows; br += 8)
	{
		for (int bc = 0; bc < src.cols; bc += 8)
		{
			for (int i = 0; i < 8; ++i)
			{
				for (int j = 0; j < 8; ++j)
				{
					ny = ncb = ncr = 0.0;
					for (int x = 0; x < 8; ++x)
					{
						for (int y = 0; y < 8; ++y)
						{       //  - 128 to convert to signed
							ny += (src(br + y, bc + x)[0] - 128) *   cosines(y,i,8) *cosines(x,j,8);
							ncb += (src(br + y, bc + x)[1]) *  cosines(y, i, 8) * cosines(x, j, 8);
							ncr += (src(br + y, bc + x)[2]) *  cosines(y, i, 8)* cosines(x, j, 8);
						}
					}

					// if i or j =0 then   ci or cj = sqrt ( 1 / N ) else ci or cj  = sqrt(2/N) 
					ci = sqrt(((i == 0) ? 1.0 : 2.0) / 8);
					cj = sqrt(((j == 0) ? 1.0 : 2.0) / 8);

					ny *= ci * cj;
					ncb *= ci * cj;
					ncr *= ci * cj;


					// quantize using quantization matrix
					ny /= ql[i][j];
					ncb /= qc[i][j];
					ncr /= qc[i][j];


					// round each result  - > high freqency will quantize to 0 
					ny = (ny - floor(ny) <= 0.5) ? floor(ny) : ceil(ny);
					ncb = (ncb - floor(ncb) <= 0.5) ? floor(ncb) : ceil(ncb);
					ncr = (ncr - floor(ncr) <= 0.5) ? floor(ncr) : ceil(ncr);

					dst(br + i, bc + j) = Vec3i{ (int)ny, (int)ncb, (int)ncr };
				}
			}
		}
	}

	return dst;
}
static Mat_<Vec3i> ICDT(const Mat_<Vec3i>& src)
{
	Mat_<Vec3i> dst(src.rows, src.cols);
	double ny, ncb, ncr, ci, cj, cy, ccb, ccr;

	for (int br = 0; br < src.rows; br += 8)
	{
		for (int bc = 0; bc < src.cols; bc += 8)
		{
			for (int x = 0; x < 8; ++x)
			{
				for (int y = 0; y < 8; ++y)
				{
					ny = ncb = ncr = 0.0;
					for (int i = 0; i < 8; ++i)
					{
						for (int j = 0; j < 8; ++j)
						{
							ci = sqrt(((i == 0) ? 1.0 : 2.0) / 8);
							cj = sqrt(((j == 0) ? 1.0 : 2.0) / 8);

							cy = src(br + i, bc + j)[0] * ql[i][j];
							ccb = src(br + i, bc + j)[1] * qc[i][j];
							ccr = src(br + i, bc + j)[2] * qc[i][j];
							  
							ny += ci * cj * cy * cosines(y, i, 8) *cosines(x, j, 8);
							ncb += ci * cj * ccb * cosines(y, i, 8) * cosines(x, j, 8);
							ncr += ci * cj * ccr * cosines(y, i, 8) * cosines(x, j, 8);
						}
					}

					// rounding 
					ny = (ny - floor(ny) <= 0.5) ? floor(ny) : ceil(ny);
					ncb = (ncb - floor(ncb) <= 0.5) ? floor(ncb) : ceil(ncb);
					ncr = (ncr - floor(ncr) <= 0.5) ? floor(ncr) : ceil(ncr);
					// + 128 convert to unsigned 
					dst(br + y, bc + x) = Vec3i{ (int)ny + 128, (int)ncb, (int)ncr };
				}
			}
		}
	}

	return dst;
}


static vector<Vec3i> Zigzag(const Mat_<Vec3i>& src, int rate)
{

	
	vector<Vec2i> zz{};

	for (int i = 0; i < 8; ++i)
	{
		for (int j = 0; j <= i; ++j)
		{
			int r = (i % 2 == 1) ? j : i - j;
			int c = (i % 2 == 1) ? i - j : j;

			zz.push_back({ r, c });
		}
	}
	for (int i = 1; i < 8; ++i)
	{
		for (int j = 0; j < 8 - i; ++j)
		{
			int r = (i % 2 == 1) ? 8 - j - 1 : i + j;
			int c = (i % 2 == 1) ? i + j : 8 - j - 1;

			zz.push_back({ r, c });
		}
	}

	vector<Vec3i> res{};
	for (int br = 0; br < src.rows; br += 8)
	{
		for (int bc = 0; bc < src.cols; bc += 8)
		{ 

			// compression ration depending on how large the values in the quantization table are
			for (int i = 0; i < rate ; ++i)
			{
				res.push_back(src(br + zz[i][0], bc + zz[i][1]));
			}
		}
	}

	return res;
}

static Mat_<Vec3i> iZigzag(const vector<Vec3i>& src, int rows, int cols, int quality)
{
	vector<Vec2i> zz{};

	for (int i = 0; i < 8; ++i)
	{
		for (int j = 0; j <= i; ++j)
		{
			int r = (i % 2 == 1) ? j : i - j;
			int c = (i % 2 == 1) ? i - j : j;

			zz.push_back({ r, c });
		}
	}
	for (int i = 1; i < 8; ++i)
	{
		for (int j = 0; j < 8 - i; ++j)
		{
			int r = (i % 2 == 1) ? 8 - j - 1 : i + j;
			int c = (i % 2 == 1) ? i + j : 8 - j - 1;

			zz.push_back({ r, c });
		}
	}
	
	Mat_<Vec3i> res(rows, cols, { 0, 0, 0 });
	for (int br = 0, k = 0; br < rows; br += 8)
	{
		for (int bc = 0; bc < cols; bc += 8)
		{
			for (int i = 0; i < quality; ++i, ++k)
			{
				res(br + zz[i][0], bc + zz[i][1]) = src[k];
			}
		}
	}

	return res;
}

void WriteJPEG(vector<Vec3i> data, int rows, int cols, char *path, int quality)
{
	//write the JPEG compressed bitstream 
	ofstream fout(path, ios::binary);

	fout.write((char *)(&rows), sizeof(rows));
	fout.write((char *)(&cols), sizeof(cols));
	fout.write((char *)(&quality), sizeof(quality));

	for (int i = 0; i < data.size(); ++i)
	{
		Vec3b px{ (uchar)data[i][0], (uchar)data[i][1], (uchar)data[i][2] };
		fout.write((char *)(&px), sizeof(px));
	}

	fout.close();
}

static vector<Vec3i> ReadJPEG(char *path, int& rows, int& cols, int& quality)
{
	ifstream fin(path, ios::binary);

	fin.read((char *)(&rows), sizeof(rows));
	fin.read((char *)(&cols), sizeof(cols));
	fin.read((char *)(&quality), sizeof(quality));

	int partitions = (rows / 8) * (cols / 8);
	vector<Vec3i> data(partitions * 64);

	for (int i = 0, k = 0; i < partitions; ++i)
	{
		for (int j = 0; j < quality; ++j, ++k)
		{
			Vec3b px;
			fin.read((char *)(&px), sizeof(px));
			data[k] = { (char)px[0], (char)px[1], (char)px[2] };
		}
	}

	return data;
}

void encoder(char *src, char *dst, int rate)
{
	
	Mat_<Vec3b> img = imread(src), img_resize;
	resize(img, img_resize, { (int)(ceil(img.cols / 8.0) * 8), (int)(ceil(img.rows / 8.0) * 8) });

	Mat img_ycbcr = convert_to_YCrCb(img_resize);
	Mat temp = FCDT(img_ycbcr);
	vector<Vec3i> jpeg_data = Zigzag(temp, rate);
	WriteJPEG(jpeg_data, temp.rows, temp.cols, dst, rate);

}

void decoder(char *src, char *dst)
{
	//decoding pipeline
	int rows, cols, rate;
	vector<Vec3i> jpegData = ReadJPEG(src, rows, cols, rate);
	Mat temp = iZigzag(jpegData, rows, cols, rate);
	Mat img_ycbcr = ICDT(temp);
	Mat img = YCbCr_to_RGB(img_ycbcr);

	imwrite(dst, img);
}

int main()
{
	char fname[MAX_PATH];
	char out[MAX_PATH];
	if (openFileDlg(fname)) {
		
		display_image(convert_to_RGB(convert_to_YCrCb(get_image(fname, CV_LOAD_IMAGE_COLOR))));
		
	}
	if (openFileDlg(out)) {
		encoder(fname,out , 4);
		decoder(out, out);
	}
	waitKey(0);
	
	
	return 0;
}