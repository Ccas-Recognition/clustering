#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
string bgfilename = "E:/proj/clustering/bg.txt", fgfilename = "E:/proj/clustering/fg.txt", bgpath = "E:/images/bg/", fgpath = "E:/images/fg/";

int main() {
	vector<Mat> fgMats;
	ifstream f(fgfilename), b(bgfilename);
	int fgcount = 0, bgcount = 0;
	string str;
	while (!f.eof())
	{
		f >> str;
		fgcount++;
	}
	while (!b.eof())
	{
		b >> str;
		bgcount++;
	}
	Mat p = Mat::zeros(fgcount, 24*24, CV_32F);
	Mat bestLabels, centers, clustered;
	int ind = 0;
	ifstream fgfile(fgfilename), bgfile(fgfilename);
	while (!fgfile.eof())
	{
		string imagepath;
		fgfile >> imagepath;
		imagepath = fgpath + imagepath;
		Mat _image = imread(imagepath, 0);
		Mat image;
		resize(_image, image, Size(24, 24));
		fgMats.push_back(image);
		for (int y = 0; y < image.rows; ++y)
			for (int x = 0; x < image.cols; ++x)
				p.at<float>(ind, y * 24 + x) = float(image.at<uchar>(y, x));
		ind++; 
	}
	int K = 4;
	cv::kmeans(p, K, bestLabels,
		TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 500, 1.0),
		3, KMEANS_PP_CENTERS,centers);
	vector<int> counts(K,0);
	for (int i = 0; i < p.rows; ++i)
	{
		
		counts[bestLabels.at<int>(i, 0)]++;
		stringstream ss;
		ss << (bestLabels.at<int>(i, 0) + 1) << "/" << i << ".png";
		imwrite(ss.str(), fgMats[i]);
	}
	for (int i = 0; i < K; ++i) {
		Mat _image(24, 24, CV_8U);
		for (int y = 0; y < 24; ++y)
			for (int x = 0; x < 24; ++x)
				_image.at<uchar>(y, x) = uchar(centers.at<float>(i, y * 24 + x));
		imshow("data", _image);
		waitKey();
	}
	imwrite("data.png", p);
	waitKey();
		//imshow("image", image);
		//waitKey();
}
