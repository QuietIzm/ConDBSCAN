// coding: utf-8

#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

#include<iostream>
#include<iomanip>

#include<cmath>
#include<ctime>

#include<set>
#include<map>
#include<queue>
#include<vector>
#include<unordered_map>

using namespace cv;
using namespace std;

typedef vector<vector<int>> Matrix2D;
typedef vector<vector<vector<int>>> Matrix3D;

// 颜色列表
const vector<Scalar> colors({ Scalar(32, 178, 170), Scalar(240, 248, 255), Scalar(135, 206, 250), Scalar(138, 43, 226), Scalar(222, 184, 135), Scalar(165, 42, 42),
						Scalar(60, 170, 113), Scalar(255, 140, 0), Scalar(173, 255, 47), Scalar(0, 128, 128), Scalar(166, 127, 120), Scalar(50, 67, 95),
						Scalar(143, 134, 129), Scalar(225, 220, 217), Scalar(166, 127, 120),  Scalar(145, 92, 76), Scalar(47, 24, 18) , Scalar(64, 104, 106),
						Scalar(250, 240, 230), Scalar(255, 228, 225), Scalar(100, 149, 237), Scalar(176, 196, 222), Scalar(46, 139, 87), Scalar(255, 174, 185),
						Scalar(139, 58, 98), Scalar(193, 205, 205), Scalar(99, 184, 255), Scalar(141, 182, 205), Scalar(171, 130, 255), Scalar(139, 34, 82),
						Scalar(139, 71, 93), Scalar(205, 179, 139), Scalar(255, 222, 173), Scalar(255, 99, 71), Scalar(205, 91, 69), Scalar(255, 165, 0), Scalar(139, 87, 66),
						Scalar(205, 102, 29), Scalar(238, 154, 73), Scalar(139, 126, 102), Scalar(255, 231, 186), Scalar(238, 197, 145), Scalar(238, 121, 66),
						Scalar(139, 101, 8), Scalar(238, 173, 14), Scalar(139, 105, 20), Scalar(238, 238, 0), Scalar(205, 205, 180), Scalar(205, 190, 112),
						Scalar(162, 205, 90), Scalar(105, 139, 34), Scalar(192, 255, 62) });

class ConDBSCAN {
private:
	Point3i kernel;
	int minpts;
	Point3i kernelRadius;

	const int Point3dToInt(Point3i point);
	void ConvolutionX(Matrix2D& data, Matrix2D& localSum, int radius);
	void ConvolutionY(Matrix2D& data, Matrix2D& localSum, int radius);
	void QuickConvolute2D(Matrix2D& data, Matrix2D& conResult2D);
	void QuickConvolute3D(Matrix3D& data, Matrix3D& conResult3D);
	void GetCoreObjects(Matrix3D& conResult3D, Matrix3D& colorMatrix, map<int, bool>& coreObjects);
	double GetDistance(Point3d p1, Point3d p2);

public:
	ConDBSCAN(const Point3i kernel, const int minpts);
	void ImageTo3DMatrix(Mat& img, Matrix3D& colorMatrix, map<int, int>& dealed);
	void Fit(Matrix3D& colorMatrix, map<int, int>& dealed, set<int>& labels);
	vector<Mat> ClusterToMutiImage(set<int>& labels, map<int, int>& cluster, Mat& img);
	Mat ClusterToSingleImage(set<int>& labels, map<int, int>& cluster, Mat& img);
	double CalSilhouette(map<int, int>& results, unordered_map<int, int>& origin, int num);
};
