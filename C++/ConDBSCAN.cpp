#include "ConDBSCAN.h"


/**************************************private**************************************/

const int ConDBSCAN::Point3dToInt(Point3i point) {
	return 1000000 * point.x + 1000 * point.y + point.z;
}

/**
 *
 */
void ConDBSCAN::ConvolutionX(Matrix2D& data, Matrix2D& localSum, int radius) {
	int rows = data.size();
	int cols = data[0].size();

	localSum.assign(rows, vector<int>(cols, 0));
	// 初始化，计算第0行local_sum_0
	for (int i = 0; i <= radius; ++i) {
		for (int j = 0; j < cols; ++j) {
			localSum[0][j] += data[i][j];
		}
	}
	// 根据公式迭代求解所有的local_sum_i
	for (int i = 1; i < rows; ++i) {
		if (i <= radius) {
			for (int j = 0; j < cols; ++j) {
				localSum[i][j] = localSum[i - 1][j] + data[i + radius][j];
			}
		} else if (i < rows - radius){
			for (int j = 0; j < cols; ++j) {
				localSum[i][j] = localSum[i - 1][j] + data[i + radius][j] - data[i - radius - 1][j];
			}
		} else {
			for (int j = 0; j < cols; ++j) {
				localSum[i][j] = localSum[i - 1][j] - data[i - radius - 1][j];
			}
		}
	}

}

/**
 *
 */
void ConDBSCAN::ConvolutionY(Matrix2D& data, Matrix2D& localSum, int radius) {
	int rows = data.size();
	int cols = data[0].size();

	localSum.assign(rows, vector<int>(cols, 0));
	// 初始化，计算第0列local_sum_0
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j <= radius; ++j) {
			localSum[i][0] += data[i][j];
		}
	}
	// 根据公式迭代求解所有的local_sum_i
	for (int i = 0; i < rows; ++i) {
		for (int j = 1; j < cols; ++j) {
			if (j <= radius) {
				localSum[i][j] = localSum[i][j - 1] + data[i][j + radius];
			}
			else if (j < rows - radius) {
				localSum[i][j] = localSum[i][j - 1] + data[i][j + radius] - data[i][j - radius - 1];
			}
			else {
				localSum[i][j] = localSum[i][j - 1] - data[i][j - radius - 1];
			}
		}
	}
}

/**
 *
 */
void ConDBSCAN::QuickConvolute2D(Matrix2D& data, Matrix2D& conResult2D) {
	Matrix2D localSum;
	ConvolutionX(data, localSum, kernelRadius.x);
	ConvolutionY(localSum, conResult2D, kernelRadius.y);
}

/**
 *
 */
void ConDBSCAN::QuickConvolute3D(Matrix3D& data, Matrix3D& conResult3D) {
	int r = data.size(), c = data[0].size(), h = data[0][0].size();
	// 首先对每一层进行二维卷积
	for (int i = 0; i < r; ++i) {
		QuickConvolute2D(data[i], conResult3D[i]);
	}
	// 对二维卷积后的矩阵换一个维度再次进行卷积，得到三维卷积结果
	for (int j = 0; j < c; ++j) {
		Matrix2D temp(r), conResult2DY(r, vector<int>(h, 0));
		for (int i = 0; i < r; ++i) {
			temp[i] = conResult3D[i][j];
		}
		ConvolutionX(temp, conResult2DY, kernelRadius.z);
		for (int i = 0; i < r; ++i) {
			conResult3D[i][j] = conResult2DY[i];
		}
	}
}

/**
 *
 */
void ConDBSCAN::GetCoreObjects(Matrix3D& conResult3D, Matrix3D& colorMatrix, map<int, bool>& coreObjects) {
	for (int i = 0; i < 256; ++i) {
		for (int j = 0; j < 256; ++j) {
			for (int k = 0; k < 256; ++k) {
				if (conResult3D[i][j][k] > minpts && colorMatrix[i][j][k] != 0) {
					coreObjects[Point3dToInt(Point3i(i, j, k))] = false;
				}
			}
		}
	}
}

/**
 * [GetDistance description]
 * @param  p1               [description]
 * @param  p2               [description]
 * @return    [description]
 */
double ConDBSCAN::GetDistance(Point3d p1, Point3d p2) {
	return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
}

/**************************************public**************************************/

// 构造函数
ConDBSCAN::ConDBSCAN(const Point3i kernel, const int minpts) {
	this->kernel = kernel;
	this->minpts = minpts;
	this->kernelRadius = {int(ceil((kernel.x - 1) / 2)), int(ceil((kernel.y - 1) / 2)), int(ceil((kernel.z - 1) / 2))};
}

/**
 *
 */
void ConDBSCAN::ImageTo3DMatrix(Mat& img, Matrix3D& colorMatrix, map<int, int>& dealed) {
	int rows = img.rows;
	int cols = img.cols * img.channels();
	for (int r = 0; r < rows; ++r) {
		uchar* linePixels = img.ptr<uchar>(r);
		for (int c = 0; c < cols; c += img.channels()) {
			int x = linePixels[c], y = linePixels[c + 1], z = linePixels[c + 2];
			colorMatrix[x][y][z] += 1;
			dealed[Point3dToInt(Point3i(x, y, z))] = -1;
		}
	}
}

/**
 *
 */
void ConDBSCAN::Fit(Matrix3D& colorMatrix, map<int, int>& dealed, set<int>& labels) {
	clock_t start, end;
	start = clock();
	Matrix3D conResult3D(256, Matrix2D(256, vector<int>(256, 0)));
	QuickConvolute3D(colorMatrix, conResult3D);
	map<int, bool> coreObjects;
	GetCoreObjects(conResult3D, colorMatrix, coreObjects);

	std::cout << "核心对象个数： " << coreObjects.size() << endl;
	end = clock();
	std::cout << "卷积时间 time = " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;

	queue<Point3i> visits;
	int tag = 0;
	for (const auto core : coreObjects) {
		// 当前对象已访问，跳过
		if (core.second) continue;
		// 解析坐标
		int x = core.first / 1000000, y = core.first % 1000000 / 1000, z = core.first % 1000;
		visits.push(Point3i(x, y, z));
		coreObjects[core.first] = true;
		Point3i last(0, 0, 0);
		while (!visits.empty()) {
			Point3i now = visits.front();
			visits.pop();
			dealed[Point3dToInt(now)] = tag;
			for (int i = now.x - kernelRadius.x; i <= now.x + kernelRadius.x; ++i) {
				for (int j = now.y - kernelRadius.y; j <= now.y + kernelRadius.y; ++j) {
					for (int k = now.z - kernelRadius.z; k <= now.z + kernelRadius.z; ++k) {
						// 跳过越界索引
						if (i < 0 || j < 0 || k < 0 || i > 255 || j > 255 || k > 255) continue;
						// 跳过已访问对象
						if (i <= last.x && j <= last.y && k <= last.z) continue;
						// 如果是样本点，并且未访问
						int key = i * 1000000 + j * 1000 + k;
						if (dealed[key] == -1) {
							dealed[key] = tag;
							if (coreObjects.find(key) != coreObjects.end() && !coreObjects[key]) {
								visits.push(Point3i(i, j, k));
								coreObjects[key] = true;
							}
						}
					}
				}
			}
			last = now;
		}
		labels.insert(tag);
		tag++;
	}
}

/**
 *
 */
vector<Mat> ConDBSCAN::ClusterToMutiImage(set<int>& labels, map<int, int>& cluster, Mat& img) {
	vector<Mat> segementations;
	for (int i = 0; i < labels.size(); ++i) {
		Mat empty(img.size(), CV_8UC1, Scalar(255));
		segementations.push_back(empty);
	}
	int rows = img.rows;
	int cols = img.cols;
	for (int r = 0; r < rows; ++r) {
		for (int c = 0; c < cols; c++) {
			Vec3b color = img.at<Vec3b>(r, c);
			int key = Point3dToInt(Point3i(color[0], color[1], color[2]));
			auto iter = cluster.find(key);
			if (iter == cluster.end()) continue;
			int label = iter->second;
			if (label == -1) continue;
			segementations[label].at<uchar>(r, c) = 0;
		}
	}
	return segementations;
}

/**
 *
 */
Mat ConDBSCAN::ClusterToSingleImage(set<int>& labels, map<int, int>& cluster, Mat& img) {
	Mat segementation(img.size(), CV_8UC3, Scalar(255, 255, 255));
	int rows = img.rows;
	int cols = img.cols;
	//vector<Scalar> colors({ Scalar(166, 127, 120), Scalar(50, 67, 95), Scalar(143, 134, 129), Scalar(225, 220, 217) });
	//vector<Scalar> colors({  Scalar(166, 127, 120),  Scalar(145, 92, 76), Scalar(47, 24, 18) , Scalar(64, 104, 106) });
	for (int r = 0; r < rows; ++r) {
		for (int c = 0; c < cols; c++) {
			Vec3b color = img.at<Vec3b>(r, c);
			int key = Point3dToInt(Point3i(color[0], color[1], color[2]));
			auto iter = cluster.find(key);
			if (iter == cluster.end()) continue;
			int label = iter->second;
			// 噪声点
			if (label == -1) segementation.at<Vec3b>(r, c) = Vec3b(153, 136, 139);
			if (label >= colors.size()) continue;
			segementation.at<Vec3b>(r, c) = Vec3b(colors[label+0][2], colors[label+0][1], colors[label+0][0]);
		}
	}
	return segementation;
}

/**
 * [CalSilhouette description]
 * @param  results               [description]
 * @param  origin                [description]
 * @param  num                   [description]
 * @return         [description]
 */
double ConDBSCAN::CalSilhouette(map<int, int>& results, unordered_map<int, int>& origin, int num) {
	vector<vector<Point3d> > clusters(num);
	int total = 0;
	for (auto& res : results) {
		int x = res.first / 1000000, y = res.first % 1000000 / 1000, z = res.first % 1000;
		if (res.second != -1 && origin.find(res.first) != origin.end()) {
			clusters[res.second].emplace_back(Point3d(x, y, z));
			total += origin[res.first];
		}
	}
	cout << total << endl;

	double si = 0.0;
	for (int i = 0; i < num; ++i) {
		if (clusters[i].size() == 1) {
			si += 1 * origin[Point3dToInt(clusters[i][0])];
			continue;
		}
		for (const Point3d& p : clusters[i]) {
			double b_score = 0.0, a_score = 0.0;
			int n = 0;
			for (const Point3d& x : clusters[i]) {
				if (p != x) {
					a_score += GetDistance(p, x) * origin[Point3dToInt(x)];
					n += origin[Point3dToInt(x)];
				}
			}
			a_score /= n;

			n = 0;
			for (int j = 0; j < num; ++j) {
				if (j != i) {
					for (const Point3d& y : clusters[j]) {
						b_score += GetDistance(p, y) * origin[Point3dToInt(y)];
						n += origin[Point3dToInt(y)];
					}
				}
			}
			b_score /= n;

			si += (b_score - a_score) * origin[Point3dToInt(p)] / max(b_score, a_score);
		}
	}

	return si / total;
}
