#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
using namespace cv;

#define PI 3.1415926535
#define ORIENTBIN 9
#define TAU  10
#define HOGPICSCALING 30
#define CELL_SIZE 10


double *** compute_HoG(const cv::Mat& img, const int cell_size, std::vector<int> &dims);
void visulizeThatHoG(double ***featArray, vector<int> dims, int cell_size);
Mat SobelX2(Mat img);
Mat SobelY2(Mat img);
Mat gradDirection2(Mat img);
Mat gradMagnitude2(Mat img, bool showMagnitude);

/*int main() {
	Mat img = imread("eye.png", CV_LOAD_IMAGE_GRAYSCALE);

	std::vector<int> vek;
	double *** featarray = compute_HoG(img, CELL_SIZE, vek);
	visulizeThatHoG(featarray, vek, CELL_SIZE);


}*/



double *** compute_HoG(const cv::Mat& img, const int cell_size, std::vector<int> &dims) {
	Mat magnitudes = gradMagnitude2(img, false);
	Mat direct = gradDirection2(img);

	//directions auf 0-pi;
	for (int y = 0; y < direct.rows; y++) {
		for (int x = 0; x < direct.cols; x++) {
			if (direct.at<double>(y, x) < 0)
				direct.at<double>(y, x) += PI;
		}
	}

	double ***hog = (double***)calloc((img.rows / cell_size), sizeof(double**));
	if (hog == NULL)
		return NULL;
	for (int y = 0; y < img.rows / cell_size; y++) {
		hog[y] = (double**)calloc((img.cols / cell_size), sizeof(double*));
		if (hog[y] == NULL) {
			for (int k = 0; k < y; k++) {
				free(hog[k]);
				free(hog);

			}
			free(hog);
			return NULL;
		}
		for (int x = 0; x < img.cols / cell_size; x++) {
			hog[y][x] = (double*)calloc(ORIENTBIN, sizeof(double));
			if (hog[y][x] == NULL) {
				for (int j = 0; j < x; j++) {
					free(hog[y][j]);
				}
				free(hog[y]);
				for (int k = 0; k < y; k++) {
					for (int j = 0; j < img.cols / cell_size; j++) {
						free(hog[k][j]);
					}
					free(hog[k]);
				}
				free(hog);
				return NULL;
			}
		}
	}
	dims.push_back((img.rows / cell_size));
	dims.push_back((img.cols / cell_size));
	dims.push_back(ORIENTBIN);
	int yc = 0;
	int xc = 0;
	for (int y = 0; y + cell_size < img.rows; y += cell_size) {
		xc = 0;
		for (int x = 0; x + cell_size < img.cols; x += cell_size) {
			for (int dy = 0; dy < cell_size; dy++) {
				for (int dx = 0; dx < cell_size; dx++) {
					double calcbin = direct.at<double>(y + dy, x + dx);
					calcbin /= (PI + 0.01);
					calcbin *= ORIENTBIN;
					double delta = calcbin - (int)calcbin;
					hog[y / cell_size][x / cell_size][(int)calcbin] += (1 - delta)*magnitudes.at<int>(y + dy, x + dx);
					hog[y / cell_size][x / cell_size][((int)calcbin + 1) % ORIENTBIN] += (delta)*magnitudes.at<int>(y + dy, x + dx);
				}
			}
			xc++;
		}
		yc++;
	}

	return hog;
}

void visulizeThatHoG(double ***featArray, vector<int> dims, int cell_size) {
	int dimy = dims.at(0);
	int dimx = dims.at(1);
	int orients = dims.at(2);
	int min = 10000;
	int max = 0;


	Mat hogPic = Mat::zeros(dimy * HOGPICSCALING, dimx * HOGPICSCALING, CV_8UC1);




	//normalisieren des HOG auf [Tau,255]
	for (int y = 0; y < dimy; y++) {
		for (int x = 0; x < dimx; x++) {
			for (int i = 0; i < ORIENTBIN; i++) {
				if (featArray[y][x][i] < min)
					min = (int)featArray[y][x][i];
				if (featArray[y][x][i] > max)
					max = (int)featArray[y][x][i];
			}
			for (int i = 0; i < ORIENTBIN; i++) {
				double wert = featArray[y][x][i];
				wert -= min;
				wert /= (max - min);
				wert *= (255 - TAU) + TAU;
				featArray[y][x][i] = wert;
			}
			max = 0;
			min = 10000;
		}
	}

	for (int y = 0; y < dimy*HOGPICSCALING; y += HOGPICSCALING) {
		for (int x = 0; x < dimx*HOGPICSCALING; x += HOGPICSCALING) {
			for (int i = 0; i < ORIENTBIN; i++) {
				double bin = i*(PI / ORIENTBIN);
				//							laenge der striche
				int dx = (int)round(cos(bin) * (cell_size*0.9*(HOGPICSCALING / (double)cell_size)) / 2);
				int dy = (int)round(sin(bin) * (cell_size*0.9*(HOGPICSCALING / (double)cell_size)) / 2);
				int xmid = x + HOGPICSCALING / 2;
				int ymid = y + HOGPICSCALING / 2;
				int xplus = xmid + dx;
				int xminu = xmid - dx;
				int yplus = ymid + dy;
				int yminu = ymid - dy;
				Point mid(xmid, ymid);
				Point plus(xplus, yplus);
				Point minus(xminu, yminu);
				uchar gray = (uchar)featArray[y / HOGPICSCALING][x / HOGPICSCALING][i];
				Scalar Color(gray, gray, gray);
				cv::line(hogPic, mid, plus, Color);
				cv::line(hogPic, mid, minus, Color);

			}
		}
	}

	imshow("HOG", hogPic);
	imwrite("hog.jpg", hogPic);
	waitKey();
	destroyAllWindows();






}

Mat SobelX2(Mat img) {
	Mat sobelx = Mat::zeros(3, 3, CV_8SC1);
	Mat target = Mat::zeros(img.size(), CV_32S);



	sobelx.at<schar>(0, 0) = -1;
	sobelx.at<schar>(1, 0) = -2;
	sobelx.at<schar>(2, 0) = -1;
	sobelx.at<schar>(0, 2) = 1;
	sobelx.at<schar>(1, 2) = +2;
	sobelx.at<schar>(2, 2) = 1;
	int sobelvalue;
	for (int y = 1; y < img.rows - 1; y++) {
		int *targetptr = target.ptr<int>(y);
		for (int x = 1; x < img.cols - 1; x++) {
			sobelvalue = 0;
			for (int dy = -1; dy <= 1; dy++) {
				uchar *imgptr = img.ptr<uchar>(y + dy);
				schar *sobelptr = sobelx.ptr<schar>(dy + 1);
				for (int dx = -1; dx <= 1; dx++) {
					sobelvalue += imgptr[x + dx] * sobelptr[dx + 1];
				}
			}



			targetptr[x] = (int)sobelvalue;
		}
	}




	return target;
}

Mat SobelY2(Mat img) {
	Mat sobely = Mat::zeros(3, 3, CV_8SC1);
	Mat target = Mat::zeros(img.size(), CV_32S);


	sobely.at<schar>(0, 0) = -1;
	sobely.at<schar>(0, 1) = -2;
	sobely.at<schar>(0, 2) = -1;
	sobely.at<schar>(2, 0) = 1;
	sobely.at<schar>(2, 1) = +2;
	sobely.at<schar>(2, 2) = 1;
	int sobelvalue;
	for (int y = 1; y < img.rows - 1; y++) {
		int *targetptr = target.ptr<int>(y);
		for (int x = 1; x < img.cols - 1; x++) {
			sobelvalue = 0;
			for (int dy = -1; dy <= 1; dy++) {
				uchar *imgptr = img.ptr<uchar>(y + dy);
				schar *sobelptr = sobely.ptr<schar>(dy + 1);
				for (int dx = -1; dx <= 1; dx++) {
					sobelvalue += imgptr[x + dx] * sobelptr[dx + 1];
				}
			}


			//std::cout << medianfilt[(kernelSize*kernelSize) / 2] << "median" << std::endl;
			targetptr[x] = sobelvalue;
		}
	}



	return target;
}

Mat gradMagnitude2(Mat img, bool showMagnitude) {
	Mat sobelx = SobelX2(img);
	Mat sobely = SobelY2(img);
	Mat magnitude = Mat::zeros(img.size(), CV_32SC1);
	Mat magnitudeNormalized = Mat::zeros(img.size(), CV_8UC1);
	if (img.channels() == 3)
		cvtColor(img, img, CV_BGR2GRAY, 1);
	int min = 1000, max = 0;

	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			int mag = abs(sobelx.at<int>(y, x)) + abs(sobely.at<int>(y, x));
			magnitude.at<int>(y, x) = mag;
			if (mag > max)
				max = mag;
			if (mag < min)
				min = mag;
		}
	}

	if (showMagnitude) {

		for (int y = 0; y < img.rows; y++) {
			for (int x = 0; x < img.cols; x++) {
				double wert = magnitude.at<int>(y, x);
				wert -= min;
				wert /= (max - min);
				wert *= 255;
				magnitudeNormalized.at<uchar>(y, x) = (uchar)wert;

			}
		}

		imshow("magnitude", magnitudeNormalized);
		imwrite("magnitude.jpg", magnitudeNormalized);
		waitKey();
		destroyAllWindows();
	}
	return magnitude;
}

Mat gradDirection2(Mat img) {

	Mat sobelx = SobelX2(img);
	Mat sobely = SobelY2(img);
	Mat directions = Mat::zeros(img.size(), CV_64FC1);
	double wert;
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			wert = atan2(sobely.at<int>(y, x), sobelx.at<int>(y, x));
			//werte zwischen -pi und pi
			directions.at<double>(y, x) = wert;

		}
	}
	return directions;
}
