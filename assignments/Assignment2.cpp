#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>
#include <cmath>
#include <iostream>
using namespace cv;
#define PI 3.1415926535
#define LENGTH 60
#define THETA 0.05
Mat createJahnePattern(int width, int heigth);
void filterFirstQuadrant(int kernelSize, int h, int w, double variance);
void filterThreeQuadrants(int kernelSize, int w, int h, double variance); 
void filterWithOpenCV(int w, int h, int kernelSize, int sigma);
Mat SobelY(Mat img);
Mat SobelX(Mat img);
Mat gradMagnitude(Mat img, bool showMagnitude);
Mat gradDirection(Mat img);
void visualizeGradients(Mat img);


/*int main() {
	//createJahnePattern(512, 512);
	filterFirstQuadrant(11, 512, 512, 2.0);
	filterThreeQuadrants(11, 512, 512, 2.0);
	filterWithOpenCV(512, 512, 11, 2);
	Mat img1 = imread("Testimage_gradients.jpg", 0);
	Mat img2 = imread("Testimage_gradients.jpg", 1);
	Mat img3 = imread("lena_std.tif", 1);
	Mat img4 = imread("lena_std.tif", 0);
	//gradMagnitude(img, true);
	gradMagnitude(img1, true);
	visualizeGradients(img2);
}*/

//Aufgabe1

Mat createJahnePattern(int width, int heigth) {
	double min = 255, max = 0;
	float A = 127;
	int o = 127;
	Mat calcMat = Mat::zeros(heigth, width, CV_8UC1);
	Mat pat = Mat::zeros(heigth, width, CV_8UC1);
	for (int y = 0; y < heigth; y++) {
		uchar* ptr = calcMat.ptr<uchar>(y);
		for (int x = 0; x < width; x++) {
			double xr = x - ((width - 1) / 2);
			double yr = y - ((heigth - 1) / 2);
			double oben = (pow(xr, 2) + pow(yr, 2));
			double bruch = (oben / (double)heigth);
			double zws = (1.0 / 2.0) * PI * bruch;
			double sinus = sin(zws);
			ptr[x] =(uchar)( A * sinus + o);
		}
	}
	//imshow("Pattern", calcMat);
	imwrite("jahnePattern.jpg",calcMat);
	waitKey();
	destroyAllWindows();
	return calcMat;
}
//wenden jeden filter einzeln auf den ersten quadranten an und zeigt das Resultat
void filterFirstQuadrant(int kernelSize, int h, int w, double variance) {
	//filter box und gauss
	Mat jahne = createJahnePattern(w, h);
	Mat median = createJahnePattern(w, h);
	Mat box = createJahnePattern(w, h);
	Mat gauss = createJahnePattern(w, h);
	Mat gaussFilter = Mat::zeros(kernelSize, kernelSize, CV_64F);
	int *medianfilt = (int*)calloc(kernelSize*kernelSize, sizeof(int));
	for (int y = 0; y < kernelSize; y++) {
		double *gauss = gaussFilter.ptr<double>(y);
		for (int x = 0; x < kernelSize; x++) {
			double bae = 1.0 / (2 * PI*  pow(variance, 2));
			double expo = (-(pow(x - (kernelSize / 2), 2) + (pow(y - (kernelSize / 2), 2)))) / (2 * pow(variance, 2));
			gauss[x] = bae * exp(expo);
		}
	}
	double boxvalue;
	double gaussvalue;
	int c = 0;
	for (int y = 0 + kernelSize / 2; y < h / 2; y++) {
		uchar *gaussptr1 = gauss.ptr<uchar>(y);
		uchar *medianptr1 = median.ptr<uchar>(y);
		uchar *boxptr1 = box.ptr<uchar>(y);
		for (int x = 0 + kernelSize / 2; x < w / 2; x++) {
			boxvalue = 0;
			gaussvalue = 0;
			c = 0;
			for (int dy = -kernelSize / 2; dy <= kernelSize / 2; dy++) {
				uchar *jahptr = jahne.ptr<uchar>(y + dy);
				double *gaussptr = gaussFilter.ptr<double>(dy + kernelSize / 2);
				for (int dx = -kernelSize / 2; dx <= kernelSize / 2; dx++) {
					boxvalue += jahptr[x + dx];
					medianfilt[c] = jahptr[x + dx];
					gaussvalue += jahptr[x + dx] * gaussptr[dx + kernelSize / 2];
					c++;
				}
			}
			std::sort(medianfilt,medianfilt + kernelSize*kernelSize);
			//std::cout << medianfilt[(kernelSize*kernelSize) / 2] << "median" << std::endl;
			medianptr1[x] = medianfilt[(kernelSize*kernelSize) / 2];
			boxptr1[x] = boxvalue / (kernelSize*kernelSize);
			gaussptr1[x] = (uchar)gaussvalue;
		}
	}
	imshow("boxfilter", box);
	imshow("median", median);
	imshow("Gauss", gauss);
	imwrite("box.jpg", box);
	imwrite("median.jpg", median);
	imwrite("Gauss.jpg", gauss);

	waitKey();
	destroyAllWindows();
}

void filterThreeQuadrants(int kernelSize, int w, int h, double variance) {
	//filter box und gauss
	Mat jahne = createJahnePattern(w, h);
	Mat filterd = createJahnePattern(w, h);
	Mat gaussFilter = Mat::zeros(kernelSize, kernelSize, CV_64F);
	int *medianfilt = (int*)calloc(kernelSize*kernelSize, sizeof(int));
	for (int y = 0; y < kernelSize; y++) {
		double *gauss = gaussFilter.ptr<double>(y);
		for (int x = 0; x < kernelSize; x++) {
			double bae = 1.0 / (2 * PI*  pow(variance, 2));
			double expo = (-(pow(x - (kernelSize / 2), 2) + (pow(y - (kernelSize / 2), 2)))) / (2 * pow(variance, 2));
			gauss[x] = bae * exp(expo);
		}
	}


	double boxvalue;
	double gaussvalue;
	int c = 0;
	//Quadranten
	//  1 | 2
	// --- ---
	//  4 | 3
	//Boxfilter im 1 Quadrant
	for (int y = 0 + kernelSize / 2; y < h / 2; y++) {
		uchar *boxptr1 = filterd.ptr<uchar>(y);
		for (int x = 0 + kernelSize / 2; x < w / 2; x++) {
			boxvalue = 0;
			for (int dy = -kernelSize / 2; dy <= kernelSize / 2; dy++) {
				uchar *jahptr = jahne.ptr<uchar>(y + dy);
				double *gaussptr = gaussFilter.ptr<double>(dy + kernelSize / 2);
				for (int dx = -kernelSize / 2; dx <= kernelSize / 2; dx++) {
					boxvalue += jahptr[x + dx];
				}
			}
			boxptr1[x] = boxvalue / (kernelSize*kernelSize);
		}
	}

	//median im 2
	for (int y = 0 + kernelSize / 2; y < h / 2; y++) {
		uchar *medi = filterd.ptr<uchar>(y);
		for (int x = w/2; x < w - kernelSize/2; x++) {
			c = 0;
			for (int dy = -kernelSize / 2; dy <= kernelSize / 2; dy++) {
				uchar *jahptr = jahne.ptr<uchar>(y + dy);
				double *gaussptr = gaussFilter.ptr<double>(dy + kernelSize / 2);
				for (int dx = -kernelSize / 2; dx <= kernelSize / 2; dx++) {
					medianfilt[c] = jahptr[x + dx];
					c++;
				}
			}
			std::sort(medianfilt, medianfilt + kernelSize*kernelSize);
			//std::cout << medianfilt[(kernelSize*kernelSize) / 2] << "median" << std::endl;
			medi[x] = medianfilt[(kernelSize*kernelSize) / 2];
		}
	}
	// Gauss
	for (int y = h / 2; y < h - kernelSize / 2; y++) {
		uchar *gausi = filterd.ptr<uchar>(y);
		for (int x = w / 2; x < w - kernelSize/2; x++) {
			gaussvalue = 0;
			for (int dy = -kernelSize / 2; dy <= kernelSize / 2; dy++) {
				uchar *jahptr = jahne.ptr<uchar>(y + dy);
				double *gaussptr = gaussFilter.ptr<double>(dy + kernelSize / 2);
				for (int dx = -kernelSize / 2; dx <= kernelSize / 2; dx++) {
					gaussvalue += jahptr[x + dx] * gaussptr[dx + kernelSize / 2];
				}
			}
			//std::cout << medianfilt[(kernelSize*kernelSize) / 2] << "median" << std::endl;
			gausi[x] = (uchar)gaussvalue;
		}
	}



	imshow("Filterd", filterd);
	imwrite("FilterdWithHands.jpg", filterd);
	waitKey();
	destroyAllWindows();
}

void filterWithOpenCV(int w, int h, int kernelSize, int sigma) {
	Mat jahne = createJahnePattern(w, h);
	Mat GaussPart = Mat::zeros(h / 2, w / 2, CV_8UC1);
	Mat BoxPart = Mat::zeros(h / 2, w / 2, CV_8UC1);
	Mat MedianPart = Mat::zeros(h / 2, w / 2, CV_8UC1);

	for (int y = 0; y < h / 2; y++) {
		for (int x = 0; x < w / 2; x++) {
			BoxPart.at<uchar>(y, x) = jahne.at<uchar>(y, x);
		}
	}
	for (int y = 0; y < h / 2; y++) {
		for (int x = w/2; x < w; x++) {
			MedianPart.at<uchar>(y, x-w/2) = jahne.at<uchar>(y, x);
		}
	}
	for (int y = h/2; y < h ; y++) {
		for (int x = w / 2; x < w; x++) {
			GaussPart.at<uchar>(y - h / 2, x - w / 2) = jahne.at<uchar>(y, x);
		}
	}
	GaussianBlur(GaussPart, GaussPart, cv::Size(kernelSize,kernelSize), sigma);
	medianBlur(MedianPart, MedianPart,kernelSize);
	boxFilter(BoxPart, BoxPart, -1, cv::Size(kernelSize, kernelSize));

	for (int y = 0; y < h / 2; y++) {
		for (int x = 0; x < w / 2; x++) {
			jahne.at<uchar>(y, x) = BoxPart.at<uchar>(y, x);
		}
	}
	for (int y = 0; y < h / 2; y++) {
		for (int x = w / 2; x < w; x++) {
			jahne.at<uchar>(y, x) = MedianPart.at<uchar>(y, x - w / 2);
		}
	}
	for (int y = h / 2; y < h; y++) {
		for (int x = w / 2; x < w; x++) {
		jahne.at<uchar>(y, x) = GaussPart.at<uchar>(y - h / 2, x - w / 2);
		}
	}


	imshow("FilteredWithCV", jahne);
	imwrite("FilterdWithCV.jpg", jahne);
	waitKey();
	destroyAllWindows();
}

Mat SobelX(Mat img) {
	Mat sobelx = Mat::zeros(3, 3, CV_8SC1);
	Mat target = Mat::zeros(img.size(), CV_32S);


	double wert;

	sobelx.at<schar>(0, 0) = -1;
	sobelx.at<schar>(1, 0) = -2;
	sobelx.at<schar>(2, 0) = -1;
	sobelx.at<schar>(0, 2) =  1;
	sobelx.at<schar>(1, 2) = +2;
	sobelx.at<schar>(2, 2) =  1;
	int sobelvalue;
	for (int y = 1; y < img.rows-1; y++) {
		int *targetptr = target.ptr<int>(y);
		for (int x = 1; x < img.cols-1; x++) {
			sobelvalue = 0;
			for (int dy = -1; dy <=1; dy++) {
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

Mat SobelY(Mat img) {
	Mat sobely = Mat::zeros(3, 3, CV_8SC1);
	Mat target = Mat::zeros(img.size(), CV_32S);
	
	

	double wert;

	sobely.at<schar>(0, 0) = -1;
	sobely.at<schar>(0, 1) = -2;
	sobely.at<schar>(0, 2) = -1;
	sobely.at<schar>(2, 0) = 1;
	sobely.at<schar>(2, 1) = +2;
	sobely.at<schar>(2, 2) = 1;
	int sobelvalue;
	for (int y = 1; y < img.rows-1; y++) {
		int *targetptr = target.ptr<int>(y);
		for (int x = 1; x < img.cols -1; x++) {
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

Mat gradMagnitude(Mat img, bool showMagnitude) {
	Mat sobelx = SobelX(img);
	Mat sobely = SobelY(img);
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

Mat gradDirection(Mat img) {

	Mat sobelx = SobelX(img);
	Mat sobely = SobelY(img);
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

void visualizeGradients(Mat img) {
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY,3);
	Mat magnitude = gradMagnitude(gray, false);
	Mat direction = gradDirection(gray);
	Mat fmagnitude = Mat::zeros(img.size(), CV_64FC1);
	int max = 0;
	int min = 1000;

	//magnitude auf [0,1] setzten
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			if (min > magnitude.at<int>(y, x))
				min = magnitude.at<int>(y, x);
			if (max < magnitude.at<int>(y, x))
				max = magnitude.at<int>(y, x);
		}
	}

	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			double wert = magnitude.at<int>(y, x);
			wert -= min;
			wert /= (max - min);
			fmagnitude.at<double>(y, x) = wert;
		}
	}
	//Einzeichnen der Gradienten
	for (int y = 0; y < img.rows; y += 5) {
		for (int x = 0; x < img.cols; x += 5) {
			if (abs(fmagnitude.at<double>(y, x)) > THETA) {
				double dir = direction.at<double>(y, x);
				double len = abs(fmagnitude.at<double>(y, x)) * LENGTH;
				int xstart = 0, xend = 0;
				int ystart = 0, yend = 0;
				int xpol, ypol;
				xpol = round(cos(dir) *len);
				ypol = round(sin(dir) * len);
				xend = x + xpol;
				yend = y + ypol;
				xstart = x - xpol;
				ystart = y - ypol;
				


			
			


				if (yend > img.rows || yend < 0 || ystart > img.rows || ystart < 0)
					continue;
				if (xend > img.cols || xend < 0 || xstart > img.cols || xstart < 0)
					continue;
				Point start(xstart, ystart);
				Point end(xend, yend);
				Point grad(x, y);
				cv::line(img, start, end, Scalar(0, 255, 0));
				circle(img, end, 2, Scalar(0, 255, 0));
				circle(img, grad, 1, Scalar(0, 0, 255));
			}
		}
	}
	imwrite("Gradients.jpg", img);
	imshow("Grads", img);
	waitKey();
	destroyAllWindows();



	
	

}

