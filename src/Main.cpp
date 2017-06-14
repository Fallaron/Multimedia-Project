#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>
#include "HOG\hog.h"

#define TEMPLATEWIDTH 64
#define TEMPLATEHEIGHT 128
#define ALPHA 5

using namespace std;
using namespace cv;

void slideOverImage(Mat img);
void getHOGFeatureArrayOnScaleAt(int x, int y, Mat img, int scale);

int main() {
	cout << "Hello World";
	getchar();
	Mat img = imread("len_full.jpg");
	slideOverImage(img);

}

//scale 0 = just img;
void getHOGFeatureArrayOnScaleAt(int x, int y, Mat img, int scale) {
	Mat cutout;
	double scalingfactor = pow(2, 1.0 / ALPHA);
	int templatew = TEMPLATEWIDTH;
	templatew*= pow(scalingfactor, scale);
	int templateh = TEMPLATEHEIGHT;
	templateh *= pow(scalingfactor, scale);
	img(Rect(x, y, templatew, templateh)).copyTo(cutout);
	vector<int> dims;
	double ***featArray = computeHoG(cutout, 8, dims);
	

}
void slideOverImage(Mat img) {
	int imgheight = img.size().height;
	int imgwidth = img.size().width;
	bool canGoDeeper = true;
	int stage = 0;
	int templateh=TEMPLATEHEIGHT;
	int templatew = TEMPLATEWIDTH;
	while (canGoDeeper) {
		//TODO think about y++
		for (int y = 0; y < imgheight-TEMPLATEHEIGHT; y++) {
			for (int x = 0; x < imgwidth-TEMPLATEWIDTH; x++) {
				//x,y for HOGfeature in Template
				getHOGFeatureArrayOnScaleAt(x, y, img, stage);

				
				

			}
		
			
		}
		//downsample
		imshow("Test", img);
		waitKey();
		int w = floor(img.cols / pow(2, 1.0 / ALPHA));
		int h = floor(img.rows / pow(2, 1.0 / ALPHA));
		//TODO Use Gaus for each octave
		resize(img, img,Size(w,h));
		stage++;

		if (img.cols < TEMPLATEWIDTH || img.rows < TEMPLATEHEIGHT)
			canGoDeeper = false;
		cout << "Width: " << img.size().width << " -- Height: " << img.size().height << endl;



	}



}