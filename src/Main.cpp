#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>

#define TEMPLATEWIDTH 64
#define TEMPLATEHEIGHT 128
#define ALPHA 5

using namespace std;
using namespace cv;

void slideOverImage(Mat img);


int main() {
	cout << "Hello World";
	getchar();
	Mat img = imread("len_full.jpg");
	slideOverImage(img);

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
		for (int y = 0; y < imgheight; y++) {
			for (int x = 0; x < imgwidth; x++) {
				if (x + TEMPLATEWIDTH >= imgwidth)
					break;
				//x,y for HOGfeature in Template
				//if found calc template height and with in original picute
				/*for (int i = 0; i < stage; i++) {
					templatew *= pow(2, 1.0 / ALPHA);
					templateh *= pow(2, 1.0 / ALPHA);
				}*/

				
				templatew *= pow(pow(2, 1.0 / ALPHA), stage);
				templateh *= pow(pow(2, 1.0 / ALPHA), stage);

			}
			if (y + TEMPLATEHEIGHT >= imgheight)
				break;
			
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