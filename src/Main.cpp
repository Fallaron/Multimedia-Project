#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>

#define TEMPLATEWIDTH 64
#define TEMPLATEHEIGHT 128

using namespace std;
using namespace cv;


int main() {
	cout << "Hello World";
	getchar();
}

void slideOverImage(Mat img) {
	int imgheight = img.size().height;
	int imgwidth = img.size().width;
	bool canGoDeeper = true;
	while (canGoDeeper) {
		//TODO think about y++
		for (int y = 0; y < imgheight; y++) {
			for (int x = 0; x < imgwidth; x++) {
				if (x + TEMPLATEWIDTH >= imgwidth)
					break;
				//x,y for HOGfeature in Template
			}
			if (y + TEMPLATEHEIGHT >= imgheight)
				break;
		}



	}



}