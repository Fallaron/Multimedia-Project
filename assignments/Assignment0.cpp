
#include <iostream>
#include <fstream>
#include <time.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void aufgabe03(int n);
void aufgabe04();

using namespace std;
using namespace cv;

/*int main() {

	srand(time(NULL));

	aufgabe03(8);
	return 0;
}*/

void aufgabe03(int n) {
	Mat img = imread("wiese.jpg");
	String text = "Hello world!";
	int font = FONT_HERSHEY_COMPLEX;
	int thick = 2;



	for (int i = 0; i < n; i++)
	{
		double fontscale = rand() % 4000 / 1000.0;
		Point textOrg(rand() % img.cols, rand() % img.rows);
		Scalar color(rand() % 256, rand() % 256, rand() % 256);
		putText(img, text, textOrg, font, fontscale, color, thick);

	}
	imshow("text", img);
	waitKey();
	destroyAllWindows();

}

void aufgabe04() {
	int hights[20];
	int widths[20];
	int c = 0;
	ofstream textfileout;
	ifstream textfile;
	textfile.open("Pictures.txt");
	string text;
	while (getline(textfile, text)) {
		Mat img = imread(text);
		if (img.rows > 0) {
			hights[c] = img.rows;
			widths[c] = img.cols;
			imshow("Bild", img);
			waitKey();
			destroyAllWindows();
			c++;
		}
	}
	//Mean:
	double meanW = 0;
	double meanH = 0;
	for (int i = 0; i < c; i++)
	{
		meanH += hights[i];
		meanW += widths[i];
	}
	meanH /= c;
	meanW /= c;

	//Median
	sort(hights, hights + c);
	sort(widths, widths + c);

	int medianW = widths[c / 2];
	int medianH = hights[c / 2];

	// Variance
	double varW = 0;
	double varH = 0;
	for (int i = 0; i < c; i++)
	{
		varW += (widths[i] - meanW) * (widths[i] - meanW);
		varH += (hights[i] - meanH) * (hights[i] - meanH);
	}
	varH /= c;
	varW /= c;

	textfileout.open("Outputfile.txt");
	textfileout << "Mean Width: " << meanW << endl;
	textfileout << "Mean Height: " << meanH << endl;
	textfileout << "Median Width: " << medianW << endl;
	textfileout << "Median Height: " << medianH << endl;
	textfileout << "Varianz Width: " << varW << endl;
	textfileout << "Varianz Height: " << varH << endl;

	textfileout.close();



}