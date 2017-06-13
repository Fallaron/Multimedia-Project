#include <opencv2\highgui.hpp>
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <math.h>
#include <iostream>
using namespace cv;

#define FILE_NAME_LENGTH 70

Mat calcHistoGramm(Mat img, int bins, bool showAndDestroyWindows);
void contrastCorrection(Mat img);
void seperateChannels(Mat img);
void reverseChannels(Mat img);
void compute3DHisto1(Mat img, int n);
void highlightPixel(Mat img, Scalar color, int radius);
void selectSpaceinPic(Mat img);
void mouseCallBack(int event, int x, int y, int flags, void* userdata);
void lowerResolutionAndBits();




/*int main()
{
	Mat img = imread("underflow.jpg", 0);
	calcHistoGramm(img, 256, true);
	//contrastCorrection(img);
	//reverseChannels(img);
	//seperateChannels(img);
	//compute3DHisto1(img, 2);
	//Scalar color(255, 0, 0);
	//highlightPixel(img, color, 10);
	//selectSpaceinPic(img);
	//lowerResolutionAndBits();
}*/

//AUFGABE 1c
void contrastCorrection(Mat img) {
	float max = 0;
	float min = 255;
	double wert;
	for (int y = 0; y < img.rows; y++)
	{
		uchar* ptr = img.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			if (max < ptr[x])
				max = ptr[x];
			if (min > ptr[x])
				min = ptr[x];
		}
	}
	//Mapping	
	for (int y = 0; y < img.rows; y++)
	{
		uchar* ptr = img.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			wert = ptr[x] - min;
			wert /= (max - min);
			wert *= 255.0;
			ptr[x] = (int)wert;
		}
	}
	imshow("enhanced", img);
	imwrite("enhanced.jpg", img);
	Mat histo = calcHistoGramm(img, 256, false);
	imshow("Histogramm", histo);
	imwrite("histogramm_enhanced.jpg", histo);
	waitKey();
	destroyAllWindows();
}

//AUFGABE 1a
Mat calcHistoGramm(Mat img, int bins, bool showAndDestroyWindows) {
	int histo[256];
	int bin;
	int balkenBreite = 512 / bins;
	double balkenHoeheSkaling;
	int max = 0;
	int posX;

	if (img.rows == 0 || img.cols == 0) {
		return Mat();
	}
	Mat histoImg = Mat::zeros(512, 512, CV_8UC1);
	

	for (int i = 0; i < 256; i++)
	{
		histo[i] = 0;
	}

	for (int y = 0; y < img.rows; y++)
	{
		uchar* imagePtr = img.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			bin = imagePtr[x] / 256.0 *bins;
			histo[bin]++;
		}
	}

	for (int i = 0; i < 256; i++)
	{
		if (histo[i] > max)
			max = histo[i];
	}
	balkenHoeheSkaling = 512.0 / max;

	posX = 0;
	for (int i = 0; i < bins; i++) {
		int height = histo[i] * balkenHoeheSkaling;
		for (int y = 0; y < height; y++)
		{
			uchar* imagePtr = histoImg.ptr<uchar>(511 - y);
			for (int x = 0; x < balkenBreite; x++) {
				imagePtr[x + posX] = 255;
			}
		}
		posX += balkenBreite;
	}


	
	if (showAndDestroyWindows) {
		imshow("histo", histoImg);
		imshow("Bild", img);
		waitKey();
		imwrite("histogramm.jpg", histoImg);
		waitKey();
		destroyAllWindows();
	}
	return histoImg;
}

//AUFGABE 2c
void reverseChannels(Mat img) {
	if (img.rows == 0 || img.cols == 0)
		return;
	if (img.channels() != 3)
		return;

	Mat imgreverse = Mat::zeros(img.rows, img.cols, CV_8UC3);
	for (int y = 0; y < img.rows; y++)
	{
		uchar* ptr = img.ptr<uchar>(y);
		uchar* ptrreverse = imgreverse.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			ptrreverse[x * 3] = ptr[x * 3 + 2];
			ptrreverse[x * 3 + 1] = ptr[x * 3 + 1];
			ptrreverse[x * 3 + 2] = ptr[x * 3];
		}

	}
	imshow("Reversed", imgreverse);
	imwrite("Reversed.jpg", imgreverse);
	waitKey();
	destroyAllWindows();
}

//AUFGABE 2a
void seperateChannels(Mat img) {
	if (img.rows == 0 || img.cols == 0)
		return;

	if (img.channels() != 3)
		return;

	Mat imgred = Mat::zeros(img.rows, img.cols, CV_8UC3);
	Mat imgblue = Mat::zeros(img.rows, img.cols, CV_8UC3);
	Mat imggreen = Mat::zeros(img.rows, img.cols, CV_8UC3);
	for (int y = 0; y < img.rows; y++)
	{
		uchar* ptr = img.ptr<uchar>(y);
		uchar* ptrred = imgred.ptr<uchar>(y);
		uchar* ptrblue = imgblue.ptr<uchar>(y);
		uchar* ptrgreen = imggreen.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			ptrred[x * 3 + 2] = ptr[x * 3 + 2];
			ptrblue[x * 3] = ptr[x * 3];
			ptrgreen[x * 3 + 1] = ptr[x * 3 + 1];
		}

	}
	imshow("green", imggreen);
	imshow("blue", imgblue);
	imshow("red", imgred);
	imwrite("green.jpg", imggreen);
	imwrite("blue.jpg", imgblue);
	imwrite("red.jpg", imgred);
	waitKey();
	destroyAllWindows();

}

//AUFGABE 2b
void compute3DHisto1(Mat img, int n) {


	int pixel[3];
	int empt = 0;
	int full = 0;
	double ratio;
	int bins = pow(2, n);
	int*** threedhisto = (int***)calloc(bins, sizeof(int**));
	if (threedhisto == NULL) {
		//exit "freet" auch speicher
		std::exit(666);
	}
	for (int i = 0; i < bins; i++) {
		threedhisto[i] = (int **)calloc(bins, sizeof(int**));
		if (threedhisto[i] == NULL) {
			std::exit(666);
		}
	}
	for (int i = 0; i < bins; i++) {
		for (int j = 0; j < bins; j++) {
			threedhisto[i][j] = (int *)calloc(bins, sizeof(int*));
			if (threedhisto[i][j] == NULL) {
				std::exit(666);
			}
		}
	}



	if (img.rows == 0 || img.cols == 0)
		return;

	if (img.channels() != 3)
		return;

	for (int y = 0; y < img.rows; y++)
	{
		uchar* ptr = img.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			for (int i = 0; i < 3; i++)
			{
				pixel[i] = (ptr[x * 3 + i] / 256.0) * bins;
			}
			threedhisto[pixel[2]][pixel[1]][pixel[0]]++;
		}
	}
	//Ratio Berechnen
	for (int r = 0; r < bins; r++)
	{
		for (int g = 0; g < bins; g++)
		{
			for (int b = 0; b < bins; b++)
			{
				if (threedhisto[r][g][b] == 0)
					empt++;
			}
		}
	}
	ratio = (double)empt / pow(bins, 3);
	std::cout << "Ratio Empty: " << ratio;
	getchar();


}
//AUFGABE 2d
void highlightPixel(Mat img, Scalar color, int radius) {

	double R, G, B;
	double Hue;
	double maxH, maxC;
	double minH, minC;
	//Calc Hue
	R = color[2] / 255;
	G = color[1] / 255;
	B = color[0] / 255;
	maxC = std::max(R, G);
	maxC = std::max(maxC, B);
	minC = std::min(R, G);
	minC = std::min(minC, B);
	//Calc Hue
	if (R >= G && R >= B)
		Hue = (G - B) / (maxC - minC);
	if (G >= B && G >= R)
		Hue = 2.0 + (B - R) / (maxC - minC);
	if (B >= G && B >= R)
		Hue = 4.0 + (R - G) / (maxC - minC);
	Hue *= 30; //normal mal 60, aber cv nimmt lieber 0-180
	if (Hue < 0)
		Hue += 180;


	if (img.rows == 0 || img.cols == 0)
		return;

	if (img.channels() != 3)
		return;

	Mat imghsv = Mat::zeros(img.size(), CV_8UC3);
	cvtColor(img, imghsv, CV_BGR2HSV);
	maxH = std::min(Hue + radius, 179.0);
	minH = std::max(Hue - radius, 0.0);
	for (int y = 0; y < img.rows; y++)
	{
		uchar *ptr = img.ptr<uchar>(y);
		uchar *ptrhsv = imghsv.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			if (ptrhsv[x * 3] >= minH && ptrhsv[x * 3] <= maxH) {
				if (ptrhsv[x * 3 + 1] >= 50 && ptrhsv[x * 3 + 1] <= 255) {
					if (ptrhsv[x * 3 + 2] >= 50 && ptrhsv[x * 3 + 2] <= 255) {
						continue;
					}
				}
			}
			for (int c = 0; c < 3; c++) {
				ptr[x * 3 + c] = 0.2126*ptr[x * 3 + 2] + 0.7152*ptr[x * 3 + 1] + 0.0722*ptr[x * 3];
			}
		}


	}
	imshow("Help", img);
	waitKey();
	imwrite("Highlighed.jpg", img);
	destroyAllWindows();

}

//AUFGABE 3
void selectSpaceinPic(Mat img) {
	if (img.empty())
		return;

	namedWindow("Picture");
	namedWindow("Red");
	namedWindow("Green");
	namedWindow("Blue");
	
	setMouseCallback("Picture", mouseCallBack, &img);
	Mat copy = Mat::zeros(img.size(), CV_8UC3);
	img.copyTo(copy);
	imshow("Picture", img);
	waitKey();
	destroyAllWindows();
	


}
//AUFGABE 3(mouse events)
void mouseCallBack(int event, int x, int y, int flags, void* userdata) {
	Mat *img = static_cast<Mat*>(userdata);
	Mat copy = Mat::zeros((*img).size(), CV_8UC3);
	static int xDown;
	static int xUp;
	static int yDown;
	static int yUp;
	(*img).copyTo(copy);
	if (event == EVENT_LBUTTONDOWN) {
		xDown = x;
		yDown = y;
	}
	if (event == EVENT_LBUTTONUP) {
		xUp = x ;
		yUp = y;
		uchar* ptr;

		if (y < 0 || x < 0 || y > img->rows || x > img->cols)
			return;
		
		//swap if wrong way round
		if (xUp < xDown) {
			int temp = xDown;
			xDown = xUp;
			xUp = temp;
		}
		if (yUp < yDown) {
			int temp = yDown;
			yDown = yUp;
			yUp = temp;
		}
		std::cout << "Pos: (" << xDown << "," << yDown << ") to ("
			<< xUp << "," << yUp << ")." << std::endl;
		//Draw each Line
		ptr = copy.ptr<uchar>(yDown);
		for (int x = xDown; x <= xUp; x += 2) {
			for (int c = 0; c < 3; c++) {
				ptr[x * 3 + c] = 255;
			}
		}

		ptr = copy.ptr<uchar>(yUp);
		for (int x = xDown; x <= xUp; x += 2) {
			for (int c = 0; c < 3; c++) {
				ptr[x * 3 + c] = 255;
			}
		}

		for (int y = yDown; y <= yUp; y += 2) {
			ptr = copy.ptr<uchar>(y);
			for (int c = 0; c < 3; c++) {
				ptr[xDown * 3 + c] = 255;
			}
		}

		for (int y = yDown; y <= yUp; y += 2) {
			ptr = copy.ptr<uchar>(y);
			for (int c = 0; c < 3; c++) {
				ptr[xUp * 3 + c] = 255;
			}
		}

		Mat red = Mat::zeros(yUp - yDown, xUp - xDown, CV_8UC1);
		Mat green = Mat::zeros(yUp - yDown, xUp - xDown, CV_8UC1);
		Mat blue = Mat::zeros(yUp - yDown, xUp - xDown, CV_8UC1);

		for (int y = yDown; y < yUp; y++) {
			ptr = (*img).ptr<uchar>(y);
			uchar* ptrRed = red.ptr<uchar>(y-yDown);
			uchar* ptrBlue = blue.ptr<uchar>(y-yDown);
			uchar* ptrGreen = green.ptr<uchar>(y-yDown);
			for (int x = xDown; x <= xUp; x++ ) {
				ptrRed[x - xDown] = ptr[x * 3 + 2];
				ptrBlue[x - xDown] = ptr[x * 3];
				ptrGreen[x - xDown] = ptr[x * 3 + 1];
			}
		}
		
		Mat histoRed = calcHistoGramm(red, 255, false);
		Mat histoGreen = calcHistoGramm(green, 255, false);
		Mat histoblue = calcHistoGramm(blue, 255, false);

		if (!histoRed.empty() && !histoblue.empty() && !histoGreen.empty()) {
			imshow("Red", histoRed);
			imshow("Blue", histoblue);
			imshow("Green", histoGreen);
		}

		

		imshow("Picture", copy);
	}
}

void lowerResolutionAndBits() {
	Mat img;
	String name;
	int lower;
	int bit;
	int count;
	int avg;
	

	std::cout << "Please enter Imagename, and two integers: " <<std::endl;
	std::cin >> name >> lower >> bit;
	img = imread(name, 0);
	Mat imgbits;
	img.copyTo(imgbits);
	if (img.empty())
		return;
	if (lower < 1 || lower > std::min(img.size().height, img.size().width))
		return;
	if (bit < 1 || bit > 8)
		return;

	for (int y = 0; y < img.rows; y += lower) {
		for (int x = 0; x < img.cols; x += lower) {
			avg = 0;
			count = 0;
			//Durchschnitt berechnen
			for (int dy = 0; dy < lower; dy++) {
				if (y + dy >= img.rows)
					break;
				uchar* ptr = img.ptr<uchar>(y + dy);
				for (int dx = 0; dx < lower; dx++) {
					if (dx + x >= img.cols)
						break;
					count++;
					avg += ptr[x + dx];
				}
			}

			avg /= count;
			//Werte setzen
			for (int dy = 0; dy < lower; dy++) {
				if (y + dy >= img.rows)
					break;
				uchar* ptr = img.ptr<uchar>(y + dy);
				for (int dx = 0; dx < lower; dx++) {
					if (dx + x >= img.cols)
						break;
					ptr[x + dx] = avg;
				}
			}
		}
	}


	//Reduce bits and AVG;
	for (int y = 0; y < img.rows; y++) {
		uchar* ptr = imgbits.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++) {
			uchar value = ptr[x];
			//Hintersten 8-bit auf 0 setzen
			value = value >> 8-bit;
			value = value << 8-bit;
			value += (256 / (pow(2, bit+1)));
			ptr[x] = value;
 		}
	}

	char datname[FILE_NAME_LENGTH];
	sprintf(datname, "Lowered_by_%i.jpg", lower);
	imwrite(datname, img);
	sprintf(datname, "Quantized_by_%i.jpg", bit);
	imwrite(datname, imgbits);
	imshow("LowerdBits", imgbits);
	imshow("LoweredRes",img);
	waitKey();
	destroyAllWindows();


}