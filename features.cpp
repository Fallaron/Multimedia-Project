#include "features.h"
#include "src\HOG\hog.h"
using namespace cv;
using namespace std;

void  getBoundingBox(std::string annotationList, std::vector<std::vector<int>>& boundingBoxes) {

	string annotationPath, line, value;
	ifstream annotationlst;
	ifstream annotationTxtFile;
	int boundingBoxLineNum = 17;
	char c;
	bool store = false;
	std::vector<int> boundingBoxValues;
	boundingBoxes = std::vector<std::vector<int>>();

	if (!annotationList.empty()) {
		// read annnotation list file from path
		annotationlst.open(annotationList);
		while (!annotationlst.eof()) {
			getline(annotationlst, annotationPath);
			// open annnotation text file from path
			annotationTxtFile.open(annotationPath);

			while (!annotationTxtFile.eof())
			{
				// get to a specific line with boundingBox info from annotation text file
				int count = 0;
				while (getline(annotationTxtFile, line)) {
					if (count == boundingBoxLineNum) {
						// look for bounding box data from this specific line

						for (int j = 69; j < line.size(); ++j) {
							c = line[j];
							while (j < line.size() && isdigit(c)) {
								value.push_back(c);
								c = line[++j];
								store = true;
							}
							if (store == true) {
								boundingBoxValues.push_back(stoi(value));
								store = false;
								value.clear();
							}
						}
						// store bounding box values for a specific row or image
						boundingBoxes.push_back(boundingBoxValues);
						boundingBoxValues.clear();
					}
					count++;
				}
			}
			annotationTxtFile.close();
		}
		annotationlst.close();
	}
}

bool is_detection_Ok(int det_detail[], int annot_index, const std::vector<std::vector<int>> boundingBoxes) {
	//det_detail[] entail detector width and height and int x, int y for locality..
	/***** still thinking *****/
	
	bool status = false;
	int x1 = boundingBoxes[annot_index][0];
	int y1 = boundingBoxes[annot_index][1];
	int x2 = boundingBoxes[annot_index][2];
	int y2 = boundingBoxes[annot_index][3];

	cv::Rect detect_rect(det_detail[0], det_detail[1], det_detail[2], det_detail[3]);
	cv::Rect annot_rect(x1, y1, x2 - x1, y2 - y1);
	cv::Rect intersect_rect = detect_rect & annot_rect;
	cv::Rect union_rect = detect_rect | annot_rect;

	double overlap = intersect_rect.area() / union_rect.area();
	if (overlap > 0.5)
		status = true;
	
	return status;
}

void get_HOG_feat_train(std::string img_list_file_path, double ***&featArray, const int cell_size, std::vector<int>& dims) {
	ifstream img_lst;
	string img_path;
	img_lst.open(img_list_file_path);
	// for each image compute the HoG and write back the values by reference repeat till list is done.
	while (!img_lst.eof()) {
		getline(img_lst, img_path);
		cv::Mat img = imread(img_path);
		if (!img.empty())
			featArray = computeHoG(img, cell_size, dims);
	}
}
