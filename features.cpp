#include "features.h"
#include "src\HOG\hog.h"

#define BLOCK_SIZE 2

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

double ** get_HOG_feat_trainSet(cv::Mat img, const int cell_size, std::vector<int>& dims) {
	double ***featArray_HoG;
	std::vector<int> dims_HoG = vector<int>(3);

	if (!img.empty())
		featArray_HoG = computeHoG(img, cell_size, dims_HoG);
	else
		cout << "No image found!" << endl;

	//Memory for features vector
	int dim_y = dims_HoG[0] * dims_HoG[1] - 2; // for a 64 x 128 patch
	int dim_x = dims_HoG[2] * BLOCK_SIZE * 2;

	double** featArray = (double**)malloc(dim_y * sizeof(double*));
	for (int i = 0; i < dim_y; ++i) {
		featArray[i] = (double*)malloc(dim_x * sizeof(double));
	}
	dims = vector<int>(2);
	dims[0] = dim_y;
	dims[1] = dim_x;

	// blocks
	int num_Vblocks = dims_HoG[0] - 1;
	int num_Hblocks = dims_HoG[1] - 1;
	int Vblocks = 0;
	int Hblocks;
	int vec = 0;

	for (int i = 0; i < dims_HoG[0] - 1; i++) {
		Hblocks = 0;
		for (int j = 0; j < dims_HoG[1] - 1; j++) {
			//traverse 2 cells right and then 2 down-- forming a single block			
			double block_norm_factor = 1;
			for (int n = i; n < BLOCK_SIZE + i; n++) {
				for (int m = j; m < BLOCK_SIZE + j; m++) {
					for (int k = 0; k < dims_HoG[2]; k++) {
						double val = featArray_HoG[n][m][k];
						block_norm_factor += val * val;
					}
				}
			}
			// normalize block and save the feature
			vector<double> feature;
			block_norm_factor = pow(block_norm_factor, 0.5);
			for (int n = i; n < BLOCK_SIZE + i; n++) {
				for (int m = j; m < BLOCK_SIZE + j; m++) {
					for (int k = 0; k < dims_HoG[2]; k++) {
						 feature.push_back(featArray_HoG[n][m][k] / block_norm_factor);
					}
				}
			} 
			//concatnate vectors
			for (int j = 0; j < dims[1]; j++) {
				featArray[vec][j] = feature[j];
			}
			vec++;
			Hblocks++;
		}
		Vblocks++;
	}
	cout << "Blocks V - H " << Vblocks << "  - " << Hblocks << " , ";
	return featArray;
}

