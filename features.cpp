#include "features.h"
#include "src\HOG\hog.h"
#include <ctime>

#define DETECTORWIDTH 64
#define DETECTORHEIGHT 128
#define CUTOFF 1

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

bool is_detection_Ok(int detector_pos[], int annot_index, const std::vector<std::vector<int>> boundingBoxes) {
	// functions determines the true positive, false negative detections.. evaluation tool
	//detector_pos[] entails int x, int y for locality..

	bool status = false;
	int x1 = boundingBoxes[annot_index][0];
	int y1 = boundingBoxes[annot_index][1];
	int x2 = boundingBoxes[annot_index][2];
	int y2 = boundingBoxes[annot_index][3];

	cv::Rect detect_rect(detector_pos[0], detector_pos[1], DETECTORWIDTH, DETECTORHEIGHT);
	cv::Rect annot_rect(x1, y1, x2 - x1, y2 - y1);
	cv::Rect intersect_rect = detect_rect & annot_rect;
	cv::Rect union_rect = detect_rect | annot_rect;

	double overlap = intersect_rect.area() / union_rect.area();
	if (overlap > 0.5)
		status = true;
	return status;
}

// set boolean to false and path to neg samples to extract neg sample features else extract pos features..
// dataSet_featArray is used to trains the classifier
void get_HoG_feat_trainSets(double **&dataSet_featArray, std::string dataSet_path, const int cell_size, std::vector<int>& feat_dims, bool pos_Set) {
	feat_dims = vector<int>(2);
	vector<string> dataSet_img_paths;
	get_dataSet(dataSet_path, dataSet_img_paths);
	int num_img = dataSet_img_paths.size();

	if (pos_Set == false) {
		int patchWidth = 80;
		int patchHeight = 144;
		srand(time(NULL));
		cv::Mat img_patch;
		int num_img_patches;
		int num_img = dataSet_img_paths.size();
		int p = 0;

		vector<int> dims;
		int y = (DETECTORWIDTH / cell_size);
		int x = (DETECTORHEIGHT/ cell_size);
		int num_dims = 32;
		int total_num_imgs = num_img * 10;
		int features = y * x * num_dims;

		// memory for negative training set HoG feature flattened
		dataSet_featArray = (double**)malloc(total_num_imgs * sizeof(double**));
		for (int i = 0; i < total_num_imgs; ++i) {
			dataSet_featArray[i] = (double*)malloc(features * sizeof(double));
		}

		feat_dims[0] = total_num_imgs;
		feat_dims[1] = features;

		//randomly select 10 patches from the neg sample from each image
		for (int i = 0; i < num_img; i++) {
			num_img_patches = 0;
			cout << i << ",";
			cv::Mat src = imread("INRIAPerson/" + dataSet_img_paths[i]);
			if (!src.empty()) {
				while (num_img_patches < 10) {
					int y_pos = rand() % (src.rows - patchHeight);
					int x_pos = rand() % (src.cols - patchWidth);
					img_patch = src(Rect(x_pos, y_pos, patchWidth, patchHeight));
					double *** feat = computeHoG(img_patch, cell_size, dims);

					//save to the dataset array bag
					int y_cells = dims[0];
					int x_cells = dims[1];
					int z = dims[2];
					int h = 0;
					for (int i = 0; i < y_cells; i++) {
						for (int j = 0; j < x_cells; j++) {
							for (int n = 0; n < z; n++) {
								dataSet_featArray[p][h++] = (feat[i][j][n]);
							}
						}
					}
					num_img_patches++;
					p++;
				}
			}
		}
	}
	// Extract features of the pos sample dataset
	else {
		vector<int> FHoG_dims;
		double ***FHoG;
		//memory for dataset featureArray bag
		int features = 16 * 8 * 32;
		dataSet_featArray = (double**)malloc(num_img * sizeof(double**));
		for (int i = 0; i < num_img; ++i) {
			dataSet_featArray[i] = (double*)malloc(features * sizeof(double));
		}
		feat_dims[0] = num_img;
		feat_dims[1] = features;

		for (int k = 0; k < num_img; k++) {
			cv::Mat img = imread("INRIAPerson/96X160H96/" + dataSet_img_paths[k]);
			int h = 0;
			if (!img.empty()) {
				FHoG = computeHoG(img, cell_size, FHoG_dims);
				int y_cells = FHoG_dims[0];
				int x_cells = FHoG_dims[1];
				int z = FHoG_dims[2];

				//chop off the boundary features from the now 80 X 144 image to retain a 64 X 128 image patch
				for (int i = CUTOFF; i < y_cells - CUTOFF; i++) {
					for (int j = CUTOFF; j < x_cells - CUTOFF; j++) {
						for (int n = 0; n < z; n++) {
							dataSet_featArray[k][h++] = FHoG[i][j][n];
						}
					}
				}

			}//end if
		}// end for
		 // deallocate memory for FHOG
		for (int i = 0; i < FHoG_dims[0]; i++) {
			for (int j = 0; j < FHoG_dims[1]; j++) {
				delete[] FHoG[i][j];
			}
			delete[] FHoG[i];
		}
		delete FHoG;
	}// end else
}

// reads the dataset *.*ls files and returns paths to images as strings in an Array
void get_dataSet(std::string dataSet_listFile_path, vector<string>& img_paths) {
	img_paths = vector<string>();
	ifstream img_lst;
	string img_path;
	img_lst.open(dataSet_listFile_path);
	while (!img_lst.eof()) {
		getline(img_lst, img_path);
		img_paths.push_back(img_path);
	}
}
