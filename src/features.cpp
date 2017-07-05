#include "features.h"
#include "HOG\hog.h"
#include "trainer.h"
#include <ctime>
#include <iostream>
#include <regex>

#define CUTOFF 1
#define IMG_PATCH_NEG 10


using namespace cv;
using namespace std;

void  getBoundingBox(std::string annotationList, std::vector<std::vector<int>>& boundingBoxes) {

	string annotationPath, line, value;
	ifstream annotationlst;
	ifstream annotationTxtFile;
	char c;
	bool store = false;
	string sub = "Bounding box.*";
	std::regex rx(sub);
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
				while (getline(annotationTxtFile, line)) {
					if (std::regex_match(line, rx)) {
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
					}
				}
				// store bounding box values for a specific image
				if (!boundingBoxValues.empty()) {
					boundingBoxes.push_back(boundingBoxValues);
					boundingBoxValues.clear();
				}
			}
			annotationTxtFile.close();
		}
		annotationlst.close();
	}
}

int detection_true_count(const std::vector<std::vector<float>> prediction_bBox, const std::vector<int> boundingBoxes) {
	int gBox_size = boundingBoxes.size();
	int overlap_Count = 0;
	double overlap = 0.0;

	for (int k = 0; k < gBox_size; k += 4) {
		// direct comparison => info is stored contagiously for each image in folder
		int x1 = boundingBoxes[k];
		int y1 = boundingBoxes[k + 1];
		int x2 = boundingBoxes[k + 2];
		int y2 = boundingBoxes[k + 3];
		cv::Rect groundtruth_bBox(x1, y1, x2 - x1, y2 - y1);
		// compare a single to all boxes stored in pred_box for this particular Image
		// go through all predicted bounding boxes until a matching ground truth is found if any.. 
		for (auto &box : prediction_bBox) {
			int pos_x = box[0];
			int pos_y = box[1];
			int pos_x2 = box[2];
			int pos_y2 = box[3];

			cv::Rect Predicted_bBox(pos_x, pos_y, pos_x2 - pos_x, pos_y2 - pos_y);
			cv::Rect intersect_rect = Predicted_bBox & groundtruth_bBox;
			cv::Rect union_rect = Predicted_bBox | groundtruth_bBox;

			overlap = (float)intersect_rect.area() / (float)union_rect.area();
			// for all bounding boxes in prediction boxes vector do this evaluation..
			if (overlap > 0.5) {
				overlap_Count++;
			}
		}
	}
	return overlap_Count;
}

// set boolean to false and path to neg samples to extract neg sample features else extract pos features..
// dataSet_featArray is used to trains the classifier
void get_HoG_feat_trainSets(double **&dataSet_featArray, std::string dataSet_path, const int cell_size, int temp_Width, int temp_Height, std::vector<int>& feat_dims, bool pos_Set) {

	feat_dims = std::vector<int>(2);
	std::vector<std::string> dataSet_img_paths;
	get_dataSet(dataSet_path, dataSet_img_paths);
	int num_img = dataSet_img_paths.size();

	if (pos_Set == false) {
		cout << "Generating negative training data... ";
		int patchWidth = 80;
		int patchHeight = 144;
		srand(time(NULL));
		cv::Mat img_patch;
		int num_img_patches;
		//int num_img = dataSet_img_paths.size();
		int p = 0;

		int x = (temp_Width / cell_size);
		int y = (temp_Height / cell_size);
		int num_dims = 32;
		int total_num_imgs = num_img * IMG_PATCH_NEG;
		int features = y * x * num_dims;

		// memory for negative training set HoG feature flattened
		dataSet_featArray = (double**)malloc(total_num_imgs * sizeof(double**));
		for (int i = 0; i < total_num_imgs; ++i) {
			dataSet_featArray[i] = (double*)malloc(features * sizeof(double));
		}

		feat_dims[0] = total_num_imgs;
		feat_dims[1] = features;

		//randomly select 10 patches from the neg sample from each image
		double *** FHoG;
		vector<int> FHoG_dims;
		for (int i = 0; i < num_img; i++) {
			num_img_patches = 0;
			cv::Mat src = imread(dataSet_img_paths[i]);
			if (!src.empty()) {
				while (num_img_patches < IMG_PATCH_NEG) {
					int y_pos = rand() % (src.rows - patchHeight);
					int x_pos = rand() % (src.cols - patchWidth);
					img_patch = src(Rect(x_pos, y_pos, patchWidth, patchHeight));
					FHoG = computeHoG(img_patch, cell_size, FHoG_dims);

					//save to the dataset array bag
					int y_cells = FHoG_dims[0];
					int x_cells = FHoG_dims[1];
					int z = FHoG_dims[2];
					int h = 0;
					for (int i = 0; i < y_cells; i++) {
						for (int j = 0; j < x_cells; j++) {
							for (int n = 0; n < z; n++) {
								dataSet_featArray[p][h++] = (FHoG[i][j][n]);
							}
						}
					}
					num_img_patches++;
					p++;

					// deallocate memory for FHOG
					for (int i = 0; i < FHoG_dims[0]; i++) {
						for (int j = 0; j < FHoG_dims[1]; j++) {
							free(FHoG[i][j]);
						}
						free(FHoG[i]);
					}
					free(FHoG);
				}
			}
		}
		
		cout << "done!" << endl;
	}
	// Extract features of the pos sample dataset
	else {
		cout << "Generating positive training data... ";
		vector<int> FHoG_dims;
		double ***FHoG;
		//memory for dataset featureArray bag
		int x = (temp_Width / cell_size);
		int y = (temp_Height / cell_size);
		int features = y * x * 32;
		dataSet_featArray = (double**)malloc(num_img * sizeof(double**));
		for (int i = 0; i < num_img; ++i) {
			dataSet_featArray[i] = (double*)malloc(features * sizeof(double));
		}
		feat_dims[0] = num_img;
		feat_dims[1] = features;

		for (int k = 0; k < num_img; k++) {
			cv::Mat img = imread(dataSet_img_paths[k]);
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
				for (int i = 0; i < FHoG_dims[0]; i++) {
					for (int j = 0; j < FHoG_dims[1]; j++) {
						free(FHoG[i][j]);
					}
					free(FHoG[i]);
				}
				free(FHoG);
			}//end if
		}// end for
		 // deallocate memory for FHOG
		
		cout << "done!" << endl;
	}// end else


}

// reads the dataset *.*ls files and returns paths to images as strings in an Array
void get_dataSet(std::string dataSet_listFile_path, vector<std::string>& img_paths) {
	img_paths = vector<string>();
	ifstream img_lst;
	string img_path;
	img_lst.open(dataSet_listFile_path);
	while (!img_lst.eof()) {
		getline(img_lst, img_path);
		if (img_path.empty())
			continue;
		img_paths.push_back(img_path);
	}
}


