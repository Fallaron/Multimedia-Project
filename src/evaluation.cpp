#include "evaluation.h"
#include "features.h"
#include <string>
using namespace cv;
using namespace std;
#define OVERLAP_THRES 0.2
#define CONTAINED_THRES 0.5
#define MAXBOX_COUNT 5

void non_Max_Suppression(std::vector<std::vector<float>>& final_BBox, std::vector<std::vector<float>> detWinFeat, int temp_Width, int temp_Height) {
	final_BBox = std::vector<std::vector<float>>(0);
	int boxes = detWinFeat.size();
	float overlap = 0.0;
	bool addNewBox = true, del = false, jumpBack = false;
	if (!detWinFeat.empty()) {
		//************* initial bBox values ********
		int count = 1;
		double scale = detWinFeat[0][3];
		float x1 = detWinFeat[0][1] * scale;
		float y1 = detWinFeat[0][2] * scale;

		float width = temp_Width * scale;
		float height = temp_Height * scale;

		vector<float> temp;
		temp.push_back(x1);
		temp.push_back(y1);
		temp.push_back(x1 + width);
		temp.push_back(y1 + height);
		temp.push_back(detWinFeat[0][0]);
		final_BBox.push_back(temp);
		temp.clear();
		for (int i = 1; i < boxes; i++) {
			addNewBox = true;
			del = false;
			const double scale = detWinFeat[i][3];
			const int pos_x = detWinFeat[i][1] * scale;
			const int pos_y = detWinFeat[i][2] * scale;

			const int width_c = temp_Width * scale;
			const int height_c = temp_Height * scale;

			cv::Rect current_bBox(pos_x, pos_y, width_c, height_c);
			float current_score = detWinFeat[i][0];

			// if current box has to replace more than one boxes delete 2nd, 3rd etc reducing the final_box vector size and avoiding duplicates	
			for (int n = 0; n < count; n++) {
				if (jumpBack) {
					del = false;
					jumpBack = false;
				}
				const int b_x1 = final_BBox[n][0];
				const int b_y1 = final_BBox[n][1];
				const int b_x2 = final_BBox[n][2];
				const int b_y2 = final_BBox[n][3];
				const int stored_score = final_BBox[n][4];

				cv::Rect stored_bBox(b_x1, b_y1, b_x2 - b_x1, b_y2 - b_y1);
				cv::Rect intersect_rect = stored_bBox & current_bBox;
				cv::Rect union_rect = stored_bBox | current_bBox;

				float inter_Area = intersect_rect.area();
				float uni = union_rect.area();
				overlap = inter_Area / uni;
				float c_Area = current_bBox.area();
				float s_Area = stored_bBox.area();
				// go through boxes and consider if overlap
				if (overlap > OVERLAP_THRES) {
					addNewBox = false;
					if (del && final_BBox.size() != 0) {
						final_BBox.erase(final_BBox.begin() + n);
						count--;
						jumpBack = true;
						n = 0; //reset loop, box positions have changed in the final vector
					}
					else {
						if (current_score < stored_score ) {
							// replace up to now best box
							final_BBox[n][0] = pos_x;
							final_BBox[n][1] = pos_y;
							final_BBox[n][2] = width_c + pos_x;
							final_BBox[n][3] = height_c + pos_y;
							final_BBox[n][4] = current_score;
						}
						del = true; // else suprress this current bbox
					}
				}// smaller rect within a bigger rect => overlap is very small.. solution.. delete contained rectangle or replace it with bigger one.. or suppress smaller in coming
				else {
					if (inter_Area == s_Area) { //supress contained boxes
						addNewBox = false;
						if (del && final_BBox.size() != 0) {
							final_BBox.erase(final_BBox.begin() + n);
							count--;
							jumpBack = true;
							n = 0;
						}
						else {
							//repalace it
							final_BBox[n][0] = pos_x;
							final_BBox[n][1] = pos_y;
							final_BBox[n][2] = width_c + pos_x;
							final_BBox[n][3] = height_c + pos_y;
							final_BBox[n][4] = current_score;
							del = true;
						}
					}
					else if ((inter_Area / s_Area) > CONTAINED_THRES) { //supress almost contained boxes
						addNewBox = false;
						if (del && final_BBox.size() != 0) {
							final_BBox.erase(final_BBox.begin() + n);
							count--;
							jumpBack = true;
							n = 0; // reset loop, box positions have changed in the final vector
						}
						else {
							//repalace it
							final_BBox[n][0] = pos_x;
							final_BBox[n][1] = pos_y;
							final_BBox[n][2] = width_c + pos_x;
							final_BBox[n][3] = height_c + pos_y;
							final_BBox[n][4] = current_score;
							del = true;
						}
					}
				}
			}
			// add new bBox.. cud be new pedstrian thus increase the size of final bBoxes if count less than max box count
			if (count <= MAXBOX_COUNT && addNewBox) {
				count++;
				vector<float> temp;
				temp.push_back(pos_x);
				temp.push_back(pos_y);
				temp.push_back(width_c + pos_x);
				temp.push_back(height_c + pos_y);
				temp.push_back(current_score);
				final_BBox.push_back(temp);
				temp.clear();
			}
		}
	}
	//fixMaxSupressions(final_BBox);
}

void showMaximabBoxes(std::vector<std::vector<float>>& final_BBox, Mat img, std::vector<int> & bBoxesOrig, string windowName) {
	
	float overlap = 0;
	if (img.empty())
		return;
	for (auto &b : final_BBox) {
		int xtl = b[0];
		int ytl = b[1];
		int xbr = b[2];
		int ybr = b[3];
		float score = b[4];
		Point tl(xtl, ytl);
		Point br(xbr, ybr);
		rectangle(img, tl, br, Scalar(0, 255, 0)); for (int i = 0; i < bBoxesOrig.size(); i += 4) {
			int xtl = bBoxesOrig[i];
			int ytl = bBoxesOrig[i + 1];
			int xbr = bBoxesOrig[i + 2];
			int ybr = bBoxesOrig[i + 3];
			Point tl2(xtl, ytl);
			Point br2(xbr, ybr);
			Rect box(tl, br);
			Rect trueBox(tl2, br2);
			Rect uni = box | trueBox;
			Rect inter = box & trueBox;
			float overlapTemp = (float)inter.area() / (float) uni.area();
			overlap = (overlap < overlapTemp ? overlapTemp : overlap);


		}
		
		putText(img, to_string(score) + "  O: " +to_string(overlap*100) + "%" , tl, FONT_HERSHEY_PLAIN, 0.85, Scalar(0, 255, 0), 1.5);
		overlap = 0;
	}

	for (int i = 0; i < bBoxesOrig.size(); i += 4) {
		int xtl = bBoxesOrig[i];
		int ytl = bBoxesOrig[i+1];
		int xbr = bBoxesOrig[i+2];
		int ybr = bBoxesOrig[i+3];
		Point tl(xtl, ytl);
		Point br(xbr, ybr);
		rectangle(img, tl, br, Scalar(0, 0, 255));
	}


	imshow(windowName, img);
	waitKey();

}

void detectionWindow_features(std::vector<std::vector<float>>& detWinFeat, int x, int y, float scale, float score) {
	std::vector<float> temp;
	temp.push_back(score);
	temp.push_back(x);
	temp.push_back(y);
	temp.push_back(scale);
	//temp.push_back(img_id);
	detWinFeat.push_back(temp);
	temp.clear();
}

