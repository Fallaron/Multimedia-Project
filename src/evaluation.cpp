#include "evaluation.h"

using namespace cv;
using namespace std;

void non_Max_Suppression(std::vector<std::vector<float>>& final_BBox, std::vector<std::vector<float>> detWinFeat, int temp_Width, int temp_Height) {
	final_BBox = std::vector<std::vector<float>>(0);
	int boxes = detWinFeat.size();
	float overlap = 0.0;
	bool addNewBox = true, del = false;
	if (!detWinFeat.empty()) {
		//************* initial bBox values ********
		int sel = 1;
		double scale = detWinFeat[0][3];
		int x1 = detWinFeat[0][1] * scale;
		int y1 = detWinFeat[0][2] * scale;

		int width = temp_Width * scale;
		int height = temp_Height * scale;

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
			double scale = detWinFeat[i][3];
			int pos_x = detWinFeat[i][1] * scale;
			int pos_y = detWinFeat[i][2] * scale;

			int width = temp_Width * scale; // pos_x2
			int height = temp_Height * scale;

			cv::Rect current_bBox(pos_x, pos_y, width, height);
			float current_score = detWinFeat[i][0];
			// go thrugh all best Boxes in final_bBoxes and determine if there is overlap if no add current bboxe to final_bBox array.
			// if you  have to replace more than one boxes in the final bBoxes delete 2nd, 3rd etc to be replaced boxes reducing the final_box vector size and avoiding duplicates
			for (int n = 0; n < sel; n++) {
				int b_x1 = final_BBox[n][0];
				int b_y1 = final_BBox[n][1];
				int b_x2 = final_BBox[n][2];
				int b_y2 = final_BBox[n][3];
				float stored_score = final_BBox[n][4];

				cv::Rect stored_bBox(b_x1, b_y1, b_x2 - b_x1, b_y2 - b_y1);
				cv::Rect intersect_rect =  stored_bBox & current_bBox;
				cv::Rect union_rect =  stored_bBox |  current_bBox;

				float inters = intersect_rect.area();
				float uni = union_rect.area();
				overlap = inters / uni;
				// there is a posiblity that we c#ud have more bboxes that are different.. so u have to go through them and consider only if overlap
				// that implies we dont store less than threshold overlaping bboxes in final_bBoxes
				if (overlap > 0.2) {
					addNewBox = false;
					if (del && final_BBox.size() > 1) {
						final_BBox.erase(final_BBox.begin() + n);
						sel--;
					}
					else {
						//if (current_score < stored_score) {
							// replace up to now best box
						final_BBox[n][0] = pos_x;
						final_BBox[n][1] = pos_y;
						final_BBox[n][2] = width + pos_x;
						final_BBox[n][3] = height + pos_y;
						final_BBox[n][4] = current_score;
						del = true;
						//}// else suprress this current bbox
					}
				}
			}
			// add new bBox.. cud be new pedstrian thus increase the size of final bBoxes
			if (addNewBox) {
				sel++;
				vector<float> temp;
				temp.push_back(pos_x);
				temp.push_back(pos_y);
				temp.push_back(width + pos_x);
				temp.push_back(width + pos_y);
				temp.push_back(current_score);
				final_BBox.push_back(temp);
				temp.clear();
			}
		}
	}
}

void detectionWindow_features(std::vector<std::vector<float>>& detWinFeat, int x, int y, float scale, float score) {
	vector<float> temp;
	temp.push_back(score);
	temp.push_back(x);
	temp.push_back(y);
	temp.push_back(scale);
	//temp.push_back(img_id);
	detWinFeat.push_back(temp);
	temp.clear();
}