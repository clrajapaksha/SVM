// SVM.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/features2d/features2d.hpp>

#define Num_Features 100
#define Num_Classes 2
#define NUm_Train_Img_per_class 10
#define Num_Test_Img_per_class 13
#define Descriptor_size 32

using namespace cv;
using namespace std;

void getTrainingMatrix(Mat *data, Mat *label){
	//Mat img;
	cout << "start" << endl;
	//Mat data(80, 32, CV_32FC1);
	char ch[30];
	for (int i = 1; i <= Num_Classes; i++){
		for (int j = 1; j <= NUm_Train_Img_per_class; j++){
			sprintf_s(ch, 30, "%s%d%s%d%s", "train2\\", i, " (", j, ").jpg");
			String imageName = ch;
			//img = cvLoadImage(imageName);
			Mat img = imread(imageName, 0);
			OrbFeatureDetector detector1;
			OrbDescriptorExtractor extractor1;
			vector<KeyPoint> keypoints_object;


			
			detector1.detect(img, keypoints_object);
			Mat features;
			extractor1.compute(img, keypoints_object, features);
			//Mat pcaFeatures;
			// pca(features, Mat(), CV_PCA_DATA_AS_ROW, 100);
			//pca.project(features, pcaFeatures);

			for (int row = 0; row<Num_Features; row++){
				for (int col = 0; col<Descriptor_size; col++){
					data->at<float>((i-1)*NUm_Train_Img_per_class+j - 1, row *Descriptor_size+col) = features.at<uchar>(row, col);
					//cout << row << " ";
				}
			}
			label->at<int>((i-1)*NUm_Train_Img_per_class+j-1,0) = i;
			cout << i << " " << j << endl;
			
			
		}
	}
}

void getTestImageResults(){

	CvSVM svm;
	svm.load("SVMmodel.xml");
	char ch[30];
	Mat groundTruth(0, 1, CV_32FC1);
	Mat results(0, 1, CV_32FC1);
	Mat testDescriptor(1, Num_Features * Descriptor_size, CV_32FC1);
	for (int i = 1; i <= Num_Classes; i++){
		for (int j = 1; j <= Num_Test_Img_per_class; j++){

			OrbFeatureDetector detector;
			OrbDescriptorExtractor extractor;
			vector<KeyPoint> keypoint2;
			sprintf_s(ch,30, "%s%d%s%d%s", "eval2/", i, " (", j, ").jpg");
			const char* imageName = ch;
			Mat img2 = imread(imageName, 0);

			detector.detect(img2, keypoint2);
			Mat features;
			extractor.compute(img2, keypoint2, features);

			for (int row = 0; row<Num_Features; row++){
				for (int col = 0; col<Descriptor_size; col++){
					testDescriptor.at<float>(0, row * Descriptor_size + col) = features.at<uchar>(row, col);
					//cout << row << " ";
				}
			}
			groundTruth.push_back((float)i);
			float response = svm.predict(testDescriptor);
			cout << response << endl;
			results.push_back(response);
		}

	}
	double errorRate = (double)countNonZero(groundTruth - results) / groundTruth.rows;
	printf("%s%f\n", "Error rate is ", errorRate);
}

void getTestImageResult(){

	CvSVM svm;
	svm.load("SVMmodel.xml");
	char ch[30];
	Mat groundTruth(0, 1, CV_32FC1);
	Mat results(0, 1, CV_32FC1);
	Mat testDescriptor(1, Num_Features * Descriptor_size, CV_32FC1);
	int i, j;
	cin >> i >> j;
	if (i < 0){
		return;
	}
	OrbFeatureDetector detector;
	OrbDescriptorExtractor extractor;
	vector<KeyPoint> keypoint2;
	sprintf_s(ch, 30, "%s%d%s%d%s", "eval2/", i, " (", j, ").jpg");
	const char* imageName = ch;
	Mat img2 = imread(imageName, 0);

	detector.detect(img2, keypoint2);
	Mat features;
	extractor.compute(img2, keypoint2, features);
	for (int row = 0; row<Num_Features; row++){
		for (int col = 0; col<Descriptor_size; col++){
			testDescriptor.at<float>(0, row * Descriptor_size + col) = features.at<uchar>(row, col);
			//cout << row << " ";
		}
	}

	float response = svm.predict(testDescriptor);
	cout << response << endl;
}

void train(){

	// Set up training data
	Mat labelsMat(Num_Classes*NUm_Train_Img_per_class, 1, CV_32SC1);
	Mat trainingDataMat(Num_Classes*NUm_Train_Img_per_class, Num_Features * Descriptor_size, CV_32FC1);
	getTrainingMatrix(&trainingDataMat, &labelsMat);

	// Set up SVM's parameters
	/*CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);*/

	CvSVMParams params;
	params.kernel_type = CvSVM::POLY;
	params.degree = 0.35;
	params.svm_type = CvSVM::C_SVC;
	params.gamma = 0.000000625;
	params.C = 220.50000000000000;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 0.000001);

	// Train the SVM
	CvSVM SVM;
	SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

	// Save the SVM
	SVM.save("SVMmodel.xml");
	/*for (int i = 0; i < labelsMat.rows;i++){
	cout << labelsMat.at<int>(i, 0) << " " << endl;
	}*/
}

int _tmain(int argc, _TCHAR* argv[])
{
	train();
	cout << "Processing evaluation data..." << endl;

	getTestImageResults();
	//getTestImageResult();
	
	
	cout << "OK" << endl;
	int a;
	cin >> a;
	waitKey(0);
	return 0;
}

